#!/usr/bin/env python3
"""Single-core latency/anytime orchestration; production and legacy are distinct."""
import argparse
import csv
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

CASES = {
    "uni_bugtrap": ("unicycle1_v0/bugtrap_0.yaml", "unicycle1_v0"),
    "uni_wall": ("unicycle1_v2/wall_0.yaml", "unicycle1_v2"),
    "quad2d_bugtrap": ("quad2d_v0/quad_bugtrap.yaml", "quad2d_v0"),
    "quad2d_hole": ("quad2d_v0/fall_through.yaml", "quad2d_v0"),
    "quad3d_block": ("quadrotor_v0/quad_one_obs.yaml", "quad3d_v0"),
    "quad3d_window": ("quadrotor_v0/window.yaml", "quad3d_v0"),
}
PAPER_MODELS = {"unicycle1_v0", "unicycle1_v2", "quad3d_v0"}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace(text, key, value):
    result, count = re.subn(r"(?m)^  " + re.escape(key) + r":.*$",
                            lambda _: "  " + key + ": " + value, text)
    if count != 1:
        raise ValueError("Expected exactly one configuration key: " + key)
    return result

def set_affinity(cpu):
    if hasattr(os, "sched_getaffinity"):
        allowed=sorted(os.sched_getaffinity(0)); selected=allowed[0] if cpu is None else cpu
        if selected not in allowed: raise ValueError("CPU outside allowed affinity mask")
        os.sched_setaffinity(0,{selected}); actual=sorted(os.sched_getaffinity(0))
    elif sys.platform == "win32":
        api=ctypes.WinDLL("kernel32",use_last_error=True)
        api.GetCurrentProcess.restype=ctypes.c_void_p
        api.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        api.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        process=api.GetCurrentProcess(); current,system=ctypes.c_size_t(),ctypes.c_size_t()
        if not api.GetProcessAffinityMask(process,ctypes.byref(current),ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        allowed=[k for k in range(ctypes.sizeof(current)*8) if current.value & (1<<k)]
        selected=allowed[0] if cpu is None else cpu
        if selected not in allowed: raise ValueError("CPU outside allowed affinity mask")
        if not api.SetProcessAffinityMask(process,1<<selected): raise ctypes.WinError(ctypes.get_last_error())
        if not api.GetProcessAffinityMask(process,ctypes.byref(current),ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        actual=[k for k in allowed if current.value & (1<<k)]
    else: raise RuntimeError("Cannot enforce single-core affinity on this platform")
    if actual != [selected]: raise RuntimeError("Single-core affinity verification failed")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    return actual

def execute(command,log,timeout):
    with log.open("w",encoding="utf-8") as stream:
        stream.write("COMMAND: "+json.dumps([str(x) for x in command])+"\n");stream.flush()
        try:
            return subprocess.run([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=timeout,check=False).returncode
        except subprocess.TimeoutExpired:
            stream.write("\nTIMEOUT: process terminated\n");return 124

def build_metadata(build):
    cache=build/"CMakeCache.txt";values={}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("//","#")) and "=" in line and ":" in line.split("=",1)[0]:
                key,value=line.split("=",1);values[key.split(":",1)[0]]=value
    return {k:values.get(k,"unavailable") for k in (
        "CMAKE_BUILD_TYPE","CMAKE_CXX_COMPILER","CMAKE_CXX_FLAGS_RELEASE","RRTSTARDFPFF_ENABLE_AVX2")}

def main():
    project=pathlib.Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynobench-source",type=pathlib.Path,required=True)
    parser.add_argument("--build-dir",type=pathlib.Path,default=project/"build_float32")
    parser.add_argument("--output",type=pathlib.Path)
    parser.add_argument("--runs",type=int,default=20)
    parser.add_argument("--timeout",type=float,default=120)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--protocol",choices=("latency","anytime"),default="latency")
    parser.add_argument("--first-solution",action="store_true",help="Compatibility alias for latency")
    parser.add_argument("--cpu",type=int)
    parser.add_argument("--df-mode",choices=("paper_r2","full_df"),default="paper_r2")
    parser.add_argument("--numeric-profile",choices=("float32_production","legacy_mixed"))
    parser.add_argument("--shortcut",choices=("true","false"),default="true")
    parser.add_argument("--shortcut-mode",choices=("flask","cost_decreasing"),default="flask")
    parser.add_argument("--shortcut-time",choices=("original_duration","optimal"),default="original_duration")
    parser.add_argument("--cases",nargs="+",choices=list(CASES),default=list(CASES))
    args=parser.parse_args()
    if args.runs<1 or not 0<args.timeout<float("inf") or not 0<=args.seed<=4294967295-args.runs+1:
        parser.error("Invalid runs, timeout or seed")
    if args.first_solution and args.protocol!="latency": parser.error("--first-solution conflicts with anytime")
    numeric=args.numeric_profile or ("legacy_mixed" if args.df_mode=="full_df" else "float32_production")
    if args.df_mode=="full_df" and numeric=="float32_production":
        parser.error("full_df remains legacy_mixed; production float32 is paper_r2 only")
    source,build=args.dynobench_source.resolve(),args.build_dir.resolve()
    binary=build/("benchmark.exe" if sys.platform=="win32" else "benchmark")
    if not binary.is_file(): parser.error("Missing executable: "+str(binary))
    selected=list(dict.fromkeys(args.cases))
    for case in selected:
        env,model=CASES[case]
        root=project/"benchmark/models/paper" if model in PAPER_MODELS else source/"models"
        for file in (source/"envs"/env,root/(model+".yaml")):
            if not file.is_file(): parser.error("Missing YAML: "+str(file))
    build_info=build_metadata(build)
    formal=args.runs==20 and args.timeout==120 and args.seed==41
    if formal and build_info["CMAKE_BUILD_TYPE"]!="Release": parser.error("Formal budget requires verified Release build")
    affinity=set_affinity(args.cpu)
    output=(args.output or project/"results"/datetime.datetime.now().strftime("strict_%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(validation_backend="simd_discrete_float8",geometry="sphere_union",
        profile="flask_strict" if numeric=="float32_production" else "paper_fair", numeric_profile=numeric,
        df_mode=args.df_mode,protocol=args.protocol,
        boundary_contract="reduced_flat_A_prime" if args.df_mode=="paper_r2" else "full_physical",
        solver_precision="float32" if numeric=="float32_production" else "float64",validation_precision="float32",
        time_root_precision="float32" if numeric=="float32_production" else "long_double",
        external_checks=0,dynobench_verified=False,continuous_collision_certified=False,
        runs=args.runs,timeout=args.timeout,seed=args.seed,seeds=list(range(args.seed,args.seed+args.runs)),
        cases=selected,first_solution=args.protocol=="latency",formal_budget=formal,platform=platform.platform(),
        cpu=platform.processor(),affinity=affinity,dynobench_source=str(source),build=build_info,
        binary_sha256=digest(binary),shortcut_enabled=args.shortcut=="true" and numeric=="float32_production",
        shortcut_mode=args.shortcut_mode,shortcut_time_mode=args.shortcut_time,
        project_fixed=dict(obstacle_sphere_cell=.1,robot_radius=dict(unicycle=.125,quad2d=.1,quad3d=.25),
            collision_margin=.00001,max_step=.005,rounding="radius_compensates_center_error_outward"),
        inputs={},environment_files={},model_files={})
    summaries=[]
    print(f"{numeric}, {args.protocol}, affinity={affinity}; sampled SIMD validation",flush=True)
    if numeric=="legacy_mixed": print("Legacy comparison: shortcut unavailable and disabled",flush=True)
    for case in selected:
        env,model=CASES[case];root=project/"benchmark/config/anytime"
        if args.df_mode=="full_df": root/= "full_df"
        elif numeric=="legacy_mixed": root/= "legacy_mixed/paper_r2"
        original=root/(case+".yaml")
        model_file=(project/"benchmark/models/paper" if model in PAPER_MODELS else source/"models")/(model+".yaml")
        text=original.read_text(encoding="utf-8")
        replacements={"source":json.dumps(source.as_posix()),"output_directory":json.dumps((output/case).as_posix()),
            "runs":str(args.runs),"timeout":str(args.timeout),"seed":str(args.seed),
            "stop_at_first_solution":str(args.protocol=="latency").lower()}
        if model in PAPER_MODELS: replacements["model_file"]=json.dumps(model_file.resolve().as_posix())
        if numeric=="float32_production":
            replacements.update(shortcut=args.shortcut,time_mode=args.shortcut_time)
            text=re.sub(r"(?m)(^postprocess:\n(?:  .*\n)*?  mode:).*$",r"\g<1> "+args.shortcut_mode,text)
        for key,value in replacements.items(): text=replace(text,key,value)
        config=output/(case+".yaml");config.write_text(text,encoding="utf-8")
        manifest["inputs"][case]={str(f):digest(f) for f in (original,source/"envs"/env,model_file,config)}
        manifest["environment_files"][case]=str(source/"envs"/env);manifest["model_files"][case]=str(model_file.resolve())
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        print(case+": starting",flush=True)
        code=execute([binary,"--config",config],output/(case+".log"),args.runs*(args.timeout+30)+30)
        records=[];results=output/case/"results.csv"
        if results.is_file():
            with results.open(newline="",encoding="utf-8") as stream: records=list(csv.DictReader(stream))
        successes=sum(r.get("success")=="1" and r.get("completed_normally","1")=="1" and
            r.get("validation_backend")=="simd_discrete_float8" for r in records)
        summaries.append(dict(environment=case,runs=args.runs,sampled_successes=successes,
            external_checks=0,return_code=code,validation_backend="simd_discrete_float8"))
        geometry=output/case/"geometry.yaml"
        if geometry.exists(): manifest["inputs"][case][str(geometry)]=digest(geometry)
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        with (output/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        print(f"{case}: sampled success {successes}/{args.runs}, exit={code}",flush=True)
    print("Results: "+str(output),flush=True)
    return 0 if all(s["return_code"]==0 and s["sampled_successes"]==s["runs"] for s in summaries) else 2

if __name__=="__main__": sys.exit(main())
