"""Check recorded production result contracts; does not redo collision checks."""
import argparse
import csv
import json
import math
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",type=Path)
    args=parser.parse_args();root=args.experiment
    m=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    assert m["numeric_profile"]=="float32_production"
    assert m["solver_precision"]==m["time_root_precision"]=="float32"
    assert len(m["affinity"])==1 and m["external_checks"]==0
    checked=successes=0
    for case in m["cases"]:
        with (root/case/"results.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        assert len(rows)==m["runs"]
        for r in rows:
            checked+=1;run=int(r["run"])
            assert int(r["seed"])==m["seeds"][run]
            assert r["numeric_profile"]=="float32_production" and r["solver_precision"]==r["time_root_precision"]=="float32"
            assert r["boundary_contract"]=="reduced_flat_A_prime" and r["pseudo_control_order"]=="2"
            assert r["dynobench_validations"]=="0" and r["validation_backend"]=="simd_discrete_float8"
            search,shortcut,total=map(float,(r["search_time"],r["shortcut_time"],r["total_time"]))
            assert math.isclose(total,search+shortcut,rel_tol=1e-12,abs_tol=1e-12)
            assert float(r["final_validation_time"])>=0 and float(r["export_time"])>=0
            if r["success"]!="1": continue
            successes+=1;assert r["completed_normally"]=="1"
            assert float(r["first_solution_time"])<=search
            assert r["termination_reason"]==("first_solution" if m["protocol"]=="latency" else "timeout")
            if m["protocol"]=="latency": assert r["rewires"]=="0"
            for kind in ("raw","first"):
                assert (root/case/f"trajectory_{kind}_{run}.yaml").is_file()
                assert (root/case/f"boundary_diagnostics_{kind}_{run}.csv").is_file()
            if r["shortcut_enabled"]=="1": assert (root/case/f"trajectory_shortcut_{run}.yaml").is_file()
            with (root/case/f"anytime_{run}.csv").open(newline="",encoding="utf-8") as f: events=list(csv.DictReader(f))
            assert events and math.isclose(float(events[0]["cost"]),float(r["first_solution_cost"]),rel_tol=1e-6)
            for a,b in zip(events,events[1:]):
                assert float(a["search_time"])<=float(b["search_time"]) and float(b["cost"])<float(a["cost"])
            with (root/case/f"boundary_diagnostics_{run}.csv").open(newline="",encoding="utf-8") as f:
                for boundary in csv.DictReader(f):
                    assert float(boundary["flat_position_error"])<=.001 and float(boundary["flat_velocity_error"])<=.001
    print(json.dumps(dict(checked_runs=checked,sampled_successes=successes,contract_checks="passed")))

if __name__=="__main__": main()
