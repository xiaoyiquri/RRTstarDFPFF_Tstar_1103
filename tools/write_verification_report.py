"""Summarize saved verification evidence in the new copy."""
import csv
import hashlib
import json
import platform
from datetime import datetime,timezone
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    binary=root/"build_float32/benchmark.exe"
    current_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
    experiments={};table=[]
    for name in ("production_latency_delivery","production_anytime_delivery","legacy_r2_smoke","legacy_full_smoke"):
        folder=root/"results"/name;m=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        rows=[]
        for case in m["cases"]:
            with (folder/case/"results.csv").open(newline="",encoding="utf-8") as f:
                rows.extend(dict(case=case,**r) for r in csv.DictReader(f))
        count=sum(r["success"]=="1" and r["completed_normally"]=="1" for r in rows)
        experiments[name]=dict(protocol=m["protocol"],numeric_profile=m["numeric_profile"],df_mode=m["df_mode"],
            runs=len(rows),successes=count,timeout=m["timeout"],seeds=m["seeds"],affinity=m["affinity"],
            binary_sha256=m["binary_sha256"],matches_current_binary=m["binary_sha256"]==current_hash)
        assert m["binary_sha256"]==current_hash
        table.append(f"| {name} | {count}/{len(rows)} | {m['timeout']} s | {m['numeric_profile']} |")
    integrity=json.loads((root/"docs/original_integrity_check.json").read_text(encoding="utf-8"))
    ctest=(root/"docs/ctest_final.txt").read_text(encoding="utf-8-sig")
    assert "100% tests passed out of 16" in ctest and integrity["original_unchanged"]
    report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        binary_sha256=current_hash,cpp_tests_passed=16,python_tests_passed=5,
        original_integrity=integrity,experiments=experiments,ubuntu_tested=False,formal_20_seeds_executed=False)
    (root/"docs/verification_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    text="""# 验证记录

在本机 Windows x86-64 上使用离线 Zig/Clang、Eigen 3.4.0、yaml-cpp 0.8.0、Release `-O3 -DNDEBUG` 和 AVX2 完成验证。
精确构建和 CPU 信息、affinity、所有输入/几何/可执行文件 SHA256 见对应结果 manifest。
本记录汇总最新交付可执行文件；机器/构建不同不能直接比较这些短时 wall-clock 数值。

## 自动检查

- 16/16 C++ 测试通过，原始日志：ctest_final.txt。
- 5/5 Python 编排测试通过，日志：python_tests_final.txt。
- 数学对照覆盖终端速度自由/固定、非对角 SPD R、时间边界与重根、归一化 cubic 导数、AVX2 partial batches 和 float NN tail。
- 物理对照覆盖 Quad2D、Quad3D 和 Unicycle 正/负 branch 的 float DF 状态/控制与独立 double 参考；检查 jerk 非零、snap=0 和实际局部采样步长。
- 覆盖隐藏内部零速、near-zero、证书预算耗尽保守拒绝，及 float sphere 包含 double reference sphere（包含大坐标的中心舍入误差）。
- 覆盖首解无 post-goal rewire、first path 快照、正常 caps=0、timeout 和 emergency safety_limit。
- shortcut 测试区分 valid-only 与 cost_decreasing，检查不修改 raw、无效连接拒绝和关闭时零 shortcut_time。

## 短时集成运行

每个场景仅 seed=41，实际 affinity=[0]；不是正式 20 seeds 实验。
success 指同一 float8 checker 下的 sampled success，不代表 DynoBench/FCL 或连续碰撞认证。

| 数据目录（results 下） | sampled success | 每场景保护/预算 | 数值模式 |
|---|---:|---:|---|
"""+"\n".join(table)+"""

production 两套六场景记录已由 tools/verify_results.py 核验：模式/seed/计时等式、正常终止、首解快照、anytime 事件、y/dy 边界容差和 raw/shortcut 文件均符合记录契约。
latency 首解均在保护时间内找到；anytime 均由 timeout 正常终止。
legacy 对照只选 Uni Wall、Quad2D Hole、Quad3D Block，确认两套原模式仍可运行，shortcut 显式 disabled。

## 可视化与原目录完整性

- 六个场景已使用 DynoBench 官方 RobotViewer 体系绘制原始环境，叠加最终轨迹。
- 图集：../results/production_latency_delivery/figures_official/index.html。
- 另验证 checker_geometry/raw 和 trajectory_only/first：../results/latency_smoke_final/figures_checker/ 与 ../results/anytime_smoke_final/figures_first_only/。
- 已实际查看 Unicycle Bugtrap 和 Quad3D Window 图，障碍物显示来自原始环境。
- 原 paper_r2_noguidance 目录的 219 个受检文件无改动、无新增、无缺失，见 original_integrity_check.json。

## 尚未执行的实验与语义限制

尚未在 Ubuntu 主机编译/运行，未执行正式 20 seeds × 120 s。
完整 Ubuntu 指令在 ../运行指令.md。
paper_r2 保持 A′：y/dy 连接受检，Quad 姿态与 omega 只诊断；部分 Quad 轨迹 physical_boundary_match=0 已如实记录，仍属于允许的降阶结果。
这些轨迹不能直接被当成完整物理连续或真实可执行的验证结果。
碰撞仍是 sampled sphere-union checking；未实施双 branch 森林，高阶 full_df 没有改成全 float。

机器可读汇总：verification_results.json。
"""
    (root/"docs/verification_report.md").write_text(text,encoding="utf-8")
    print(json.dumps(dict(cpp=16,python=5,verified_experiments=len(experiments),original_unchanged=True)))

if __name__=="__main__": main()
