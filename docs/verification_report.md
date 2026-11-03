# 验证记录

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
| production_latency_delivery | 6/6 | 3.0 s | float32_production |
| production_anytime_delivery | 6/6 | 2.0 s | float32_production |
| legacy_r2_smoke | 3/3 | 1.0 s | legacy_mixed |
| legacy_full_smoke | 3/3 | 1.0 s | legacy_mixed |

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
