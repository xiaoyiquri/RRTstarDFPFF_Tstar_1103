# paper_r2 float32 strict 实施说明（2026-10-07）

本次从 `RRTstarDFPFF_paper_r2_noguidance_20261006` 另建
`RRTstarDFPFF_paper_r2_float32_strict_20261007`，只修改新副本。
`tools/audit_original.py` 按复制前的 SHA256 清单检查原目录；构建、结果、Git 和 Python 缓存不属于源码清单。

## 模式与适用范围

| 配置 | 计算链 | 边界语义 |
|---|---|---|
| 主目录六个 YAML：`flask_strict` + `numeric.profile: float32_production` | 专用 r=2 float PFF、float NN、float DF、float8 validator | A′：y、dy 连续；姿态、omega 只诊断 |
| `benchmark/config/legacy_mixed/paper_r2` | 保留原 double/long-double 求解及 float8 validator | 原 r=2 A′ |
| `benchmark/config/full_df` + `legacy_mixed` | 保留原高阶计算链 | 高阶 lifting 与完整物理端点约束 |
| `double_reference` | C++ 测试内的独立参考计算 | 不进入 production 计时 |

`full_df + float32_production` 会报错。`double_reference` 不是可运行的规划模式。
`tests/double_reference.h` 要求显式 test-only 宏，防止被 production 误包含；全工程禁止 fast-math / Ofast / fp:fast，保留严格浮点语义。
这是保留 RRT* 树搜索、对齐部分 FLASK 计算语义的实现，不宣称复现 FLASK 全算法或性能。
第一阶段没有双根搜索、双 branch 森林，也没有改为基于代价半径的 Near；Near 的加权欧氏候选集及 PFF 代价排序保持原设计。

## 真正的 float 主路径

- `FlatState`、`FlatControl`、`FlatTrajectory`、`RRTNode` 和 `PlannerConfigT` 增加数值类型参数，旧模式默认仍为 double。
- `production/model.h` 为三个系统提供 float 模型，物理状态、边界 lifting、DF 参数和固定尺寸 actuation inverse 都使用 float。
- `pff/r2_float_kernel.h` 用固定尺寸 float 状态和四个归一化 cubic 系数；不调用通用 Gramian/MatrixXd/long-double 求根。
- `simd/production_avx2.cpp` 提供 float8 PFF 能量和 NN 距离；`production/float_validator.h` 将 float 系数转换为 Bernstein jet，直接接入原 float8 DF、状态/控制约束与碰撞检查。
- 树累计代价、parent ranking、rewire、采样分布和 NN 输入保持 float。wall-clock、CSV 和诊断统计允许 double。
- YAML 环境/模型读取与保守 sphere cover 属于输入阶段，仍保留 double 原始数据用于几何包络和结果诊断；这些不进入搜索计算链。
- 可执行文件仍包含 legacy 模式需要的 double 后端；“production 不调用”指模式的调用链，不指整个二进制没有 double 指令。

## 两种 r=2 闭式连接

设 `s=t/T`，轨迹为 `p0+p1*s+p2*s²+p3*s³`。
令 `δ=yf-y0`、`d=δ-T*v0`，内积 `<a,b>=aᵀRb`。

**普通 expansion：固定 yf，终端速度自由。**

```
p0=y0, p1=T*v0, p2=1.5*d, p3=-0.5*d
vf*=v0+1.5*d/T
J(T)=T+3<d,d>/T³
A=3<δ,δ>, B=-6<v0,δ>, C=3<v0,v0>
```

**goal、rewire、shortcut：固定 yf、vf。** 令 `e=vf-v0`。

```
p0=y0, p1=T*v0, p2=3*d-T*e, p3=-2*d+T*e
J(T)=T+12<d-0.5*T*e,d-0.5*T*e>/T³+<e,e>/T
A=12<δ,δ>, B=-12<v0+vf,δ>
C=4(<v0,v0>+<v0,vf>+<vf,vf>)
```

两者的时间驻点均满足 `T⁴-C*T²-2B*T-3A=0`，但 A/B/C 不同。
实现先用 `u=T/max_time` 缩放，以导数递归划分单调区间，再用 float 二分隔离根；包含近似重根候选。
比较区间内根、min/max 边界以及原有 12 个额外候选时间的真实闭式 J，按代价排序。
不使用 Ferrari 公式作为唯一求根方法，不回退到 double。浮点根的精度有限，不能把数值驻点声称为精确代数根。

归一化导数乘 `T^-d`。r=2 的 pseudo-control 为加速度，真实 cubic jerk 被保留，snap 为零。
最终验证还检查实际多项式端点与存储节点，以及实际多项式之间的 y/dy 连接误差。
Quad 的姿态和 omega 起终点及连接误差分别输出 CSV，保持 A′ 的 diagnostic-only 语义。
这类降阶轨迹不能仅凭 y/dy 拼接就宣称完整物理状态连续或可执行。

## Guidance、Unicycle 与零速检查

主模式强制 guidance_probability=0，并拒绝 region、passage、tilt hints。
关闭 Unicycle turn-limited sampling、额外 duration_factors、terminal speed/velocity hints 和 model_time_hints。
保留普通范围内随机采样及 goal sampling。

Unicycle 按起终点 signed_velocity 选择同一固定正/负 branch；每条 edge 不允许切换。
相反符号、零速或不合法的端点 lifting 会报错。默认仍为 +0.30。
独立负 branch 实验可将两端都设为合法负值；本次不自动生成两个根节点，也不自动替 Bugtrap 更换 branch。

零速证书在每次 edge 检查时运行，包含 expansion、goal、rewire、shortcut 和 final validation：

1. 由 float cubic 建立速度平方的四次多项式。
2. 近似驻点及端点只用于提前拒绝。
3. 接受必须通过带 nextafter 向外舍入的 Bernstein 区间下界，覆盖完整 `[0,1]`。
4. 最大深度 20、最大 cells 4096；无法确认时保守拒绝，记录 `speed_certificate_uncertain`。

该证书仅约束速度奇异性，不将 sampled collision 升级为连续碰撞保证。

## Geometry 与采样

内部保持 **single sphere robot + sphere obstacle union**，**8 lanes = 同一 edge 的 8 个时间点**。
不链接/调用 DynoBench runtime、FCL 或 scalar check_trajectory。

固定项目参数（`project_fixed`，不是全部都出自论文）：

| 参数 | 值 |
|---|---|
| box decomposition cell | 0.1 m |
| robot radius | Uni 0.125 / Quad2D 0.1 / Quad3D 0.25 m |
| collision margin | 0.00001 m |
| max_step | 0.005 s |
| 单条 edge max_samples | 20000 |

沿用原 box partition 产生 double reference sphere 后，`conservativeSphere` 将中心转换为 float，
半径补偿中心量化误差，并对误差运算及最终 float 半径向外舍入。原生 sphere 同样处理。
目标是 float sphere 包含对应的 double reference sphere，而不是只让 radius 自身增大。
机器人半径加 margin 也向外舍入。
时间网格额外增加一个 interval，为 float 舍入保留余量；有限采样依然可能漏掉采样点之间的碰撞。

## 搜索终止、计时与记录

`max_nodes=0`、`max_iterations=0` 表示 unlimited。正式主模式拒绝非零正常 caps。
另有 `hard_safety_node_cap=1000000`；触发时 termination_reason=safety_limit、completed_normally=false，
不计为正常实验成功。搜索 bad_alloc 同样报告 safety_limit。它不是内存字节上限。

latency 首次建立 goal 后立即保存 first path 和 J_first，退出循环，先于该 iteration 的 rewire/prune。
anytime 保存首解快照后继续，记录 incumbent cost 变化事件。

```
latency: 20 seeds (41..60), stop_at_first_solution=true, 120 s 失败保护
anytime: 20 seeds (41..60), stop_at_first_solution=false, 每 seed 120 s
```

`tools/run_six.py` 固定并核验单 CPU affinity，子进程继承；设置常用数值库线程数为 1。
正式 20/120/41 预算要求 CMakeCache 中的 Release 构建。manifest 保存 binary、配置、环境、模型、几何 SHA256、
compiler/flags/platform/CPU/affinity、所有 seeds 和项目固定参数。

- search_time：solve 及必要的 solver/planner 初始化，结果路径提取；不含最终 trajectory 验证/导出。
- first_solution_time：首次建立 goal 的搜索时刻，包含上述初始化。
- shortcut_time：shortcut 整体连接搜索和候选验证。
- total_time：search_time + shortcut_time。
- final_validation_time、export_time：另外记录，不加到 FLASK-style total。
- raw/shortcut cost、length、duration 分别保存，length/duration 仅为指标；优化目标一直是 `J=T+∫wᵀRw dt`。

Wall-clock 在一个局部检查结束后才再次检查 timeout，因此可能有少量超时；外层 watchdog 超时后会终止子进程并记录 124。
失败/异常不生成虚构成功轨迹，summary.csv 保留失败。退出 2 表示至少一项没有全部成功，不能当成构建错误。

## Algorithm 4 风格 shortcut

```
postprocess:
  shortcut: true
  mode: flask             # 或 cost_decreasing
  time_mode: original_duration  # 或 optimal
```

按前端 path node 和后端远节点尝试连接；使用 fixed-terminal PFF + 同一个 DF/float8 validator。
original_duration 用替换段累计 T；optimal 用原 free-time 候选集合。
flask 只要求有效；cost_decreasing 还要求 J 严格降低（含 float 容差）。
shortcut 只改路径副本，不改树。最终复检失败会退回 raw 并记录 shortcut_fallback。

输出包括 `trajectory_raw_N.yaml`、`trajectory_shortcut_N.yaml`（开启时）、`trajectory_first_N.yaml`、
兼容的 `trajectory_N.yaml`、对应 boundary diagnostics、anytime_N.csv、results.csv、validation.csv、geometry.yaml。
legacy 模式暂不执行此新 shortcut，脚本明确显示 disabled。

## 显示与碰撞完全解耦

`plot_successful_results_paper.py` 支持：

- dynobench（默认）：直接加载官方 Unicycle1Viewer / Quad2dViewer / Quad3dViewer，调用 view_problem 绘制原始环境，再叠加本项目轨迹。
- checker_geometry：旧灰色 sphere cover 的调试图；3D 使用 sphere 中心云。
- trajectory_only：只绘轨迹。

`--trajectory final/raw/shortcut/first` 可选择路径版本；`--skip-failed` 跳过未成功的 case。
官方模式缺少依赖或源码会明确报错，不自动换成重写的障碍物绘制。
依赖见 requirements-viewer.txt；运行与验证记录见运行指令及 verification_report.md。
