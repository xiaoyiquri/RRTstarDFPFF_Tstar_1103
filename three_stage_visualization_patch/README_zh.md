# 同一棵 RRT* 搜索树的三阶段轨迹与 Cost 可视化

适用于 `RRTstarDFPFF_paper_r2_float32_strict_20261007`。本方案只修改 `benchmark/production_runner.cpp` 和 `plot_successful_results_paper.py`，不修改 PFF、DF、SIMD、搜索树或 Rewire 逻辑。保留你本地绘图脚本已经添加的机器人姿态绘制代码。

## 1. 安装补丁（Ubuntu）

先将本文件夹里的 `enable_three_stage.py` 和 `run_three_stage.sh` 放入 Ubuntu（例如 `~/Downloads`），然后执行：

```bash
PROJECT="/home/cxt/MYFILE/newfile/RRTstar/RRTstarDFPFF_paper_r2_float32_strict_20261007"
python3 ~/Downloads/enable_three_stage.py "$PROJECT"
```

补丁会在原文件旁保留 `*.backup_three_stage` 备份。重复执行是安全的。**C++ 有变化，因此要重新编译**；运行脚本会自动尝试编译 `benchmark`。

## 2. 保留你之前的 PFF 额外时间候选 12

如果本地 `run_six.py` 的 `--help` 不包含 `--time-policy` 和 `--additional-time-samples`，请明确将 `benchmark/config/anytime/` 下六个 YAML 的 `additional_time_samples` 设为 12；若包含 `time_policy`，必须设为 `candidate_set`。

```bash
for cfg in "$PROJECT"/benchmark/config/anytime/*.yaml; do
  cp -n "$cfg" "$cfg.backup_three_stage"
  sed -i -E \
    -e 's/^(  additional_time_samples:).*/\1 12/' \
    -e 's/^(  time_policy:).*/\1 candidate_set/' \
    "$cfg"
done
```

补丁本身不自动改 YAML。运行脚本会检查本地六个 anytime YAML 是否已使用 12 个额外时间候选，防止静默改变实验条件。新版本 runner 如支持这两个命令行选项，则自动显式传入。

## 3. 运行（六环境，各 120 秒；seed 20）

```bash
bash ~/Downloads/run_three_stage.sh 2>&1 | tee "$HOME/three_stage_run.log"
echo "script exit: ${PIPESTATUS[0]}"
```

单核运行六个环境，每个环境运行到 120 秒上限或其他终止条件；先找到首解，继续 Rewire；搜索结束后分别对首解副本和 120 秒最终原始路径执行相同的 `cost_decreasing` / `optimal` Shortcut。

每个环境输出：

- `trajectory_first_0.yaml` — 首解 Raw
- `trajectory_first_shortcut_0.yaml` — **同一个首解**的 Shortcut
- `trajectory_raw_0.yaml` — 120 秒 RRT* 原始结果（额外保留，用于隔离 Rewire）
- `trajectory_shortcut_0.yaml` — 120 秒搜索后的 Shortcut
- `first_shortcut_metrics_0.csv` — 首解 Shortcut 的耗时、次数、Cost、长度和是否有效
- `results.csv` — 首解 Cost、120 秒 Raw Cost、120 秒 Shortcut Cost 与 rewires 等

绘图同时生成：

- `figures_01_first_raw/`
- `figures_02_first_shortcut/`
- `figures_03_anytime_shortcut/`

每张 PNG 的标题会包含对应的 **LQMT Cost J** 和几何长度 L（米）。J 最小化平坦伪控制能量与时间之和，不是几何路径长度。`first_shortcut` 后处理是**搜索结束后**针对保留下来的首解副本进行的，绝不会干扰 120 秒搜索随机序列；所以 `results.csv` 的 `total_time` 仍按原代码表示搜索 + 120 秒最终路径 Shortcut，不包含额外的首解 Shortcut。`first_shortcut_metrics_0.csv` 独立记录该耗时。

本方案不改变碰撞验证的物理依据：仍然是项目的 `simd_discrete_float8`，不是 DynoBench/FCL 独立连续验证。
