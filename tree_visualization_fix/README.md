# 完整搜索树与部分失败后继续绘图

基于仓库 master 提交 642548e92c4de3678eccf28602241002d351cf82。

## 安装

下载补丁压缩包，解压后，在你自己的项目根目录执行（把补丁路径换成实际路径）：

```bash
git apply --check /path/to/tree_visualization_fix/changes.patch
git apply /path/to/tree_visualization_fix/changes.patch
```

先检查、再应用。如果你本地已修改相同代码，检查可能失败：不要强制覆盖；按压缩包 `files/` 中的对应文件手动合并。没有使用 Git 管理本地副本时，也可以先备份，再将 `files/` 内容按相同相对路径复制进去。

C++ 修改必须重新编译：

```bash
cmake --build build --target benchmark -j 4
```

## 一键运行

激活已安装 requirements-viewer.txt 的 Python 环境，再运行：

```bash
export DYNOBENCH_DIR="/实际路径/dynobench-main"
export BUILD_DIR="$PWD/build"
export PYTHON_BIN="$(command -v python3)"
export SEARCH_SECONDS=120
bash run_three_stage.sh
```

也支持 `PROJECT_DIR`、`SEED`（默认 20）和 `OUTPUT_DIR`（必须是新目录）。没有设置 BUILD_DIR 时，依次找 build、build_float32、build_ubuntu_float32。默认搜索预算明确为 120 秒，不含导出、shortcut、绘图。

脚本读取当前 run_six.py 真正使用的 `benchmark/config/<case>.yaml`，打印并保留各场景原有的 additional_time_samples，不把它们强制改成 12。

## 新生成的树数据

每个正常返回的 production 搜索，在 planner 销毁前输出：

```
results/<实验>/<场景>/search_tree_0.json
```

保存所有当前保留节点的 ID、最终父节点 ID、位置、累计代价，以及每条实际 PFF 入边的持续时间、归一化标记、多项式系数。包括成功路径之外的分支。根节点没有入边。没有找到目标但搜索正常返回时，也会导出树；本绘图流程仍只画成功轨迹。

导出位于 solve() 后，不进入 search_time，耗时记入 export_time。磁盘导出异常会告警，不把规划成功改写为规划失败；此时 `--show-tree` 会明确报告树文件缺失。

这张树是**搜索结束时的快照**，不是搜索历史中曾经尝试的所有边；被拒绝、被 rewiring 替换或被 prune 删除的旧边不在其中。三阶段图均叠加同一个最终树，标题和图例注明 final search tree，不能把首解图上的背景误认为首解时刻的树。

旧结果目录没有树数据，无法仅用成功轨迹还原整棵树：需要重新运行修改后的 benchmark。

## 手动画树与成功轨迹

```bash
MPLBACKEND=Agg python3 plot_successful_results_paper.py \
  --experiment "/实际路径/结果目录" \
  --dynobench-source "$DYNOBENCH_DIR" \
  --mode dynobench --trajectory shortcut --run 0 \
  --show-tree --tree-samples 33 --skip-failed \
  --output "/实际路径/结果目录/figures_tree_new"
```

`--trajectory` 可选 first、first_shortcut、raw、shortcut、final。灰色显示树边和节点，蓝色显示当前选中的成功轨迹。二维与三维均支持，Quad3D 只绘空间 xyz，不把 yaw 当空间坐标。

所有保留树边都会绘制，无边数截断。每条 PFF 多项式默认采样 33 点显示，可调整 --tree-samples（2..4096）；它仅控制显示精度，不改变规划或碰撞检查。树很大时文件和绘图开销也会增加。采样画出的曲线是视觉近似，不是新的碰撞验证。

旧结果只画成功轨迹时，省略 `--show-tree`。`--skip-failed` 跳过无 results.csv、所选 run 失败、所选轨迹缺失的场景；不会掩盖缺少 Viewer、损坏树文件等绘图错误。

## 部分失败的退出行为

规划返回非零后，脚本先记录状态，再尝试三组图。某组绘图失败也会继续下一组。最后才返回非零：优先保留规划退出码，否则返回绘图错误码。因此 shell/CI 仍能识别实验失败，同时你能查看已生成的成功图片。

编译或输入检查失败仍立即退出；Ctrl+C（130）和终止信号状态（143）也不会触发后续绘图。没有成功轨迹时不声称生成了 index.html。

例如六个场景中两项失败，脚本最终可能仍返回 2，这是正确行为：请到结果目录查看成功场景的图片，而不是把返回 2 理解成所有图都没有生成。

## 验证

```bash
python3 -m unittest discover -s tests -p test_tree_visualization.py -v
python3 -m unittest discover -s tests -p test_run_six.py -v
bash -n run_three_stage.sh
```

新测试覆盖二维/三维全部树边绘制、多项式曲线（而非端点直线）、缺失树文件、成功/失败/缺失场景混合、规划非零后继续绘图、绘图失败后继续下一组、全部失败以及构建失败/中断退出。测试中的合成场景不是 DynoBench 六场景实验成绩。
