# 实验产物索引

本目录按实验阶段保存原始数值、图像和日志。阅读时建议从“正式结果”开始，
不要把早期 smoke test 当成最终模型表现。

## 1. 正式 `L=64` 结果

- `final_l64/`：最终汇总数据、8 组核心图、CSV 表格及正式采样原始数组。
- `pilot_l64/`：训练历史、日志、数据 QA、里程碑评估、不同 reverse steps
  和 3 个正式采样 seed 的结果。
- `scaling_fit_l64/`：`G(r)` 幂律 scaling 拟合。
- `finite_size_scaling/`：有限尺寸磁化诊断。

建议首先查看：

1. `final_l64/final_summary.json`：机器可读的最终结论；
2. `final_l64/figures/01_training_curves.png`：训练与验证曲线；
3. `final_l64/figures/06_correlation_and_structure.png`：关联函数与结构因子；
4. `final_l64/figures/08_seed_stability.png`：采样 seed 稳定性。

## 2. Monte Carlo 数据诊断

- `data_preview_l32_pilot/`：最初的 `L=32` 数据预览。
- `data_preview_l32_pilot_fixed/`：修正后的 `L=32` 数据预览。
- `data_preview_smoke/`：最小 smoke 数据预览。
- `l256_preview/`、`l256_sampling_diagnosis/`：`L=256` 平衡态和关联函数诊断。
- `peer_wolff_l256/`：同学 Wolff 生成机制的只读对照结果。

## 3. 采样器开发与稳定性实验

- `smoke_eval*`：ancestral、confidence、corrector、backtracking 等早期消融。
- `stability_*`：不同采样 seed 下的稳定性检查。
- `step015000_*`、`step030000_*`：训练中期 checkpoint 的采样快照。

这些目录用于解释为什么最终选择 posterior-consistent ancestral sampler，
不应作为正式性能数字引用。

## 4. 大文件

7 个约 203 MB 的 `.pt` 文件未作为普通 Git 对象提交，而是发布在 GitHub
Release `l64-pilot-checkpoints`。恢复路径和 SHA-256 见根目录
[`CHECKPOINTS.md`](../CHECKPOINTS.md)。

