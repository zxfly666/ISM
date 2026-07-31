# 实验产物索引

## L=128 零样本尺度外推

- `final_l128_zero_shot/`：三 seed 聚合后的 JSON、CSV 和 6 组 PNG/PDF 图表；总体结论为 `PARTIAL`。
- `l128_zero_shot/formal_s*/`：逐 seed 指标和快速对比图；原始 `samples.npz` 仅保存在本地，不进入 Git。
- `l128_zero_shot/formal_*.log`：后台采样、补跑和自动分析的审计日志。
- 正式合并样本 `model_samples_4608.npz` 仅保存在本地；可用仓库脚本和 `best.pt` 复现。

建议首先阅读 `final_l128_zero_shot/final_summary.json`，然后查看
`figures/03_correlation_and_structure.png` 与 `figures/06_cross_scale_scaling.png`。

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

正式实验的 `best.pt` 使用 Git LFS 管理，clone 后会恢复到原始项目路径。
`last.pt` 和里程碑权重不上传；路径和 SHA-256 说明见根目录
[`CHECKPOINTS.md`](../CHECKPOINTS.md)。
