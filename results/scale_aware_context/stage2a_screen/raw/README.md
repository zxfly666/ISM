# Stage 2A 原始评估数据

本目录保存 Stage 2A 的完整评估中间量：

- `coordinates/`：2048 个基础样本 × 4 diffusion times 的配对坐标 probe；
- `markov_curve/`：`W={8,16,24,32,48,64}` 下的逐样本 Markov 污染曲线；
- `generation/`：生成样本、关联函数与物理统计；
- `stage0/`：冻结采样器选择；
- `final/`：由上述原始数据生成的聚合判定和图表。

大体积 CSV/NPZ 使用 Git LFS。阅读结论优先看上一级
[`stage2a_decision.json`](../stage2a_decision.json)，复核统计时再使用本目录。

Stage 0 的 `manifest.json` 已内嵌全部 72 个模型/采样器/步数/seed 记录，可直接由
`analyze_stage0_sampler.py` 重新排名。为保持仓库清晰，未重复上传不参与后处理的
72 组逐 run 样本数组及其重复 JSON 副本。
