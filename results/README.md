# 实验结果索引

本仓库按研究问题分隔实验，避免把早期基线与后续方法验证混为一谈。

| 实验线 | 状态 | 结果入口 | 说明 |
|---|---|---|---|
| `L=64` 临界 Ising 离散扩散基线 | 已完成 | [`../artifacts/final_l64/`](../artifacts/final_l64/) | 同分布生成、采样器消融与物理量评估 |
| `L=64 → L=128` 直接零样本外推 | 已完成，结论 `PARTIAL` | [`../artifacts/final_l128_zero_shot/`](../artifacts/final_l128_zero_shot/) | 冻结基线模型，检验更大晶格上的有限尺寸 scaling |
| 尺度感知坐标与 Local--Global 因果实验 | 已完成单 seed 筛选，待多 seed 确认 | [`scale_aware_context/`](scale_aware_context/)；[完整中文报告](../docs/SCALE_AWARE_CONTEXT_EXPERIMENT_REPORT_ZH.md) | Level 1、Stage 2A 与 Stage 2B；验证 matched physical coordinates 的独立贡献 |

`artifacts/` 保留早期与基线实验的既有组织；`results/` 收录后续研究线的精选、
可公开复核结果。完整训练缓存、checkpoint 和大规模父场数据不重复提交。
