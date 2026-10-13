# 早期实验结果入口

当前完整研究地图和汇报报告已整理在[experiments](../experiments/README.md)。本目录保存早期精选结果，不应将此处的旧GO状态当作最新结论。

| 历史实验线 | 当前阅读方式 | 证据 |
|---|---|---|
| L64同分布生成 | 早期pilot有限物理指标通过，采样器修正历史保留；不是后期dense模型 | [机器摘要](../artifacts/final_l64/final_summary.json) |
| L64到L128直接外推 | 部分局部指标保留，但长程关联过强，作为问题起点 | [机器摘要](../artifacts/final_l128_zero_shot/final_summary.json) |
| Level1及Local–Global筛选 | 探索性开发证据；Stage2B旧native-unit对照已更正，多seed正式确认未执行 | [历史目录](scale_aware_context/README.md)、[更正](../experiments/CORRECTIONS.md) |

建议先读[历史起点解释](../experiments/HISTORY_ZH.md)，再按[精读路线](../experiments/STUDY_GUIDE_ZH.md)阅读G、I、N/W与RG。原始科学数组保留在既有路径；整理文档不代表重新训练或重新判定旧结果。
