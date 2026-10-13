# 技术文档与历史材料索引

当前汇报与科学结论从[综合报告](../experiments/REPORT_ZH.md)进入。本目录保留技术设计、原始方案和历史报告；文件的旧状态标签不自动代表当前结论。

## 准备汇报先看这些

- [精读与汇报路线](../experiments/STUDY_GUIDE_ZH.md)：全部实验价值、阅读优先级和40分钟组织方式。
- [共同方法与指标](../experiments/METHODS_ZH.md)：尺寸、距离、跨度、clock、MASK/PAD、CE/KL和独立重复。
- 四个重点：[G](../experiments/size-spacing-factorial/REPORT_ZH.md)、[I](../experiments/fine-geometry-identification/REPORT_ZH.md)、[N/W](../experiments/canonical-context-size-generalization/REPORT_ZH.md)、[RG](../experiments/rg-coarse-data-training-transfer/REPORT_ZH.md)。
- [常见追问](../experiments/QUESTIONS_ZH.md)与[历史更正](../experiments/CORRECTIONS.md)。

## 早期模型与方案

| 文档 | 适用范围 |
|---|---|
| [L64实现说明](L64_IMPLEMENTATION_GUIDE.md) | 约1270万参数轴向模型的结构、目标及早期工程命令，不是后期dense模型 |
| [TECHNICAL_DESIGN_ZH](TECHNICAL_DESIGN_ZH.md) | 早期离散扩散、采样器和物理评价设计 |
| [L64 pilot方案](L64_PILOT_EXPERIMENT_PLAN_ZH.md) | 原同分布基线的流程和判据 |
| [L128零样本外推方案](L128_ZERO_SHOT_PROTOCOL_ZH.md) | 冻结L64权重扩大晶格的评价，不是新训练 |
| [Level1结果](LEVEL1_RAPID_RESULT_REPORT_ZH.md)与[方案](LEVEL1_RAPID_GO_NOGO_PROTOCOL_ZH.md) | 历史dense尺度筛选及局部/长程tradeoff |
| [Stage2综合报告](SCALE_AWARE_CONTEXT_EXPERIMENT_REPORT_ZH.md) | 2026年8月历史稿，必须先读Stage2B对照更正，不作为当前确认结论 |
| [Local–Global方案](LEVEL2_SCALE_COORDINATE_LOCAL_GLOBAL_PROTOCOL_ZH.md) | 当时设计与计划；多seed正式确认没有执行 |
| [实现记录](STAGE2_IMPLEMENTATION_LOG_ZH.md) | 历史代码与开发过程 |
| [方案审阅回应](PROTOCOL_REVIEW_RESPONSE_ZH.md) | 当时的意见处理，不替代后续结果 |

## 其他技术材料

[实验日志](EXPERIMENT_LOG.md)、[L256 MC诊断](L256_MC_DIAGNOSIS_ZH.md)、[同学代码分析](PEER_HACKATHON3_ANALYSIS_ZH.md)和[本地下载清单](LOCAL_DOWNLOAD_MANIFEST_ZH.md)用于追查特定背景。它们不是新的独立神经网络确认实验。

后期各轮原科学方案、有效运行配置、统计及图件从对应实验的EVIDENCE进入：[完整实验索引](../experiments/README.md)。公开报告是独立科研表述，不是私人通信、原始会议材料或未公开长报告的逐字副本。
