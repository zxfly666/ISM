# 技术文档索引

- `L128_ZERO_SHOT_PROTOCOL_ZH.md`：冻结 L=64 权重并在 L=128 上检验尺度外推的实验定义、参数、预注册判据和结论边界。

| 文档 | 内容 |
|---|---|
| `SCALE_AWARE_CONTEXT_EXPERIMENT_REPORT_ZH.md` | **本次完整实验报告**：研究问题、数据、模型、三级实验设置、全部关键结果、结论边界与复现归档 |
| [`../results/scale_aware_context/README.md`](../results/scale_aware_context/README.md) | **最新结果入口**：Level 1、Stage 2A 与 Stage 2B causal screen 的证据链、图表和结论边界 |
| `LEVEL2_SCALE_COORDINATE_LOCAL_GLOBAL_PROTOCOL_ZH.md` | 当前执行依据：Stage 2A 与单 seed 因果筛选已完成，下一步为 3-seed、15000-step 正式确认 |
| `STAGE2_IMPLEMENTATION_LOG_ZH.md` | Stage 2 代码、单测、参数公平性、方案修正、远端运行、GPU 和结果审计日志 |
| `LEVEL1_RAPID_RESULT_REPORT_ZH.md` | 第一级正式结果、硬判据、异常诊断与 `NO-GO for unchanged T3` 结论 |
| `LEVEL1_RAPID_GO_NOGO_PROTOCOL_ZH.md` | 已完成的第一级快速 Go/No-Go 预注册方案 |
| `PROTOCOL_REVIEW_RESPONSE_ZH.md` | 外部复核意见的逐条采纳/部分采纳判断与修改理由 |
| `TECHNICAL_DESIGN_ZH.md` | 离散扩散、模型结构、采样器和物理评估设计 |
| `L64_PILOT_EXPERIMENT_PLAN_ZH.md` | 正式 `L=64` 实验流程、参数和预期现象 |
| `EXPERIMENT_LOG.md` | 实验实施记录与关键决策 |
| `L256_MC_DIAGNOSIS_ZH.md` | 临界 MC 图像、平衡态与 `G(r)` 诊断 |
| `PEER_HACKATHON3_ANALYSIS_ZH.md` | 同学代码的负面结果审计与可借鉴组件 |
| `LOCAL_DOWNLOAD_MANIFEST_ZH.md` | 本地数据与 checkpoint 完整性记录 |

最终综合报告按照仓库所有者要求未上传。本目录保留的是实验设计、诊断和审计
文档，便于读者复现实验和理解代码演化。
