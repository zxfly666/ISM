# 证据与复现入口

本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。

| 类型 | 入口 |
|---|---|
| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |
| 原始/公开版本SHA | [provenance.json](provenance.json) |
| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |
| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |

## 原科学代码

- [run_generation_bridge_20260924.py](../../scripts/research20260921/run_generation_bridge_20260924.py)

共享依赖及冻结身份：[全源码清单](../source-manifest.json)。

## 机器结果、正式图与数据

| 公开文件 | 字节 | SHA-256（完整值见provenance） |
|---|---:|---|
| [evidence/complete_scientific_backup_audit_20260925.json](evidence/complete_scientific_backup_audit_20260925.json) | 4192 | `6f37bc8f4a71cfad…` |
| [evidence/final_summary.json](evidence/final_summary.json) | 3109 | `a4cc9677da58b4a4…` |
| [evidence/manifest.json](evidence/manifest.json) | 209173 | `1bf098f5ba964f9a…` |
| [evidence/run_protocol.json](evidence/run_protocol.json) | 9790 | `7a069bdfc38f3014…` |
| [evidence/analysis/conditional_bootstrap.npz](evidence/analysis/conditional_bootstrap.npz) | 3768792 | `c7710f5ad3a57eac…` |
| [evidence/analysis/conditional_contrasts.csv](evidence/analysis/conditional_contrasts.csv) | 20095 | `42519c3a516950ac…` |
| [evidence/analysis/conditional_point.npz](evidence/analysis/conditional_point.npz) | 716544 | `0159452a43fd0e9b…` |
| [evidence/analysis/coverage_map.pdf](evidence/analysis/coverage_map.pdf) | 18403 | `3e77a55ca0c99f10…` |
| [evidence/analysis/coverage_map.png](evidence/analysis/coverage_map.png) | 242384 | `e50867ab6974aa47…` |
| [evidence/analysis/distance_descriptive.csv](evidence/analysis/distance_descriptive.csv) | 129969 | `ffc139ba0678a60e…` |
| [evidence/analysis/fixed_models_block8.npz](evidence/analysis/fixed_models_block8.npz) | 146331 | `7242bbad0aeee211…` |
| [evidence/analysis/generation_confirmation.pdf](evidence/analysis/generation_confirmation.pdf) | 13682 | `0276643921f2032a…` |
| [evidence/analysis/generation_confirmation.png](evidence/analysis/generation_confirmation.png) | 180028 | `9f8e4ca4053d8798…` |
| [evidence/analysis/generation_point.npz](evidence/analysis/generation_point.npz) | 1366 | `7ef16ecd518eb9b2…` |
| [evidence/analysis/generation_uncertainty.json](evidence/analysis/generation_uncertainty.json) | 18318 | `984437d18df4d3b7…` |
| [evidence/analysis/joint_block16.npz](evidence/analysis/joint_block16.npz) | 147290 | `7210a2715ad523c2…` |
| [evidence/analysis/joint_block4.npz](evidence/analysis/joint_block4.npz) | 147326 | `0fb6c9ea361ec2e1…` |
| [evidence/analysis/joint_block8.npz](evidence/analysis/joint_block8.npz) | 293997 | `8493153d02181f2c…` |
| [evidence/analysis/mc_only_block8.npz](evidence/analysis/mc_only_block8.npz) | 133484 | `91bcfcbbb22f60c1…` |
| [evidence/analysis/sampling_only_block8.npz](evidence/analysis/sampling_only_block8.npz) | 143248 | `351e2fd7d5f42240…` |
| [evidence/analysis/seed_only_block8.npz](evidence/analysis/seed_only_block8.npz) | 88825 | `8864d7f861fd5e96…` |
| [protocol/GENERATION_BRIDGE_PROTOCOL_20260924_ZH.md](protocol/GENERATION_BRIDGE_PROTOCOL_20260924_ZH.md) | 7382 | `37ca964ef02ccf56…` |
| [evidence/reference/complete.json](evidence/reference/complete.json) | 7894 | `0a5f52eb3ecd6ba3…` |

## 图的来源

此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。
原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。
未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。
