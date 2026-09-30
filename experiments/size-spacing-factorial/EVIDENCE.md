# 证据与复现入口

本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。

| 类型 | 入口 |
|---|---|
| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |
| 原始/公开版本SHA | [provenance.json](provenance.json) |
| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |
| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |

## 原科学代码

- [run_core_geometry_factorial_20260925.py](../../scripts/research20260921/run_core_geometry_factorial_20260925.py)
- [core_geometry_factorial_analysis.py](../../scripts/research20260921/core_geometry_factorial_analysis.py)

共享依赖及冻结身份：[全源码清单](../source-manifest.json)。

## 机器结果、正式图与数据

| 公开文件 | 字节 | SHA-256（完整值见provenance） |
|---|---:|---|
| [evidence/budget_v2.json](evidence/budget_v2.json) | 661 | `f02f7790a6fb7c67…` |
| [evidence/final_summary.json](evidence/final_summary.json) | 19932 | `62b63f7255f07dca…` |
| [evidence/manifest.json](evidence/manifest.json) | 215980 | `e6ee8e7ce965d84a…` |
| [evidence/run_protocol.json](evidence/run_protocol.json) | 12472 | `79ca5867e680b105…` |
| [evidence/analysis/conditional_point.npz](evidence/analysis/conditional_point.npz) | 1751565 | `59097094f8ae3091…` |
| [evidence/analysis/crossed_uncertainty.json](evidence/analysis/crossed_uncertainty.json) | 19680 | `3f58d5e3d446f3f6…` |
| [evidence/analysis/distance_descriptive.csv](evidence/analysis/distance_descriptive.csv) | 145497 | `19b4324e4f6ec4ca…` |
| [evidence/analysis/generation_bootstrap_block16.npz](evidence/analysis/generation_bootstrap_block16.npz) | 1190899 | `1689773b9dcd2f61…` |
| [evidence/analysis/generation_bootstrap_block4.npz](evidence/analysis/generation_bootstrap_block4.npz) | 1191090 | `4d18dfc72b196336…` |
| [evidence/analysis/generation_bootstrap_block8.npz](evidence/analysis/generation_bootstrap_block8.npz) | 2380851 | `090e0758133ccebd…` |
| [evidence/analysis/generation_point.npz](evidence/analysis/generation_point.npz) | 1733 | `515b9368571c4f3e…` |
| [evidence/analysis/learning_curves.pdf](evidence/analysis/learning_curves.pdf) | 18241 | `0f1d8ddca2d8de9b…` |
| [evidence/analysis/learning_curves.png](evidence/analysis/learning_curves.png) | 163841 | `0ab5578f3de09248…` |
| [evidence/analysis/native_coordinate_control.pdf](evidence/analysis/native_coordinate_control.pdf) | 14857 | `64ae23734417e97b…` |
| [evidence/analysis/native_coordinate_control.png](evidence/analysis/native_coordinate_control.png) | 329000 | `36c6942481fa5208…` |
| [evidence/analysis/paired_contrasts.csv](evidence/analysis/paired_contrasts.csv) | 25394 | `5d71d1b179d3c28f…` |
| [evidence/analysis/per_seed_metrics.csv](evidence/analysis/per_seed_metrics.csv) | 76105 | `2417ec8bb7dc57e9…` |
| [evidence/analysis/plot_data.npz](evidence/analysis/plot_data.npz) | 18359 | `e88aae01f0089bd0…` |
| [evidence/analysis/primary_factorial_contrasts.pdf](evidence/analysis/primary_factorial_contrasts.pdf) | 12702 | `a3f587ed8af4122d…` |
| [evidence/analysis/primary_factorial_contrasts.png](evidence/analysis/primary_factorial_contrasts.png) | 77974 | `d5e896797890d125…` |
| [evidence/analysis/primary_paired_CE.pdf](evidence/analysis/primary_paired_CE.pdf) | 19121 | `0b2feaba5bbc9c4d…` |
| [evidence/analysis/primary_paired_CE.png](evidence/analysis/primary_paired_CE.png) | 365605 | `dda656c34388e8fa…` |
| [evidence/analysis/secondary_generation.pdf](evidence/analysis/secondary_generation.pdf) | 15454 | `a3f4907113247f30…` |
| [evidence/analysis/secondary_generation.png](evidence/analysis/secondary_generation.png) | 259218 | `926b69c3d0d8242b…` |
| [evidence/analysis/secondary_interaction.pdf](evidence/analysis/secondary_interaction.pdf) | 13730 | `a9f1a4ef4a7d7995…` |
| [evidence/analysis/secondary_interaction.png](evidence/analysis/secondary_interaction.png) | 218966 | `ba2c92a18efaf618…` |
| [evidence/analysis/timing.json](evidence/analysis/timing.json) | 272 | `bfd8d12a483fbe12…` |
| [protocol/CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md](protocol/CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md) | 21433 | `330b5ab60b793005…` |
| [protocol/CORE_GEOMETRY_FACTORIAL_BUDGET_V2_20260925_ZH.md](protocol/CORE_GEOMETRY_FACTORIAL_BUDGET_V2_20260925_ZH.md) | 2844 | `c688000fd10b1844…` |
| [evidence/analysis/conditional_bootstrap_block1_primary_projection.npz](evidence/analysis/conditional_bootstrap_block1_primary_projection.npz) | 361643 | `7a84f93a731b52ce…` |
| [evidence/analysis/conditional_bootstrap_block2_primary_projection.npz](evidence/analysis/conditional_bootstrap_block2_primary_projection.npz) | 721559 | `e66a0837cb218657…` |
| [evidence/analysis/conditional_bootstrap_block4_primary_projection.npz](evidence/analysis/conditional_bootstrap_block4_primary_projection.npz) | 361097 | `510f3742a3319ab3…` |
| [evidence/analysis/conditional_mc_only_primary_projection.npz](evidence/analysis/conditional_mc_only_primary_projection.npz) | 360545 | `1b03ec3741293d7d…` |
| [evidence/analysis/conditional_seed_only_primary_projection.npz](evidence/analysis/conditional_seed_only_primary_projection.npz) | 70841 | `5c95c82b98d61230…` |

## 图的来源

此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。
原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。
未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。
