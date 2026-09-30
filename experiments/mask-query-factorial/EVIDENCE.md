# 证据与复现入口

本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。

| 类型 | 入口 |
|---|---|
| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |
| 原始/公开版本SHA | [provenance.json](provenance.json) |
| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |
| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |

## 原科学代码

- [run_mask_query_factorial_20260924.py](../../scripts/research20260921/run_mask_query_factorial_20260924.py)
- [finalize_mask_query_factorial_20260924.py](../../scripts/research20260921/finalize_mask_query_factorial_20260924.py)

共享依赖及冻结身份：[全源码清单](../source-manifest.json)。

## 机器结果、正式图与数据

| 公开文件 | 字节 | SHA-256（完整值见provenance） |
|---|---:|---|
| [evidence/final_summary.json](evidence/final_summary.json) | 18713 | `322b53cda4a080cb…` |
| [evidence/manifest.json](evidence/manifest.json) | 170128 | `1040775da889879f…` |
| [evidence/analysis/bootstrap_block16.npz](evidence/analysis/bootstrap_block16.npz) | 2181689 | `8d97690dff3ace84…` |
| [evidence/analysis/bootstrap_block4.npz](evidence/analysis/bootstrap_block4.npz) | 2183730 | `e8c46d6f8d9a072c…` |
| [evidence/analysis/crossed_uncertainty.json](evidence/analysis/crossed_uncertainty.json) | 342920 | `e49dd4b82cc48e48…` |
| [evidence/analysis/figures/accuracy_and_consistency.pdf](evidence/analysis/figures/accuracy_and_consistency.pdf) | 16841 | `01366b86cd87f7d5…` |
| [evidence/analysis/figures/accuracy_and_consistency.png](evidence/analysis/figures/accuracy_and_consistency.png) | 307226 | `7ece1e1717763d20…` |
| [evidence/analysis/figures/generation_correlations.pdf](evidence/analysis/figures/generation_correlations.pdf) | 33075 | `97a7a90168e991e9…` |
| [evidence/analysis/figures/generation_correlations.png](evidence/analysis/figures/generation_correlations.png) | 498998 | `70d821dcd21d5cf8…` |
| [evidence/analysis/figures/primary_factorial_contrasts.pdf](evidence/analysis/figures/primary_factorial_contrasts.pdf) | 15395 | `18ca2e1ec17759e8…` |
| [evidence/analysis/figures/primary_factorial_contrasts.png](evidence/analysis/figures/primary_factorial_contrasts.png) | 152993 | `7bc086e43b716444…` |
| [evidence/analysis/figures/primary_paired_seeds.pdf](evidence/analysis/figures/primary_paired_seeds.pdf) | 20946 | `5f6c11de9e3fe782…` |
| [evidence/analysis/figures/primary_paired_seeds.png](evidence/analysis/figures/primary_paired_seeds.png) | 422390 | `759547f2ab2e03e1…` |
| [evidence/analysis/paired_contrasts.csv](evidence/analysis/paired_contrasts.csv) | 182771 | `e862939748a6be68…` |
| [evidence/analysis/per_seed_metrics.csv](evidence/analysis/per_seed_metrics.csv) | 135544 | `5c1de4f513e299e0…` |
| [evidence/analysis/plot_data.npz](evidence/analysis/plot_data.npz) | 921577 | `b6546b224ef8612b…` |
| [protocol/MASK_QUERY_FACTORIAL_PROTOCOL_20260924_ZH.md](protocol/MASK_QUERY_FACTORIAL_PROTOCOL_20260924_ZH.md) | 8017 | `096f171ad424be57…` |
| [evidence/run_protocol.json](evidence/run_protocol.json) | 12459 | `6b0a25fafaa993c5…` |

## 图的来源

此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。
原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。
未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。
