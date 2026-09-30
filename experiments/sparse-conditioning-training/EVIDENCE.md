# 证据与复现入口

本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。

| 类型 | 入口 |
|---|---|
| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |
| 原始/公开版本SHA | [provenance.json](provenance.json) |
| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |
| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |

## 原科学代码

- [run_sparse_conditioning_20260923.py](../../scripts/research20260921/run_sparse_conditioning_20260923.py)
- [finalize_sparse_conditioning_20260923.py](../../scripts/research20260921/finalize_sparse_conditioning_20260923.py)

共享依赖及冻结身份：[全源码清单](../source-manifest.json)。

## 机器结果、正式图与数据

| 公开文件 | 字节 | SHA-256（完整值见provenance） |
|---|---:|---|
| [evidence/cases.json](evidence/cases.json) | 251416 | `c4de76682210d71c…` |
| [evidence/final_summary.json](evidence/final_summary.json) | 16708 | `801bf687e07b43e2…` |
| [evidence/manifest.json](evidence/manifest.json) | 165009 | `ba3ea20ea86bd1ff…` |
| [evidence/run_protocol.json](evidence/run_protocol.json) | 12482 | `e7e4dcb9ef796f0c…` |
| [evidence/analysis/bootstrap_block16.npz](evidence/analysis/bootstrap_block16.npz) | 893559 | `186d9ed32717288e…` |
| [evidence/analysis/bootstrap_block4.npz](evidence/analysis/bootstrap_block4.npz) | 894759 | `44514d8d3cb3dbdd…` |
| [evidence/analysis/bootstrap_block8.npz](evidence/analysis/bootstrap_block8.npz) | 1788285 | `f8f1cce823895ac9…` |
| [evidence/analysis/crossed_uncertainty.json](evidence/analysis/crossed_uncertainty.json) | 162121 | `b38c1ef0284ca621…` |
| [evidence/analysis/figures/accuracy_and_consistency.pdf](evidence/analysis/figures/accuracy_and_consistency.pdf) | 15425 | `0c730628cf5ea32c…` |
| [evidence/analysis/figures/accuracy_and_consistency.png](evidence/analysis/figures/accuracy_and_consistency.png) | 250608 | `1a0090fafe2ec840…` |
| [evidence/analysis/figures/generation_correlations.pdf](evidence/analysis/figures/generation_correlations.pdf) | 32539 | `af3ef10929822e83…` |
| [evidence/analysis/figures/generation_correlations.png](evidence/analysis/figures/generation_correlations.png) | 383696 | `d4fb5e3990be0e49…` |
| [evidence/analysis/figures/primary_paired_seeds.pdf](evidence/analysis/figures/primary_paired_seeds.pdf) | 19100 | `b9f1638a1a5bec52…` |
| [evidence/analysis/figures/primary_paired_seeds.png](evidence/analysis/figures/primary_paired_seeds.png) | 362881 | `5187f46e939c5309…` |
| [evidence/analysis/paired_contrasts.csv](evidence/analysis/paired_contrasts.csv) | 89882 | `c665d4751b5ae042…` |
| [evidence/analysis/per_seed_metrics.csv](evidence/analysis/per_seed_metrics.csv) | 107174 | `1dbe0955d22189e7…` |
| [evidence/analysis/plot_data.npz](evidence/analysis/plot_data.npz) | 693979 | `bdf62231a1551122…` |
| [protocol/SPARSE_CONDITIONING_TRAINING_PROTOCOL_20260923_ZH.md](protocol/SPARSE_CONDITIONING_TRAINING_PROTOCOL_20260923_ZH.md) | 13278 | `5b287d35ce4203e2…` |
| [evidence/reference/complete.json](evidence/reference/complete.json) | 6016 | `048c06177b8be30b…` |
| [evidence/reference/protocol.json](evidence/reference/protocol.json) | 603 | `9788f77eca3ee2c5…` |

## 图的来源

此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。
原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。
未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。
