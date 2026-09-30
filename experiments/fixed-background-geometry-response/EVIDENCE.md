# 证据与复现入口

本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。

| 类型 | 入口 |
|---|---|
| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |
| 原始/公开版本SHA | [provenance.json](provenance.json) |
| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |
| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |

## 原科学代码

- [run_fixed_geometry_joint_20260923.py](../../scripts/research20260921/run_fixed_geometry_joint_20260923.py)
- [analyze_fixed_geometry_joint_20260923.py](../../scripts/research20260921/analyze_fixed_geometry_joint_20260923.py)

共享依赖及冻结身份：[全源码清单](../source-manifest.json)。

## 机器结果、正式图与数据

| 公开文件 | 字节 | SHA-256（完整值见provenance） |
|---|---:|---|
| [evidence/cases.json](evidence/cases.json) | 198946 | `6de47a5f825c1728…` |
| [evidence/final_summary.json](evidence/final_summary.json) | 28093 | `0cd5bfdede081201…` |
| [evidence/inference_complete.json](evidence/inference_complete.json) | 744 | `b54137f06218aba6…` |
| [evidence/manifest.json](evidence/manifest.json) | 23072 | `78a69800049ea72a…` |
| [evidence/preflight.json](evidence/preflight.json) | 1165 | `4bf651ef937e588a…` |
| [evidence/query_specs.json](evidence/query_specs.json) | 6666 | `f28c703eac4ad0e4…` |
| [evidence/reference.npz](evidence/reference.npz) | 2070732 | `f61ea75583ad5a77…` |
| [evidence/reference_complete.json](evidence/reference_complete.json) | 7181 | `e70e6478af589c53…` |
| [evidence/run_protocol.json](evidence/run_protocol.json) | 4508 | `5398f7e9361e53cd…` |
| [evidence/analysis/bootstrap_block16.npz](evidence/analysis/bootstrap_block16.npz) | 923386 | `45100fd7cd368df3…` |
| [evidence/analysis/bootstrap_block4.npz](evidence/analysis/bootstrap_block4.npz) | 925534 | `e7cf42efdd801c95…` |
| [evidence/analysis/bootstrap_block8.npz](evidence/analysis/bootstrap_block8.npz) | 1847703 | `e297f961710f56f2…` |
| [evidence/analysis/complete.json](evidence/analysis/complete.json) | 558 | `672daa46a8819335…` |
| [evidence/analysis/crossed_uncertainty.json](evidence/analysis/crossed_uncertainty.json) | 84749 | `8ab6040334343522…` |
| [evidence/analysis/figures/geometry_response_ABC.pdf](evidence/analysis/figures/geometry_response_ABC.pdf) | 20712 | `c27b6d4dcd209118…` |
| [evidence/analysis/figures/geometry_response_ABC.png](evidence/analysis/figures/geometry_response_ABC.png) | 386746 | `c36767995fed351c…` |
| [evidence/analysis/figures/geometry_response_F012.pdf](evidence/analysis/figures/geometry_response_F012.pdf) | 20511 | `4929dc60c478eb52…` |
| [evidence/analysis/figures/geometry_response_F012.png](evidence/analysis/figures/geometry_response_F012.png) | 325378 | `7a20582580840e8b…` |
| [evidence/analysis/figures/joint_accuracy.pdf](evidence/analysis/figures/joint_accuracy.pdf) | 25460 | `5b82631e38285742…` |
| [evidence/analysis/figures/joint_accuracy.png](evidence/analysis/figures/joint_accuracy.png) | 408012 | `0fb8549a80284d23…` |
| [evidence/analysis/per_seed_metrics.csv](evidence/analysis/per_seed_metrics.csv) | 35569 | `569104d778758e2e…` |
| [evidence/analysis/plot_data.npz](evidence/analysis/plot_data.npz) | 786087 | `e770a14e14da9fbe…` |
| [evidence/analysis/point_estimates.json](evidence/analysis/point_estimates.json) | 13049 | `8ceeaa47f950cc21…` |
| [protocol/FIXED_GEOMETRY_JOINT_PROTOCOL_20260923_ZH.md](protocol/FIXED_GEOMETRY_JOINT_PROTOCOL_20260923_ZH.md) | 7215 | `25daa1b7d54b5ead…` |

## 图的来源

此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。
原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。
未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。
