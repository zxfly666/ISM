# 证据与复现入口

本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。

| 类型 | 入口 |
|---|---|
| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |
| 原始/公开版本SHA | [provenance.json](provenance.json) |
| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |
| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |

## 原科学代码

- [run_size_clock.py](../../../scripts/research20260930/run_size_clock.py)
- [sg_preflight.py](../../../scripts/research20260930/sg_preflight.py)

共享依赖及冻结身份：[全源码清单](../../source-manifest.json)。

## 机器结果、正式图与数据

| 公开文件 | 字节 | SHA-256（完整值见provenance） |
|---|---:|---|
| [evidence/capacity_A_prediction.npz](evidence/capacity_A_prediction.npz) | 1635 | `74bec664cd60b34b…` |
| [evidence/capacity_B_prediction.npz](evidence/capacity_B_prediction.npz) | 1650 | `ac0167a6bd944345…` |
| [evidence/capacity_complete.json](evidence/capacity_complete.json) | 988 | `376ea608465cb4ab…` |
| [evidence/environment.json](evidence/environment.json) | 4104 | `da74dfe9eca9aea6…` |
| [evidence/launch_gate.json](evidence/launch_gate.json) | 6029 | `1713a5106119a2f4…` |
| [evidence/model_checks_cpu.json](evidence/model_checks_cpu.json) | 432 | `bbcd66141740ab0c…` |
| [evidence/model_checks_gpu.json](evidence/model_checks_gpu.json) | 477 | `1ff05d72296dcfe4…` |
| [evidence/truth_checks.json](evidence/truth_checks.json) | 350 | `2453ec1d72eca33b…` |
| [evidence/update_timing.json](evidence/update_timing.json) | 3132 | `e459b34aa9a30634…` |
| [protocol/DENSE_SIZE_CLOCK_GEOMETRY_12H_PLAN_20260930_ZH.md](protocol/DENSE_SIZE_CLOCK_GEOMETRY_12H_PLAN_20260930_ZH.md) | 24687 | `e84c72933378170c…` |
| [protocol/DENSE_SIZE_CLOCK_GEOMETRY_12H_CONFIG_20260930.json](protocol/DENSE_SIZE_CLOCK_GEOMETRY_12H_CONFIG_20260930.json) | 9677 | `2466a9872a6e23e3…` |

## 图的来源

此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。
原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。
未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。
