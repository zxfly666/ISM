# RG 数据训练证据索引

原研究标识为 `rg_data_training_pilot_20261001`。此目录使用语义名称，未改写原科学路径。
全部 382 条公开来源记录、原始与公开 SHA 和文本转换见 [provenance](provenance.json)。

| 证据 | 用途 |
|---|---|
| [冻结运行协议](evidence/run_protocol.json) | 33 项源和输入身份、配置、唯一运行范围 |
| [固定计划](protocol/RG_DATA_TRAINING_PILOT_PLAN_20261001_ZH.md) | 预定设计、主比较和解释边界 |
| [固定配置](protocol/RG_DATA_TRAINING_PILOT_CONFIG_20261001.json) | 三臂更新、随机种子、期限、统计量 |
| [启动前验证](evidence/preflight/20261001_141634/summary.json) | 真 RG 映射、零嵌入身份、四步确定性、实际吞吐 |
| [样本拆分](evidence/data_manifest.json) | 768/64/256 整场无重合；测试 16 链 |
| [训练审计](evidence/audit/training.json) | 30720 日志、全部输入重建、9 final/9 midpoint、CPU 恢复 |
| [bank 审计](evidence/audit/banks.json) | 18 组输入全部字段重建 |
| [预测审计](evidence/audit/predictions.json) | 297 个文件、源/输入摘要和逐父样本指标重算 |
| [全部预测](evidence/predictions/) | 初始、final、midpoint 的父样本概率、标签、CE/KL |
| [完整统计](evidence/analysis/summary.json) | 主/次比较、两类区间、符号翻转、留一、学习曲线 |
| [全部谱系摘要](evidence/analysis/all_seed_metrics.json) | 每个固定任务、每臂、每谱系的结果 |
| [三图和图注](FIGURES.md) | 原 PNG/PDF，不挑最好谱系 |
| [远端科学完成](evidence/final_summary.json) | 远端状态；不独自代表本地交付完成 |
| [联合覆盖](closure/rgpilot_independent_union_v1.json) | 454/454 路径、字节和 SHA，含所有正式 final |
| [科学包本地验证](closure/rgpilot_complete_v1.local.verification.json) | 417 成员逐一核验 |
| [科学包独立目录验证](closure/rgpilot_complete_v1.independent.verification.json) | 第二处逐成员核验 |
| [管理补包验证](closure/rgpilot_administrative_closure_v1.independent.verification.json) | 最后 stdout/run/status 字节收齐 |
| [可用性清单](withheld-manifest.json) | 公开副本和保管者大文件的逐路径映射 |
| [实测时序](status.json) | 启动、训练、评价、审计、导出、联合核验 |

冻结科学入口为 [run_pilot.py](../../scripts/research20261001_rgpilot/run_pilot.py)，
[训练及恢复审计](../../scripts/research20261001_rgpilot/training.py)、
[评价及统计](../../scripts/research20261001_rgpilot/evaluation.py)、
[模型和数据](../../scripts/research20261001_rgpilot/model_data.py) 保持原 SHA。
不要在已有 run.lock 的原目录再次启动，也不要将本地验证等同于所有大文件已公开托管。
