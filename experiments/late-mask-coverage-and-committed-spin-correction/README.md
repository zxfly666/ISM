# 末期 MASK 覆盖与已提交自旋纠错

**Late-mask coverage and committed-spin correction** · A/L/E · 2026-10-01

| 30秒摘要 | 当前事实 |
|---|---|
| 问题 | 显式少MASK训练覆盖，以及修正已提交自旋，是否改善W96长程生成？ |
| 设计 | 六个旧I-F基座×三臂续训；全部final预先锁定；S256对比reveal192+repair64，等256次网络调用 |
| 完成 | 144000训练更新，18final/36EMA；新MC2048；6912正式图+384基座诊断；546预测/82176输入 |
| 主结果 | **P1、P2均未通过预定实质门或完整方向门**；科学成功为false |
| 条件与代价 | E的W48四项局部门通过、W96四项失败；旧CE保留通过；stress失败；P1能量/磁化保留未过 |
| 完整性 | CPU重建审计、两处逐成员备份、2256/2256联合覆盖与六图PNG/PDF实际检查通过 |

训练ID：`endpoint_mask_sampler_disentanglement_20261001`；评价ID：`endpoint_mask_sampler_evaluation_20261001`。
当前状态 `training_and_evaluation_complete`，**计算完整不等于科学主门通过**。

## 两个预定义主结果

终点为W96 G25–48 NRMSE，越低越好。实质门要求六谱系配对t和联合bootstrap的
97.5%区间上界同时小于−.05；方向门要求两者同时小于0。

| 比较 | 点估计 | 配对t 97.5%区间 | 联合bootstrap 97.5%区间 | 决定 |
|---|---:|---|---|---|
| P1：E,S − A,S | +.025709 | [−.043549, .094967] | [−.028896, .074126] | 实质、方向均未过 |
| P2：E,R − E,S | −.030331 | [−.058572, −.002090] | [−.080395, .021720] | 实质、方向均未过 |

P1点估计变差，但未确认伤害；P2有改善点估计，只有t区间排除0，**完整联合区间仍跨0**。
不选择更乐观区间替代主检验，也不把未过门解释为等效或永久无效。

![预定义主比较、六谱系点与两类区间](evaluation/analysis/03_predefined_primary.png)

[完整中文结果、配置、物理tradeoff与边界](RESULTS_ZH.md) ·
[全精度summary](evaluation/analysis/summary.json) · [六图、PDF/PNG与图注](evaluation/analysis/FIGURES.md)

## 真实谱系与执行范围

| 臂 | 普通遮盖 | 原sparse-visible | 显式少MASK |
|---|---|---|---|
| A original-support | 75%，t∈[.01,1] | 25% | 无 |
| L lower-time-floor | 75%，t∈[.002,1] | 25%，同A | 无 |
| E explicit-late-mask | 50%，t∈[.002,1] | 25%，同A | 25%，合法M均匀选择 |

六个旧I-F 12k基座92601–92606各分三支，每支+8k到20k；**不是18个fresh独立重复**。
不是J-C或N/W权重继续。M是MASK数，K是visible数，少MASK与少visible是相反区间。
物理数据、geometry、augmentation及同seed初始完整状态配对，干预mask/time有意不同。
E−L是整套覆盖策略增量，不单独识别绝对M的因果效应。

训练12:03:34.857–14:40:56.457（UTC+8），**2h37m21.600s**；
新增评价15:34:36.364–19:10:53.386，**3h36m17.023s**。
19:11:08.670远端导出，19:22:57.216完成独立备份联合核验。评价未重训、重启、挑点、缩样本或追加轮次。

BF16训练与推理精度门不是一回事：原BF16及三个新BF16混合候选失败均保留。
正式生成采用通过固定验证门的FP16 autocast、FP32参数/输出头/概率，条件及学习评价仍FP32。
原FP32完整W96 shard **113.381秒**导致旧12h全流程时间门失败，后续新评价时窗不回写旧门。
原18:29训练窗口、新21:30科学计算/22:00交付边界明确分开，没有续费或扩资源。

## 证据和数据可用性

公开1086条来源记录（训练63+评价1023）：546份条件/学习预测、全部统计及敏感性数组、
通过及失败精度候选、六图PNG/PDF、384图phase0诊断含24个spin shard、协议/审计/备份记录。
新增七个冻结评价源码保持原SHA；图和数值数组不改动，公开文字的机器信息脱敏逐项记录。
历史公开设计副本基于本地设计稿，而运行锁定的是初始备份内的冻结方案；二者一行文字有差别，
不能把该公开设计稿的SHA当作运行冻结SHA。实际方案身份与更正见[完整结果](RESULTS_ZH.md)。

**普通Git不含新大权重/tar、原始训练日志、大训练池、新MC原场、输入bank或6912正式生成spin/RNG轨迹。**
它们有逐路径/字节/SHA与归档映射，并不等于公开完整原始镜像或永久下载服务。
两处备份是同D盘独立目录，**不是异盘/异地灾备**。正式final没有从覆盖范围排除。

- [完整证据索引](EVIDENCE.md) · [来源账本](provenance.json) · [2256条可用性与归档清单](withheld-manifest.json)
- [当前状态](evidence/phase_status.json) · [联合覆盖回执](evaluation/closure/evaluation_independent_union_v1.json) · [六图实际检查](evaluation/closure/visual_review_v1.json)
- [实际评价协议](evaluation/run_protocol.json) · [冻结评价入口](../../scripts/research20261001_eval/run_evaluation.py)
- [训练时点历史说明](TRAINING_STAGE_ZH.md) · [历史状态快照](evidence/training_phase_status_20261001.json) · [原科学方案](protocol/SCIENTIFIC_PLAN_ZH.md)

结论限于固定六谱系、固定计算与观测任务：**未确认两项主收益，W96局部能力和压力能力仍有缺口**。
条件CE不是joint NLL、局部KL不是联合正确性、oracle不是神经成果、粗粒化诊断不是RG训练。
本轮完成不自动授权或启动下一轮。

[研究地图](../README.md) · [公共综合报告](../REPORT_ZH.md) · [复现说明](../REPRODUCING.md) · [数据可用性](../DATA_AVAILABILITY.md)
