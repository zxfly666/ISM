# Sparse mask × query-placement factorial

Historical alias **R00/R01/R10/R11** · ID `mask_query_factorial_20260924` · 2026-09-24

**Question:** 上轮收益来自稀疏mask还是near-query，二者是否交互？**Design:** 六F0-36k基座×2×2，追加4k。**Main result:** KL获益但原多重比较校正后的生成门未过。**Status:** 完成，合取主门失败。

| 臂 | 辅助mask | 辅助query | 共同控制 |
|---|---|---|---|
| R00 | ordinary | uniform | 同F0、完整raw/EMA/AdamW/RNG恢复、+4k到40k |
| R01 | ordinary | near | 同上 |
| R10 | sparse K | uniform | 同上 |
| R11 | sparse K | near | 同上 |

R从F0分支，**不从T的40k接着训练**；使用新训练流。所有R辅助CE不除以t，不能把R00当T0的逐位复现。物理训练流/初始化按seed配对；64查询槽在隐藏点不足时可重复，重复不增加独立样本。

| 预定义主比较 | 局部sequential KL差，98.75%CI | W128 G49–64 NRMSE差，98.75%CI | 合取 |
|---|---|---|---|
| R10−R00：sparse效应 | −.156301 [−.182838,−.133513] | −.196626 [−.372084,+.012976] | **未通过** |
| R11−R10：near-query增量 | −.014932 [−.024146,−.005470] | +.000797 [−.090677,+.076985] | **未通过** |

![原校正区间，不以更宽松95%改判](evidence/analysis/figures/primary_factorial_contrasts.png)

[完整交叉统计](evidence/analysis/crossed_uncertainty.json)、[原图数据](evidence/analysis/plot_data.npz)、[分析代码](../../scripts/research20260921/finalize_mask_query_factorial_20260924.py)。四主比较采用98.75%区间；R10生成的95%更乐观也不能替换原门。

```mermaid
flowchart LR
  F[六F0 36k] --> A[ordinary / sparse mask]
  A --> Q[uniform / near query]
  Q --> R[24配对续训到40k]
  R --> E[独立MC 条件与生成评价]
```

## 设置与独立单位

1,976,706参数dense、128/4/7、RoPE10000、dropout0，AdamW(.9,.95)、wd.05、clip1、EMA.999；固定4k追加、18,432token/update，普通/辅助训练结构和lr详见[实际协议](evidence/run_protocol.json)及源码。不能按validation选checkpoint。

新参考seed2026092411，8×128=1024父场；每模型W12896图、cont48及stride10各64，共224，24模型5,376图/336shard；**无W96生成**。S0cos²256/T1/noMC correction。六训练seed×MC链/父场时间块×16图shard，主块8的4000次、块4/16各1000。mask/query嵌套不增加seed。保留门通过，order一致性不全改善。

实测约9.292h，888/888必需路径本地核验；重复last与明示scratch权重排除，不排除正式final。结论是稀疏目标有局部准确性证据，生成确认仍未达原门；不能以条件结果宣称完整联合生成改善。

## 证据、参数与可用性

[完整机器证据与协议索引](EVIDENCE.md) · [逐文件来源/编辑SHA](provenance.json) · [科学路径及未公开材料](withheld-manifest.json) · [本地核验回执](backup-verification.json)。

公开仓库不包含所有checkpoint、MC父场或bulk generation；从权重重跑与核对统计是不同复现层级。请先读[数据可用性](../DATA_AVAILABILITY.md)和[复现说明](../REPRODUCING.md)。

[返回研究证据地图](../README.md)。
