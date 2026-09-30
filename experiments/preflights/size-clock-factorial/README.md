# Size × clock factorial：预算停止预检

Historical alias **12h** · ID `dense_size_clock_geometry_12h_20260930` · 2026-09-30

**Question:** 多尺寸训练和固定clock分别能否改善精确局部条件泛化？**Design:** 计划六seed×2×2×24k。**Actual status:** **正式未启动：0个正式final、0个正式测试预测。** 这不是主假设失败，也不是下一轮6h的中途缩减。

| 计划臂 | 有效尺寸 | clock |
|---|---|---|
| A | 4/6 | native：1−K/Nvalid |
| B | 4/6 | canonical：.75 |
| C | 4/6/8/12 | native |
| D | 4/6/8/12 | canonical |

| 实际完成的预检 | 结果 | 不能当作 |
|---|---|---|
| 数学/恢复/人工3240单元fixture | 通过 | 正式模型泛化 |
| 两个2048步scratch，64条train-only G8 | maxKL约5.33e−7 / 1.63e−6 | 六seed确认或test性能 |
| 真实更新均速 | .108157秒 | 已完成576k训练 |
| 保守总投影 | **26.486h > 12h** | 12小时任务正式完成 |

```mermaid
flowchart LR
  P[预注册24模型576k更新] --> C[正确性与train-only拟合]
  C --> T[真实GPU测速]
  T --> S[预算门失败: 正式启动0]
  S --> N[另行授权的新6h问题]
```

[launch_gate.json](evidence/launch_gate.json)、[update_timing.json](evidence/update_timing.json)、[源码](../../../scripts/research20260930/sg_preflight.py)。计划原dense1976706参数FP32，AdamW(.9,.95)、wd.05、clip1、EMA.999、warm512峰3e−4到3e−5、batch128、raw24k主；这些是**计划值**。无正式MC或生成。

G8隐藏2×2、八点边界、16状态精确求和，1024条按72个D4×flip组拆696/160/168；这个预先确定的拆分被后续6h原样继承，不继承scratch权重。首次GPU预检优化器设备问题修正在冻结前，随后预算停止记录完整保留；不重置旧预算、不静默减seed。

03:47预检起点、04:03预算失败、04:07备份（UTC+8）。尚无可评价的正式主终点；所有人工/拟合图不能放进正式实验结果列。[后续独立新设计](../../canonical-context-size-generalization/README.md)缩为两组×六seed×8k、只问固定clock下的size覆盖。

## 证据、参数与可用性

[完整机器证据与协议索引](EVIDENCE.md) · [逐文件来源/编辑SHA](provenance.json) · [科学路径及未公开材料](withheld-manifest.json) · [本地核验回执](backup-verification.json)。

公开仓库不包含所有checkpoint、MC父场或bulk generation；从权重重跑与核对统计是不同复现层级。请先读[数据可用性](../../DATA_AVAILABILITY.md)和[复现说明](../../REPRODUCING.md)。

[返回研究证据地图](../../README.md)。
