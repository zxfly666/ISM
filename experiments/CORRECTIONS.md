# 历史更正与不应抹去的失败

不改历史commit，不更改冻结结果；本页记录当前阅读方式。原始文件中旧`GO`、`design_only`或“等待交付”字样是当时的状态，不自动代表今天的科学判断。

| 历史表述 / 事件 | 新证据或问题 | 正确阅读与证据入口 |
|---|---|---|
| 早期L64采样器原实现 | clock/schedule错误曾被修复并重新评价冻结权重 | 不能说原采样器从未有错，也不叫新训练复现；[历史结果](../artifacts/final_l64/final_summary.json) |
| Stage2B旧matched vs native-unit比较、GO_FULL_2B | 错误native对照曾给CE .553880；正确unit约.475944，对比matched .471505 | 正确点差约−.004440，非旧−.08238；旧CI不能套给新对照；3seed×15k正式方案没有执行，只有单seed8k筛选；[历史档案](../results/scale_aware_context/README.md) |
| F扩容M筛选 | 两seed变差，其余四个未跑 | 是2seed开发选择，非完整六seed容量确认；[F](context-consistency-training/README.md) |
| F统计口径 | 主原始点差−.006257336，与bootstrap均值−.005385059不同；实际draws2000/1000/1000 | 不把bootstrap均值当原始点估计；本次校正文字口径，不更改历史机器数组 |
| T、R被误读成一条连续续训链 | 二者分别由F0-36k分支 | R不是T的40k继续训练，辅助loss权重还不同；[R](mask-query-factorial/README.md) |
| Bridge独立MC和采样seed | 权重仍是旧R六lineages | 新sampling≠新training replication；[Bridge](independent-generation-bridge/README.md) |
| G10h预算门失败后12h执行 | 正式开始前明确版本化的新窗口 | 不隐藏失败，不称原10h门通过；[G协议](size-spacing-factorial/protocol/) |
| G-H3没有显著性 | 98.333333%CI跨0，也跨−.002 | 未通过，不等效；次要生成阳性不能重新判主假设；[G](size-spacing-factorial/README.md) |
| I-P1点差小于−.002 | 区间上界−.001332未越实质门 | 仅方向通过，不称完整联合成功；[I](fine-geometry-identification/README.md) |
| J observed-only固定t不变 | 特定K1/同符号K2构造下V缺少位置内容，可几何盲 | 结构不变性≠条件准确；[J](observed-key-value-intervention/README.md)、[精确能力](exact-local-capability/README.md) |
| J `all_64_backgrounds_independent` | K4局部输入可在不同背景记录中重复 | 字段措辞过强，不是64份独立物理信息；未重跑或覆盖原统计 |
| A4/B46初始2h门与启动v1路径错误 | 显式3h授权保留原起点；v1在训练lock前报错 | v2只修管理路径，无重复正式训练；[A4/B46](local-context-size-diagnostic/README.md) |
| 12h size-clock方案 | 真实测速投影26.486h | 正式模型0；scratch/CPU人工fixture不冒充结果；[停止档案](preflights/size-clock-factorial/README.md) |
| 6h平均KL大幅下降 | W最坏误差门0/6 | 相对成功、绝对失败，完整门失败；[6h](canonical-context-size-generalization/README.md) |
| 6h原manifest的stdout字节 | 打包后管理打印使0→115字节 | 增量closing manifest覆盖，不重跑科学分析；[闭环记录](canonical-context-size-generalization/evidence/closing_union_v1.json) |

## 不确定性不是“错误可以被美化”

有限六seed和有限MC链使区间可能很宽；选择更乐观的固定模型/MC-only区间不能替代预定总体。signed magnetization/energy偏差的目标是0，不能全部按“更小”排序。旧MC条件CE不等于精确KL；精确局部任务的KL也不是全图joint NLL。

本次归档新增的是信息组织和公开来源链，不更改任何科学门、seed、band、样本量、采样器或统计定义。原本地162页总报告保留，公共阅读版明确是编辑版，不伪装成同字节历史原件。
