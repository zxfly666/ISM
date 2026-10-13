# 实验精读与汇报路线

本项目研究：在临界 Ising 上，让模型接触不同的上下文尺寸和真实物理距离，能否改善未见布局上的条件推断，并进一步支持正确生成和跨尺度迁移。

准备汇报时，先掌握 G、I、N/W、RG 四组重点，再用局部诊断与生成实验限定结论。研究登记有14个已完成单元，其中包含冻结诊断和探索先导；另有一个只完成预检、未正式启动的方案。这个数量不是14次独立训练复现。

[综合报告](REPORT_ZH.md) · [方法与术语](METHODS_ZH.md) · [常见追问](QUESTIONS_ZH.md) · [历史更正](CORRECTIONS.md)

## 建议阅读顺序

| 顺序 | 阅读材料 | 读完需要能够解释 |
|---|---|---|
| 1 | [方法与指标](METHODS_ZH.md) | 尺寸、距离、跨度；CE、局部KL、生成误差；训练seed与采样seed |
| 2 | [G详细报告](size-spacing-factorial/REPORT_ZH.md) | 为什么是四个因素组合加一个跨度对照，两项正结果的范围是什么 |
| 3 | [I详细报告](fine-geometry-identification/REPORT_ZH.md) | 正确细排列、只保留跨度和打乱对应关系分别检验什么 |
| 4 | [精确局部基础能力](exact-local-capability/README.md) | 结构盲点、训练能力和扩容能力为什么要分开 |
| 5 | [N/W详细报告](canonical-context-size-generalization/REPORT_ZH.md) | 真值不变时扩容为何仍能出错，平均与最坏误差为什么不矛盾 |
| 6 | [RG详细报告](rg-coarse-data-training-transfer/REPORT_ZH.md) | 真正粗粒化怎样进入训练，为什么需要细重放对照，迁移证据缺在哪里 |
| 7 | [生成结果综述](REPORT_ZH.md#条件改善尚未稳定转化为整体生成收益) | T的正结果、R与Bridge的未决、末期MASK的负结果如何共同解释 |
| 8 | [问答](QUESTIONS_ZH.md)和[更正](CORRECTIONS.md) | 不混用模型、旧结论、统计区间和未执行计划 |

精读不是背诵所有超参数。每轮至少能画出对照表，指出相同条件和有意改变的条件，解释主图上的每条误差线，并说出一个限制结论的反例。

## 全部实验的阅读与汇报取舍

重要性指它对本项目研究判断的贡献，不等于结果是否显著。主讲顺序也不表示权重依次继承。

| 实验 | 提供的认识与主线关系 | 阅读深度 | 汇报位置 |
|---|---|---|---|
| [A/B/C几何对齐](geometry-aligned-conditioning/README.md) | 真实坐标在部分任务有利、另一任务反转；建立几何外推并不普遍成功的起点 | 设计和摘要 | G/I之前简述 |
| [F上下文一致性](context-consistency-training/README.md) | 自洽性改善没有确认主生成收益；检验一种修复策略 | 摘要及反例 | 与生成支线合并或备份 |
| [固定背景几何响应](fixed-background-geometry-response/README.md) | 控制背景后模型有符合参考的几何响应；响应随扩容减弱，准确与顺序一致性可冲突 | 构造与主图 | 几何机制短讲或问答 |
| [T稀疏条件](sparse-conditioning-training/README.md) | 补充低可见点任务后两主指标通过；顺序一致性仍变差 | 设计和主结果 | 与R、Bridge合并讲 |
| [R掩码与查询位置](mask-query-factorial/README.md) | 拆分稀疏mask和near-query，局部收益未稳定转为生成门通过 | 对照表与主区间 | 接在T之后 |
| [Bridge独立生成评价](independent-generation-bridge/README.md) | 新参考下完整区间仍跨零；防止把有利点估计当作确认 | 结论与独立单位 | 展示T/R后必须交代 |
| [G尺寸与间距](size-spacing-factorial/REPORT_ZH.md) | 最直接拆分两种训练覆盖；指定条件任务有明确正结果 | 精读 | 核心主讲 |
| [I细几何识别](fine-geometry-identification/REPORT_ZH.md) | 检验正确几何对应的增量价值及跨尺寸反转 | 精读 | 核心主讲 |
| [J观测K/V干预](observed-key-value-intervention/README.md) | 扩容不变性不等于准确性，具体结构干预存在能力缺口 | 方法、主结果与结构局限 | 与Basic合并短讲 |
| [Basic精确局部能力](exact-local-capability/README.md) | 区分不可表示、可以学到、扩大容器后仍失败 | 理解构造 | N/W前的必要解释 |
| [A4/B46尺寸对照](local-context-size-diagnostic/README.md) | 固定clock的初步尺寸收益，但可计数任务和三seed限制解释 | 略读，理解clock | N/W的前期动机 |
| [N/W精确尺寸泛化](canonical-context-size-generalization/REPORT_ZH.md) | 强相对改善与最坏误差失败同时成立 | 精读 | 核心主讲 |
| [末期MASK与纠错](late-mask-coverage-and-committed-spin-correction/README.md) | 检查生成末期训练覆盖与采样纠错；两主门未过 | 对照、主结果和物理代价 | 生成现状一页 |
| [RG粗数据训练](rg-coarse-data-training-transfer/REPORT_ZH.md) | 粗任务学习有信号，细尺度迁移尚未确认 | 精读，特别是限制 | 新方向与下一步讨论 |
| [尺寸与clock预检](preflights/size-clock-factorial/README.md) | 预算门失败，正式模型0；clock因素仍是未回答问题 | 只读状态 | 被问及时解释，不列入成功或失败科学结果 |

早期 L64、L128 和 Level1/Stage2 筛选见[历史起点](HISTORY_ZH.md)。先读Stage2B更正，不从旧GO标签判断当前成果。

## 四十分钟汇报建议

| 内容 | 时间 | 应呈现的材料 |
|---|---:|---|
| 问题起点与共同方法 | 6分钟 | L128偏差；条件与生成的区别；尺寸与距离示意 |
| G | 8分钟 | 五组对照、三项主比较、不利任务和生成代价 |
| I | 5分钟 | 三种坐标信息、主区间、H96反例 |
| Basic与N/W | 8分钟 | 精确腔体、训练尺寸、平均KL与最坏误差两张图 |
| 生成结果边界 | 4分钟 | T/R/Bridge一张对比表，末期MASK两主结果 |
| RG | 6分钟 | 三臂更新量、粗粒化构造、三谱系配对差 |
| 综合结论与讨论问题 | 3分钟 | 已支持的子命题、尚未解决的目标、下一步取舍 |

若时间缩短，先压缩历史过程和重复参数，不删各重点实验的主要反例。若时间增加，展开数据划分、真值构造和独立重复，不增加无关实验数量。

## 讲每个重点实验的固定顺序

1. 前一轮留下了什么问题，为什么现有证据不能回答它。
2. 对照组分别改变什么，哪些输入和训练条件配对。
3. 数据与测试怎么分开，主权重和主指标怎样预先固定。
4. 图的横轴、纵轴、点、误差线分别代表什么。
5. 正结果支持哪一句话，反例禁止哪一种推广。
6. 下一步需要补的证据是什么，而不是简单增加预算求显著。

共同方法只讲一次，各轮说明差异。尤其不要把后期dense网络与早期轴向模型混成同一个架构，也不要把RG暖启动称为从零训练。

## 汇报前自检

- 是否把G的间距覆盖收益误写成纯RoPE机制证明？
- 是否把I的改善方向与实质门混为一谈？
- 是否同时呈现N/W的平均收益和0/6绝对门？
- 是否在RG里明确三条旧谱系、已查看的参考与物理视野差异？
- 是否保留T的正结果，同时交代R、Bridge和末期MASK的限制？
- 是否把未执行的size×clock方案当成已回答的问题？
- 是否把旧Stage2B大效应、旧GO标签或采样seed数量误当作当前确认？

归档条目与统计口径见[实验地图](README.md)。需要复核数字时，顺着每轮详细报告中的机器摘要和原图数据进入证据，不必先浏览全部训练日志或下载权重。
