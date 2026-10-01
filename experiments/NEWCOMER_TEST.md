# 陌生读者测试与公开版质量核查

这是本次归档作者的自查，**不是外部独立同行审查**。从root README进入，不依赖聊天记录或本地私人报告；下列问题都有公共链接可以回答。

| 10分钟阅读应能回答 | 当前答案 / 最短入口 |
|---|---|
| 1 项目研究什么？ | 物理坐标/上下文/训练几何与条件推断及联合生成的关系；[总图](README.md) |
| 2 为什么critical Ising？ | 可独立参考、已知局部规律与长程关联并存，静态非world model；[总图首段](README.md) |
| 3 核心scientific question？ | 条件网络怎样利用真实几何、如何跨观测布局/容器泛化；条件收益是否足以生成正确系综 |
| 4 最新实验有哪些？ | 13个已完成campaign；10月1日末期MASK三臂续训和评价完成，P1/P2主门未过；12h尺寸时钟仅预检 |
| 5 哪个最强？ | 物理因素分离以[G](size-spacing-factorial/README.md)最完整；精确真值以[6h](canonical-context-size-generalization/README.md)最清晰；不作跨任务单一排行榜 |
| 6 改了哪些变量？ | 每轮臂表，G尤其分开size、spacing与span-only，J分K/V来源 |
| 7 fresh/continued/frozen？ | 总图第二个Mermaid与registry；F0分别生T/R，Bridge只冻结 |
| 8 最强positive？ | G-H1/H2的预定条件门；6h主KL显著实质下降 |
| 9 重要negative？ | G-H3未决且非等效、W96held反转；Bridge/J主门失败；6h绝对0/6 |
| 10 仍不知道什么？ | 普遍细几何/任意距离/正确joint分布，见各轮限制与总图末节 |
| 11 下一问为什么值得做？ | 先区分未见边界模式尾部误差、容器聚合与学习不足，不自动扩大生成 |
| 12 原始结果在哪里？ | 每轮EVIDENCE→JSON/CSV/NPZ→plot_data/code→protocol与provenance |
| 13 哪些数据没上传？ | [可用性](DATA_AVAILABILITY.md)与每轮withheld manifest；不伪造下载链接 |
| 14 怎样复现？ | [三级路线及只读命令](REPRODUCING.md)，大型输入/环境缺口明确 |

## 结构与科学语义自查

| 维度 | 检查结果 |
|---|---|
| 研究定位/问题 | 有总证据地图；从物理条件到joint缺口，再到精确必要能力，不把静态模型升级为world model |
| 方法可理解性 | 每轮独立臂表、有效config、参数/预算、fresh/continued/frozen及主要数据来源 |
| 实验支持强度 | 原primary与secondary/diagnostic分开；CI层级和实际独立单位说明；不改门或筛seed |
| 叙述/证据一致性 | G主数组、J主record、F点估计、6h全部129600行重算；不是所有历史分析从头重跑 |
| 局限/复现 | 公共材料与本地backup明确不同；私人通信不公开；后续再训练需缺失数据与环境 |

科研写作检查重点用于“主张—证据—限制”的组织，而不是增强正结果的语气。

## 视觉核查（原图保留，不重画数据）

实际查看了21张PNG：G全部6图、最新6h全部5图、J主/能力2图、I主/几何压力2图，
以及A/B/C、F、固定几何、T、R、Bridge各1张README主图。
图例、轴、配对seed、0线/实质阈值、主次标签可读，无遮挡导致的科学误读。
G-H3相较H1尺度很小，故README同时给出精确CI表，不仅靠缩小的点线图判断。
J原图的连续连接线不表示cohort可配对；README明确S和C的独立谱系/推断角色。
Basic/A4B46使用正式表和流程图，没有借用CPU人工fixture作为训练结果图。

Mermaid均使用GitHub支持的基础flowchart语法，不依赖插件、脚本或外部资源。
本地链接/结构检查不冒充实际GitHub站点截图；推送后的远端文件与HEAD另行核验。

## 已运行的核查

- [公共SHA、NPZ、语法、链接与秘密模式扫描](validation.json)。
- [实际 Git 提交载荷检查](index-validation.json)：逐字节核对科学文件、确认链接目标进入提交，并检查包含既有两次未推送提交的完整发布差量；不以本地未跟踪文件冒充公开材料。
- [数值重放记录](numerical-replay.json)：G的三个主CI、held-gap反例、J主判定、F原始点差；6h全部600预测/120控制、标签和final SHA身份。
- [冻结Python源身份对照](source-identity-audit.json)：可从公开root协议解析的447个记录，当前本地源全部逐字节一致；重复引用不计作447个独立文件。
- 原数学测试：`tests/test_diagnostic_math.py`、`tests/test_fixed_geometry_math.py`、`tests/test_mechanism_design.py`，12项通过；未运行新模型训练。
- 四个历史冻结Python源保留原有EOF空行，`.gitattributes`仅对这些文件明确允许该空白规则，以保持原SHA；没有为消除lint提示改写冻结源码。

尚未完成或不承诺：所有历史checkpoint从GitHub下载恢复、全部MC/生成原数组公开、
所有bootstrap从底层重新抽样、对所有软件版本的位级复现、外部独立学术评价。

## 2026-10-01训练阶段增补自查（历史时点）

从root最新入口可以直接回答：为何做末期覆盖/纠错、A/L/E改变什么、六旧I-F如何分支、
18final和36EMA保存什么、P1/P2为何还没有数值、原精度/预算门为何仍标失败、
173路径本地覆盖与普通Git公开子集有何区别，以及后续缺少哪些评价。
该阶段不展示尚不存在的正式结果图，不把预检人工六图冒充训练后能力；
设计表/阶段验收表/谱系图承担本页可视解释。完整144k输入重建属于CPU完整性审计，
不是重新训练，也不是科学主效应分析。以上仅记录当时自查，正式评价已在之后的新窗口完成。

## 2026-10-01完整评价交付自查

最新入口能回答两个主效应、为何P2的t阳性不能代替joint门、W48/W96局部能力区别、
三CE保留、learning-incomplete未触发不等于完全收敛、物理非劣门与次要95%图的区别。
六张原PNG和渲染PDF逐张检查，图3两类区间、图5oracle身份、图6各自MC目标均正确保留。
[详细结果](late-mask-coverage-and-committed-spin-correction/RESULTS_ZH.md)与
[图检回执](late-mask-coverage-and-committed-spin-correction/evaluation/closure/visual_review_v1.json)给出证据。

546预测、全部分析/敏感性、precision候选、384图phase0公开；原MC和正式生成spin等仍为已备份但未公开托管。
联合2256/2256路径不是公共完整原始镜像，也不是异盘灾备。
[本次公共载荷检查](validation-evaluation-20261001.json)检查SHA/NPZ/语法/链接与凭据模式；
[精确本次Git差量检查](index-validation-evaluation-20261001.json)单独核对本次暂存发布内容；远端HEAD另行确认，均不冒充外部科学复现。
