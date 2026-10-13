# ISM 条件推断与尺度外推研究

本项目用临界二维 Ising 系统研究：训练时覆盖不同上下文尺寸和真实物理距离，能否让模型在未见布局上准确推断，以及这些条件能力能否支持正确的整体生成和跨尺度迁移。

**目前最明确的结论是：指定条件任务上的尺寸与间距覆盖有收益，但可靠的绝对外推、完整联合生成和RG细尺度迁移仍未建立。** 正结果、反例和未通过的判据一起保留。

文档整理日期为2026年10月13日；下列科学结果来自截至2026年10月1日的已完成实验，不是新一轮计算。

## 从哪里开始

| 你的目的 | 最短阅读路线 |
|---|---|
| 首次了解研究 | [综合报告](experiments/REPORT_ZH.md) → [方法与指标](experiments/METHODS_ZH.md) |
| 准备30分钟以上汇报 | [精读与汇报路线](experiments/STUDY_GUIDE_ZH.md) → 四个重点报告 → [常见追问](experiments/QUESTIONS_ZH.md) |
| 查看做过哪些实验 | [14个研究单元与权重谱系](experiments/README.md) → [早期基线与筛选](experiments/HISTORY_ZH.md) |
| 核对某个结论 | 每轮报告 → 原统计JSON/图数据 → EVIDENCE与来源清单 |
| 重放公开结果或重新训练 | 先读[复现路线](experiments/REPRODUCING.md)与[数据可用性](experiments/DATA_AVAILABILITY.md) |
| 查看代码与原方案 | [技术文档索引](docs/README.md)；各轮EVIDENCE链接原driver与协议 |

## 四个重点实验

| 实验 | 回答的问题 | 当前结论 | 详细讲解 |
|---|---|---|---|
| G 尺寸与间距 | 两种训练覆盖分别是否有用 | H1/H2通过；细几何H3未确认，部分任务反转 | [从对照设计读起](experiments/size-spacing-factorial/REPORT_ZH.md) |
| I 细几何识别 | 正确位置关系是否超过总跨度或随机对应 | 指定任务F−R实质门通过，F−S仅方向通过；H96有反例 | [理解两个不同对照](experiments/fine-geometry-identification/REPORT_ZH.md) |
| N/W 精确尺寸泛化 | 真值不变时扩容能否仍准确 | 平均KL约下降90%，W最坏误差门仍0/6 | [同时看均值和最坏误差](experiments/canonical-context-size-generalization/REPORT_ZH.md) |
| RG 粗尺度训练 | 粗任务学习是否帮助细任务 | 粗CE改善，三旧谱系的细W96迁移未确定 | [学习信号与迁移边界](experiments/rg-coarse-data-training-transfer/REPORT_ZH.md) |

这些不是同一个模型连续升级的四站。G、I、N/W分别从零训练；RG由三个旧I-F EMA暖启动并重置优化器。模型身份、独立重复和评价任务不能跨轮混用。

## 生成结果也必须一起读

[T稀疏训练](experiments/sparse-conditioning-training/README.md)有指定局部与生成主指标正结果；[R因素对照](experiments/mask-query-factorial/README.md)和[Bridge独立参考评价](experiments/independent-generation-bridge/README.md)未确认完整生成收益；[末期MASK覆盖与纠错](experiments/late-mask-coverage-and-committed-spin-correction/README.md)两项主门也未过。条件CE、局部KL、关联误差和能量磁化是不同证据，不互相替代。

## 历史结果怎样使用

早期L64同分布pilot与L64到L128零样本外推是问题起点，模型结构不同于后期dense实验。[历史说明](experiments/HISTORY_ZH.md)把基线、Level1及Stage2筛选分开。

**旧Stage2B的native-unit对照已经更正，旧−.08238及其置信区间、GO_FULL_2B不能作为当前确认性结论。** 正确点差约−.004440，多seed正式确认未执行。所有历史更正集中在[CORRECTIONS](experiments/CORRECTIONS.md)，原数据和失败记录保留。

[尺寸与clock的12小时方案](experiments/preflights/size-clock-factorial/README.md)只有预检，没有正式模型；后来的N/W不是这项2×2实验的完成版。

## 仓库结构

| 目录 | 用途 |
|---|---|
| [experiments](experiments/) | 当前阅读入口、语义化实验报告、正式证据、来源和可用性清单 |
| [docs](docs/README.md) | 技术设计、历史报告与协议；阅读时注意日期和更正 |
| [results](results/README.md) | 早期精选结果的历史入口，不代表最新主结论 |
| [ism_diffusion](ism_diffusion/) | 模型、数据与物理指标实现 |
| [scripts](scripts/) | 各轮训练、评价、分析与管理脚本，原科学路径保留 |
| [configs](configs/) | 早期配置；后期配置按每轮协议定位 |
| [artifacts](artifacts/README.md) | 早期已发布产物及原始运行组织；本地未跟踪文件不等于GitHub公开内容 |
| [tests](tests/) | 实现测试，不把软件测试通过当作科学假设通过 |

不为整理目录而移动冻结源码、协议、数组或权重。阅读层负责解释，证据层负责提供原始身份和可复核数值。

## 不训练的核对入口

以下脚本只读取公开科学证据，不启动GPU、MC或新训练：

```bash
pip install -r experiments/requirements-replay.txt
python scripts/archive20260930/validate.py
python scripts/archive20260930/replay.py
```

后者重放选定历史决定，不代表重新分析全部实验。当前文档的链接、引用数字及改动范围另由[文档检查](scripts/reporting20261013/check_docs.py)核查。

完整训练需要按[复现说明](experiments/REPRODUCING.md)获取缺失输入并使用独立输出目录，不能在原run.lock目录再次启动。早期L64实现和smoke命令保留在[实现说明](docs/L64_IMPLEMENTATION_GUIDE.md)。

## 公开范围

公开源码、方案、主要预测与统计、正式图及清单；并非所有大权重、完整MC父场、生成spin/RNG轨迹都在普通Git中。历史LFS对象、保管者备份与公开下载是不同概念，详见[数据可用性](experiments/DATA_AVAILABILITY.md)和[CHECKPOINTS](CHECKPOINTS.md)。私人通信、连接信息和凭据不属于公开科研资料。
