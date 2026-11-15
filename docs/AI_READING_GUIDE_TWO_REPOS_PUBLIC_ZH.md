# ISM 与 Hackathon-3 双仓库复习指南（公开共享版）

整理日期：2026年11月15日。本指南保留公开实验的设置、过程、结果和证据限制，不转载私人聊天、录音转写、身份细节或连接凭据。

**先分清时间范围：本文第二至十三节复习的是早期公开版本，不是项目截至10月1日的全部结果。** ISM的G、I、Basic、N/W、末期MASK评价与RG训练先导，请以[当前综合报告](../experiments/REPORT_ZH.md)和[核心精读路线](../experiments/STUDY_GUIDE_ZH.md)为准。这里的“后续”主要指早期Level 1／Stage 2A／Stage 2B，不要误认为包含所有后来实验。

| 你现在需要什么 | 阅读入口 |
|---|---|
| 从遗忘状态重新理解项目，并准备40分钟汇报 | [完整复习计划](PROJECT_REACTIVATION_AND_ADVISOR_REVIEW_PLAN_20261104_ZH.md) |
| 了解合作者的数据、正式基线和冻结扩容 | 本文第五、第七、第九节；原仓库按固定commit引用 |
| 比较两套早期基线的细节 | [早期基线设置与结果对照](EARLY_BASELINE_COMPARISON_PUBLIC_ZH.md) |
| 理解旧Stage 2B为何必须更正 | 本文第八节；不要沿用旧大效应或GO标签 |
| 阅读我们后来最重要的实验 | [G](../experiments/size-spacing-factorial/REPORT_ZH.md)、[I](../experiments/fine-geometry-identification/REPORT_ZH.md)、[Basic](../experiments/exact-local-capability/REPORT_ZH.md)、[N/W](../experiments/canonical-context-size-generalization/REPORT_ZH.md)、[RG](../experiments/rg-coarse-data-training-transfer/REPORT_ZH.md) |

## 一、两条实验线共同追问什么？

研究问题不是“模型能否接收更大的数组”，而是：模型在训练范围外的物理尺寸、观测间距和布局上，能否仍然做准确的条件推断，并最终生成正确的整体分布？

在这份公开指南中，我们把研究意图概括成四个可检验的问题。它们是科学层面的总结，不是任何私人通信的逐字引文：

1. **尺寸覆盖：** 训练接触多种窗口大小后，扩大输入容器时能否保持原有能力？
2. **真实距离覆盖：** 改变物理取样间距，并提供与取样一致的坐标，是否有助于未见几何上的推断？
3. **几何利用：** 模型是否利用正确的细位置对应，而不只是token数量、总跨度或局部计数？
4. **条件到整体：** 更好的条件预测，是否能转化为正确的长程关联、磁化、能量和整体生成？

这里必须分开三种操作：增加窗口中的点数；改变实际采样点的物理间距；只改变输入的位置编码。它们不等价。多尺寸、多间距训练也不自动证明尺度不变性或普遍外推。

合作者的公开路线先建立MC与开放窗口基线，再检验冻结模型的直接扩容；ISM早期周期实验与后续对照逐渐将失败现象拆成更具体的问题。**研究问题相接，不表示两仓库的权重、数据、预算或Stage编号相同。**

RG是另一层干预：真实粗粒化数据参与学习，而不只是把坐标乘以比例。后来的[RG训练先导](../experiments/rg-coarse-data-training-transfer/REPORT_ZH.md)应单独阅读；早期两仓库基线不能当作RG训练已经完成的证据。

建议先读本文版本表与合作者部分，再回完整复习计划的G、I、Basic/NW与RG模块。每个结论都要同时记录比较对象、指标、独立重复单位和不能推出的内容。

---

## 二、阅读范围、版本与证据规则

### 2.1 本文核查的公开版本

| 仓库 | 分支／固定版本 | 最近提交日期（UTC） | 定位 |
|---|---|---|---|
| A：ISM | `main`，`ed2235990fd213fcfaad30e6bde7480b08780c30` | 2026-08-19 | 早期周期基线＋L128 外推＋后续 scale-aware／Local–Global 筛查 |
| B：Hackathon-3 | `main`，`3f147535389fd489775011469d338e4402eb58cb` | 2026-08-06 | MC 数据验证＋开放窗口 RoPE 基线＋冻结模型直接扩大窗口 |

固定版本入口：[A 的目录快照](https://github.com/zxfly666/ISM/tree/ed2235990fd213fcfaad30e6bde7480b08780c30)，[B 的目录快照](https://github.com/zdacongming-glitch/Hackathon-3/tree/3f147535389fd489775011469d338e4402eb58cb)。后续若仓库更新，应记录新 commit，再核查差异；不能默认本文覆盖对方尚未公开的代码或实验。

下文 `A:路径`、`B:路径` 都指对应仓库根目录下的文件，不是同一工作目录中的同名文件。

### 2.2 遇到报告、配置和代码冲突时怎么判断？

按问题追溯证据，而不是机械地只信某一种文件：

1. 用原始逐样本数据、样本数组和日志确认实际输出。
2. 用运行时配置、checkpoint 元数据和模型加载代码确认究竟执行了什么。
3. 用评估／分析代码确认指标定义、筛选范围与对照条件。
4. 再用报告、README 和图注理解实验叙事。
5. 方案书中的计划参数和默认类参数不能覆盖实际运行配置；后续修改的脚本也不一定就是历史结果的生成版本。

“图看起来接近”“程序能接受大尺寸”“文件名写了 formal”“判断字段写了 GO”都不能单独代替证据链。

---

## 三、先建立全局实验地图，不要被 Stage 编号误导

| 实验线 | 研究问题 | 主入口 | 当前应如何理解 |
|---|---|---|---|
| A：早期 L64 周期基线 | 能否学习训练尺寸的临界统计？ | `docs/L64_PILOT_EXPERIMENT_PLAN_ZH.md`、`artifacts/final_l64/` | 训练尺寸内有较好的统计复现 |
| A：L64→L128 | 旧模型冻结后能否直接生成更大周期晶格？ | `docs/L128_ZERO_SHOT_PROTOCOL_ZH.md`、`artifacts/final_l128_zero_shot/` | 部分保留局部统计，但中长程存在系统偏差 |
| B：Stage 0／Stage 1 | MC 数据是否可靠？开放窗口 dense RoPE 基线是否有效？ | `STAGE0_FINDINGS.md`、`results/stage1_baseline_64step/` | 数据审计与训练尺寸内基线 |
| B：Stage 2 | 冻结 Stage 1，直接扩大开放窗口是否成功？ | `results/stage2_direct_context_extrapolation/` | 大窗口直接外推失败；没有训练变物理间距 |
| A：Level 1 | 多尺寸／坐标／真实间距的快速筛查 | `results/scale_aware_context/level1_rapid/` | 部分指标改善，但短程与 MASK 稳定性代价明显，原方案 NO-GO |
| A：Stage 2A | 局部路径保护能否减轻远端 context 干扰？ | `results/scale_aware_context/stage2a_screen/` | 明显改善特定局部诊断，坐标收益证据仍不足，CONDITIONAL_GO |
| A：Stage 2B | 随机真实间距＋匹配坐标是否帮助条件推断？ | `results/scale_aware_context/stage2b_causal_screen/` | 有较小但一致的单训练种子筛查收益；需按本文修正主对照 |

理解工作全貌的主线是：**建立有效基线 → 发现直接外推失败 → 检验老师的尺度训练设想 → 诊断失败机制 → 加入局部保护 → 对坐标作用做更细的筛查。**

这不是一条“每一步都成功”的故事，也不是两个仓库之间直接竞争一个排行榜分数。

---

## 四、A 仓库 ISM：结构与阅读方法

### 4.1 目录地图

```text
ISM/
├─ README.md / CHECKPOINTS.md         项目入口、早期权重说明
├─ docs/                             方案、技术设计、结果报告、实施记录
├─ configs/                          各条实验线的配置；包括 smoke 和 benchmark
├─ ism_diffusion/
│  ├─ model.py / diffusion.py        早期周期模型与离散扩散
│  ├─ ising.py / ising_numba.py      MC 数据生成实现
│  ├─ metrics.py / diagnostics.py    早期指标与诊断
│  ├─ scale_model.py                后续二维坐标 dense RoPE 去噪器
│  ├─ scale_data.py                 Level 1 几何与取样方式
│  ├─ scale_diffusion.py            坐标实验的 MASK 扩散与采样
│  ├─ scale_evaluation.py           开放窗口指标
│  ├─ stage2_model.py               Local–Global 双路径模型
│  ├─ stage2_data.py                后续随机间距数据与坐标条件
│  └─ stage2_sampling.py            后续采样器分支
├─ train.py / train_level1.py / train_stage2.py
├─ generate_data.py / generate_reference.py / generate_level1_parents.py
├─ sample_and_evaluate.py / sample_level1.py
├─ probe_level1.py / probe_stage2_coordinates.py / probe_stage2b_causal.py
├─ analyze_final_l64.py / analyze_l128_zero_shot.py
├─ analyze_level1.py / analyze_stage2a.py / analyze_stage2b_causal.py
├─ artifacts/
│  ├─ pilot_l64/                    早期训练记录等
│  ├─ final_l64/                    L64 正式聚合结果、表格和图
│  └─ final_l128_zero_shot/          L128 正式聚合结果、表格和图
├─ results/scale_aware_context/
│  ├─ level1_rapid/                 第一轮快速判断
│  ├─ stage2a_screen/               局部—全局筛查
│  ├─ stage2b_causal_screen/         随机间距与坐标反事实筛查
│  └─ checkpoints/                 后续实验归档权重、配置与训练记录
├─ data/                            早期数据及 level1 父场；部分为 Git LFS
├─ scripts/                         队列、分析与报告辅助入口
└─ tests/                           核心、Level 1、Stage 2 测试
```

这是一张选取关键文件的导航图，不是穷举。`artifacts/` 中其他历史试跑不应自动并入正式结果。

### 4.2 先读文档，再沿“配置→数据→网络→采样→评估”追踪

**理解早期路线：**

1. `docs/TECHNICAL_DESIGN_ZH.md`：总体方法与旧模型设计。
2. `docs/L64_PILOT_EXPERIMENT_PLAN_ZH.md`：训练与正式评估约定。
3. `docs/L128_ZERO_SHOT_PROTOCOL_ZH.md`：什么被冻结、什么变大、比较对象是什么。
4. `artifacts/final_l64/final_summary.json` 与 `artifacts/final_l128_zero_shot/final_summary.json`：实际结果。
5. 两个结果目录中的 `tables/correlation.csv`：直接检查整条关联曲线，而非只读拟合指数。

**理解老师 idea 的后续实验：**

1. [综合报告](https://github.com/zxfly666/ISM/blob/ed2235990fd213fcfaad30e6bde7480b08780c30/docs/SCALE_AWARE_CONTEXT_EXPERIMENT_REPORT_ZH.md)：先建立全貌，但必须结合本文第八节的修正。
2. `docs/LEVEL1_RAPID_GO_NOGO_PROTOCOL_ZH.md`、`docs/LEVEL1_RAPID_RESULT_REPORT_ZH.md`：为什么没有直接延长原方案训练。
3. `docs/LEVEL2_SCALE_COORDINATE_LOCAL_GLOBAL_PROTOCOL_ZH.md`、`docs/STAGE2_IMPLEMENTATION_LOG_ZH.md`：架构和实验路线为何改变。
4. `results/scale_aware_context/` 下三个阶段的 summary／decision 文件与原始数据。
5. 最后阅读对应代码，以确认文档中的名词究竟怎样实现。

### 4.3 A 实际上有三类模型，不能混为一个 baseline

| 项目 | 早期周期模型 | Level 1 模型 | Stage 2A／2B 模型 |
|---|---|---|---|
| 主要实现 | `model.py` | `scale_model.py` | `stage2_model.py` |
| 位置机制 | 周期距离 attention bias；不是 RoPE | 二维 RoPE | 局部、全局两条路径均使用坐标／RoPE |
| attention | 行列交替的轴向 attention，配合局部周期卷积 | dense attention | 半径受限局部路径＋dense 全局路径 |
| 数据语义 | 直接生成的完整周期 L64 系统 | 大父场上的开放／稀疏窗口 | 父场窗口、真实间距与随机间距 |
| 用途 | 早期基线与 L128 零样本外推 | 快速尺度训练筛查 | 局部保护与坐标机制筛查 |

早期模型约 **12,674,434** 参数。Stage 2A 的 Local–Global 约 **1,978,501** 参数，容量匹配 Dense+ 约 **1,976,706** 参数；两者约 0.09% 的参数量差异使其比直接拿旧大模型对比更合理，但仍不意味着所有计算量、归纳偏置都已匹配。

Local–Global 的关键设计是：局部分支只进行一次物理半径 1 的空间传播，后续局部块是逐点 MLP；全局分支保持 dense attention。两条路径不混合隐藏状态，而是在输出处以 gate 加入全局 logit 残差。它不是“完全关闭全局信息”，也不意味着实现了计算复杂度意义上的稀疏加速。

### 4.4 三轮坐标实验分别做了什么？

#### Level 1：先看数据与坐标的不同作用

| 变体 | 窗口宽度 | 自旋采样 | 输入坐标 |
|---|---|---|---|
| T0 | 固定 48 | 连续取样，间距 1 | 单位间距 |
| T3 | 16、24、32、48 | 间距 1、2、4 | 与真实间距匹配 |
| Pphase | 16、24、32、48 | 连续取样，间距 1 | 人为采用 1、2、4 的坐标间距 |
| Punit | 16、24、32、48 | 间距 1、2、4 | 单位间距 |

注意：T0 配置文件里的 widths 列表不能直接当成其实际训练尺寸；数据分支会选择最大宽度作为固定窗口。应查看 `scale_data.py` 中变体几何的逻辑。

T3 与 T0 同时改变窗口大小分布和物理间距，**不是严格隔离“只改变距离”的单因素对照**。不要把方案中曾设想的所有消融都当成已经运行。

#### Stage 2A：保护局部条件推断

- 训练宽度仍为 **16、24、32、48**，间距仍为 **1、2、4**；不是 Stage 2B 的更大配置。
- 8,000 更新步，单训练种子 1234，使用 EMA；比较 Local–Global 与容量匹配 Dense+。
- 重点测试扩大 context 后，远端 MASK 对中心点预测造成的额外扰动。
- 另有 W64 开放窗口无条件生成检查：模型 256 个样本、MC 1,024 个样本。
- 结果支持进一步研究，但坐标主效应当时尚未达到预设实用阈值，因此为 `CONDITIONAL_GO`。

#### Stage 2B：随机间距与坐标对照

- 训练宽度为 **16、24、32、48、64**；随机间距取自 **1、2、4、8**。
- `Gap-Matched`：按随机间距取真实自旋，并给真实坐标。
- `Gap-Unit`：类似的真实随机间距数据，但使用单位／排序坐标。
- `U-RandPE`：连续数据配随机位置模式，用来检查仅位置扰动／正则化能解释多少收益。
- 都是 8,000 更新步、单训练种子筛查；逐样本测试包含 5 个模型、5 种几何、4 个 t、3 种坐标条件，每项 512 个窗口。
- 测试几何包括训练间距混合、3/6 混合、固定 3、固定 6，以及 W48 的固定 10。

RandomGap 当前通过分别累加 x、y 方向的间距形成网格；它不是任意不规则点云或带孔几何。3、6 未作为这些变体的训练间距，但在验证配置中出现过，因此不能表述为“整个研发过程完全未见过的盲测尺度”。间距 10 超出训练间距范围，但对应测试不是更多 token 的完整大晶格生成。

### 4.5 数据、训练和采样的核查点

- 后续共享父场为 L1024，共 12 条链×每链 128 个保存构型，合计 1,536 个；split 按链划分为 6／2／2／2。不能把在线裁剪出的窗口数量当成独立 MC 样本量。
- 父场生成记录使用 Wolff、20 个等效 sweep 热化、每 2 个等效 sweep 保存；这只是生成设置，不能单凭它宣称有效样本数等于名义数量。
- 后续配置采用 t 范围 0.01～1，并额外给 0.02 的全 MASK 概率；需与 B 的 t 设置分开。
- Stage 2A 的冻结 S0 sampler 使用 256 步、temperature=1。某些配置里出现 `refinement_sweeps: 2`，但 **S0 不执行这两轮 refinement**；应看 `stage2_sampling.py` 的分支，不能只抄参数名。
- Stage 2B 条件评估元数据记录使用最终 step-8000 的 `last.pt`，加载时优先使用 EMA。后续归档文件名却是 `best_val.pt`；第十节说明为什么不能直接认为两者相同。

---

## 五、B 仓库 Hackathon-3：结构与阅读方法

### 5.1 目录地图

```text
Hackathon-3/
├─ README.md
├─ STAGE0_FINDINGS.md               MC 生成、精确小系统验证与统计诊断
├─ STAGE1_PROTOCOL.md               开放窗口 masked Transformer 方案
├─ STAGE2_PROTOCOL.md               冻结模型直接 context 外推方案
├─ configs/
│  ├─ stage0_*.json                 父场生成、扩展 train/val/test
│  ├─ stage1_*.json                 早期 pilot、50k 训练、诊断
│  └─ stage2_*.json                 直接外推、W128 补充及正式评估
├─ scripts/
│  ├─ run_stage0.py / audit_stage0.py / validate_crops.py
│  ├─ run_stage1.py / run_stage1_data_ablation.py
│  ├─ run_stage1_diagnostics.py
│  └─ run_stage2_context_extrapolation.py
├─ src/critical_ising/
│  ├─ wolff.py / exact.py           MC 与小系统精确参照
│  ├─ observables.py / diagnostics.py / crops.py / pipeline.py
│  ├─ masked_transformer.py         dense Transformer＋AxialRoPE2D
│  ├─ masked_generation.py          MASK 生成与采样器
│  ├─ stage1_data.py / stage1_metrics.py / stage1_pipeline.py
│  ├─ stage1_diagnostics.py
│  └─ stage2_metrics.py / stage2_pipeline.py
├─ results/
│  ├─ stage1_baseline_64step/
│  │  ├─ README.md / run_config.json
│  │  ├─ baseline_summary.csv / sampling_seed_metrics.csv
│  │  └─ figures/
│  └─ stage2_direct_context_extrapolation/
│     ├─ README.md / run_config.json / stage2_result.json
│     ├─ width_metrics.csv / sampling_seed_metrics.csv
│     └─ figures/
├─ outputs/                        本地运行产物；不是完整公开备份
├─ tests/
└─ pyproject.toml / uv.lock / requirements-stage0.txt
```

### 5.2 最有效的阅读顺序

1. `STAGE0_FINDINGS.md`：理解为什么用大父场与开放裁剪窗口，以及数据如何审计。
2. [Stage 1 最新基线结果](https://github.com/zdacongming-glitch/Hackathon-3/blob/3f147535389fd489775011469d338e4402eb58cb/results/stage1_baseline_64step/README.md)，再读同目录 `run_config.json` 与两个 CSV。
3. [Stage 2 直接外推结果](https://github.com/zdacongming-glitch/Hackathon-3/blob/3f147535389fd489775011469d338e4402eb58cb/results/stage2_direct_context_extrapolation/README.md)，再读 `width_metrics.csv`、`run_config.json`、`stage2_result.json`。
4. 回读两个 protocol，区分预设与事后说明。
5. 沿 `stage1_data.py` → `masked_transformer.py` → `masked_generation.py` → `stage1_metrics.py`／`stage2_metrics.py` 追踪实现。

### 5.3 当前公开基线的实际设置

| 内容 | 最新公开正式基线，而非早期 smoke 默认值 |
|---|---|
| 训练数据 | L512 父场，4,096 个训练父构型；在线连续裁剪开放窗口 |
| 训练宽度 | 32、48、64 |
| 网络 | 4 层、4 heads、宽度 64，约 204,672 参数，dropout=0.05 |
| attention／位置 | 全 token dense 双向 attention；2D RoPE，单位格点坐标 |
| 优化 | 50,000 更新，峰值学习率 1e-4，BF16；梯度累积 2 |
| MASK 设置 | t 在 0.2～1 均匀取样；无额外全 MASK 概率质量 |
| checkpoint | 最佳验证点约 46,500；原始权重，不是 A 后续的 EMA 规则 |
| 正式 sampler | absorbing diffusion，64 步，cosine schedule，temperature=1 |
| Stage 1 样本 | 每个宽度模型合计 1,024：4 个采样 seed×256；MC 1,024 |
| Stage 2 样本 | 每个宽度模型合计 256：4 个采样 seed×64；MC 256，另有独立 MC 对照 |

`AxialRoPE2D` 的 axial 表示位置编码按空间轴组织，**不是只沿行／列做 attention**。不要将 B 的 dense attention 与 A 旧模型的轴向 attention 混淆。

`MaskedTransformerConfig` 类里的默认宽度也不代表实际训练宽度；最新结果归档里的 run_config 才给出本次用的 64。

### 5.4 B 的 Stage 2 到底验证了什么？

它冻结 Stage 1 模型，在 W64、96、128、192、256 上直接生成。没有为这些大窗口重新训练，也没有执行老师后来提出的真实变间距／匹配坐标训练。因此这是**直接扩展现有 context 的失败边界实验**，不是尺度训练方法的最终效果。

它预设 (R_{seen}=32)，正式 tail 范围为：

\[
32<r\le \lfloor W/4\rfloor.
\]

所以 W64、W96、W128 的正式 tail 区间为空；W192 检验 r=33～48，W256 检验 r=33～64。W128 是有价值的尺寸诊断，但按该规则不构成非空的正式 tail 检验。

还要注意：(R_{seen}=32) 是协议选取的保守评价界线，不是“训练 W64 时网络从未计算过超过 32 的 token 相对距离”的字面证明。开放 W64 内本来就可以存在更远点对。

代码／方案中提到的某些后续 patch 或粗粒化方向，不能算到这一批冻结直接外推结果中。

---

## 六、两者如何比较？先承认比较对象不同

| 维度 | A 的早期周期实验 | B 的公开基线／直接外推 | A 的后续尺度实验 |
|---|---|---|---|
| 目标分布 | 有限周期 L64／L128 Ising | 大周期父场上的开放窗口边缘分布 | L1024 父场上的连续／稀疏窗口边缘分布 |
| 位置归纳偏置 | 周期距离 bias＋局部卷积，无 RoPE | 固定单位坐标 2D RoPE | 可变真实／对照坐标 RoPE；后来加入物理局部路径 |
| 规模 | 约 1,267 万参数 | 约 20.5 万参数 | 2A 约 198 万参数，含匹配 Dense+ 对照 |
| 训练数据记数 | 15,000 个完整训练构型 | 4,096 个训练父构型＋在线裁剪 | 父场按链拆分，再在线构造不同几何 |
| 外推问题 | 64→128 完整周期晶格 | 开放窗口扩展到 96／128／192／256 | 多距离泛化与局部稳定性；不是全部大尺寸生成已验证 |
| 训练与采样 | 60k 更新、EMA、128 步 sampler | 50k 更新、best raw checkpoint、64 步 sampler | 8k 单 seed 筛查；采样与条件 probe 分阶段设置 |
| 结果强项 | 训练尺寸统计较准，完整正式评估样本较多 | 数据审计、开放窗口定义与直接外推边界清楚 | 更直接拆解老师 idea，加入对照与条件诊断 |

据此，不能只用两张 (G(r)) 图就给出“哪个位置编码最好”的因果结论。边界条件、目标分布、容量、训练预算、采样器、评价尺寸、样本数量同时变化。

“A 短程能学到，B 不能”也过于粗糙：两者在训练范围内都能较好复现部分统计；扩展尺度下误差有距离与尺寸依赖。A 的 L128 在 r=8 已约高估 8%，r=32 约高估 20%，不是整个 r≤32 都保持精确。

---

## 七、关键结果索引：这些数字分别支持什么？

以下是导航性摘录，不是可直接合并的排行榜。各行指标定义和比较范围不同。

| 结果 | 关键数值 | 可以说什么／不能说什么 |
|---|---|---|
| A 旧 L64 正式评估 | raw G 的 r=1～16 NRMSE≈0.00405 | 训练尺寸内该关联指标约 0.405%；不等于全部分布已精确恢复 |
| A 旧 L128 零样本 | raw G 的 r=1～32 NRMSE≈0.12167；r=33～64≈0.22093 | 更大周期晶格存在明显系统误差，PARTIAL |
| A 旧 L128 比值 | 模型／MC 在 r=8、16、32、64 约为 1.081、1.137、1.198、1.232 | 距离增大时关联高估加重；不能写成远程恢复成功 |
| B Stage 1 | W32／48／64 raw G 相对 L2 约 0.0194／0.0380／0.0404 | 当前 64-step 基线不同于早期较差的少步采样结果 |
| B Stage 2 W128 | short connected G 误差约 0.41978 | 直接扩展可显著扰动原范围内统计；该 W 的协议 tail 为空 |
| B Stage 2 W192 | short connected≈0.53704；tail≈0.97153，独立 MC tail≈0.07213 | 明显超出 MC 间波动的直接外推失败 |
| B Stage 2 W256 | short connected≈0.13569；tail≈0.37640，独立 MC tail≈0.01440 | 仍有明显误差；失效强度并非随 W 单调增加 |
| A Level 1 | T0→T3：expanded connected 误差约 0.1643→0.0466；short 约 0.0354→0.0541 | 某段关联改善，但不是无代价成功；当轮 NO-GO |
| A Stage 2A 局部诊断 | Dense-T3+→LG-T3：额外 MASK ΔKL 0.182165→0.019273，降低约 89.42% | 特定局部条件诊断明显改善；不是生成整体误差下降 89% |
| A Stage 2A 生成 | LG-T3 short connected NRMSE≈0.01618，expanded≈0.04918 | 支持局部保护后仍能生成合理窗口；不代表所有尺度外推成功 |
| A Stage 2B 公平主对照 | 去噪 NLL 0.475944→0.471505，降低约 0.93% | 小幅、单训练种子的距离条件化收益；不是巨大提升或 RoPE-only 证明 |

对应原始入口：

- A 旧结果：两个 `artifacts/final_*/final_summary.json` 及 correlation CSV。
- B：`results/stage1_baseline_64step/baseline_summary.csv`、`results/stage2_direct_context_extrapolation/width_metrics.csv`。
- A Level 1：`results/scale_aware_context/level1_rapid/final/level1_summary.json`，配合生成与 probe summary。
- A Stage 2A：`results/scale_aware_context/stage2a_screen/stage2a_decision.json` 与该目录的 `raw/`。
- A Stage 2B：[逐样本结果目录](https://github.com/zxfly666/ISM/tree/ed2235990fd213fcfaad30e6bde7480b08780c30/results/scale_aware_context/stage2b_causal_screen)，重点为 `per_sample.npz`，而非仅引用旧 decision。

---

## 八、必须读：已发现的解释问题与公平对照修正

### 8.1 Stage 2B 旧主对照并不公平

固定版本中的 `analyze_stage2b_causal.py` 把 `matched_minus_data_only` 定义为：

```text
Gap-Matched(correct) − Gap-Unit(correct)
```

但 Gap-Unit 在训练中使用的是单位／排序坐标，测试时突然给真实坐标，会让这个对照模型额外承受输入语义改变。于是，该差值不仅包含“匹配坐标训练的收益”，也包含“把单位坐标模型放到不匹配测试坐标下的损失”。

对比两种方法按自身训练语义使用时的表现，更合理的主对照是：

```text
Gap-Matched(correct) − Gap-Unit(unit)
```

在 `held_mix_3_6_w64`、四个 t 等权平均的归档数据上：

| 指标 | 重算结果 |
|---|---:|
| Gap-Unit(unit) NLL | 0.4759444215 |
| Gap-Matched(correct) NLL | 0.4715047225 |
| 配对差值，匹配减单位 | −0.0044396991 nat |
| 相对 NLL 降幅 | 0.9328188% |
| 配对窗口 bootstrap 95% CI | [−0.0046497，−0.0042289] |

旧对照的差值约 −0.08238，**不应再作为真实坐标带来巨大收益的主要证据**。上面的公平重算仍然是正面信号，但幅度小得多。

误差条说明：这里对固定模型和现有父场池上的 512 个窗口进行配对 bootstrap，不覆盖独立训练 seed 或独立 MC 链的全部不确定性；如果窗口共享父构型，其相关性也不能由这种简单 bootstrap 完整处理。不能把这个窄区间当成跨训练复现保证。

分别在 t=0.2、0.5、0.8、0.95 时，公平 NLL 差值约为 −0.007668、−0.005686、−0.003049、−0.001356；更接近全 MASK 时收益减小。这提醒我们继续检查无条件生成，而不是用有可见上下文的去噪任务替代它。

这项修正并不等于已经重新完成整套 GO／NO-GO 判定。完整重判还要审计其他对照、阈值和不确定性层级。原 `GO_FULL_2B` 最多表示当时建议进入更完整实验，**不表示完整确认性 Stage 2B 已完成**。

### 8.2 可直接核查的只读 NumPy 示例

以下代码在 A 仓库根目录运行，只读取已下载的 NPZ，不训练模型、不使用 GPU、不改写结果。若本地仍是 Git LFS 文本指针，需先取得对应真实文件。

```python
from pathlib import Path
import numpy as np

path = Path("results/scale_aware_context/stage2b_causal_screen/per_sample.npz")
with np.load(path, allow_pickle=False) as z:
    models = z["model_names"].astype(str).tolist()
    geometries = z["geometry_names"].astype(str).tolist()
    conditions = z["coordinate_conditions"].astype(str).tolist()
    t_grid = z["t_grid"].copy()
    nll = z["nll"].astype(np.float64)

# 维度：model, geometry, t, coordinate condition, sample
assert nll.ndim == 5
assert nll.shape[:4] == (
    len(models), len(geometries), len(t_grid), len(conditions)
)
assert np.isfinite(nll).all()
g = geometries.index("held_mix_3_6_w64")

def values(model, condition):
    return nll[models.index(model), g, :, conditions.index(condition), :]

matched = values("Gap-Matched", "correct")
unit = values("Gap-Unit", "unit")
shifted_unit = values("Gap-Unit", "correct")
paired_delta = (matched - unit).mean(axis=0)  # 先对 t 平均，保留窗口配对

print("t grid:", t_grid)
print("unit NLL:", unit.mean())
print("matched NLL:", matched.mean())
print("fair delta:", paired_delta.mean())
print("relative reduction (%):", -100 * paired_delta.mean() / unit.mean())
print("per-t delta:", (matched - unit).mean(axis=1))
print("old shifted-control delta:", (matched - shifted_unit).mean())

rng = np.random.default_rng(20260918)
bootstrap = np.empty(10000)
for b in range(len(bootstrap)):
    indices = rng.integers(0, len(paired_delta), size=len(paired_delta))
    bootstrap[b] = paired_delta[indices].mean()
print("paired-window 95% CI:", np.quantile(bootstrap, [0.025, 0.975]))
```

被核查的 `per_sample.npz` SHA-256 为：

```text
217df961d1fe92a43add6056adf9b582fc1570669f1770e89cfd3041f37b1e89
```

### 8.3 为什么即使公平重算，也不能说“证明是 RoPE 的功劳”？

在 `stage2_model.py` 中，坐标不仅输入 RoPE，也进入 `LocalPhysicalAttention._attention_bias`，决定哪些 token 被视为物理半径 1 内的局部邻居。

所以 correct→unit 或 shuffled 会同时改变：

1. RoPE 所表达的相对位置；
2. 局部分支允许连接的邻接关系。

随机间距下，某些观测点物理上并非最近邻；使用 unit 坐标却可能把它们连成“局部邻居”。这种信息的影响是方法的一部分，但它使得当前结果属于**坐标条件化架构整体**的证据，而非孤立的 RoPE 因果证据。

要分离机制，应在后续实验中分别固定邻接关系、只改变 RoPE 坐标，或固定 RoPE、只改变邻接关系；也可以在纯 dense 架构中做匹配对照。这些是后续建议，本文不表示已执行。

### 8.4 89.4% 的诊断改进到底测了什么？

Stage 2A 的量是：

\[
\Delta KL_{MASK}=KL(P_{exact}\Vert P_{model,LargeMASK})
-KL(P_{exact}\Vert P_{model,Small}).
\]

在中心点被遮蔽、其真实四个最近邻可见且相同的受控设置下，比较加入远端 MASK 后预测恶化了多少。最近邻 Ising 在给定这些邻居后具有相应的局部条件独立性，因此这个设置有精确条件概率参照。

这不是说临界系统没有长程关联，而是说**给定完整局部邻域后**，远端内容不应改变这个中心条件概率。若关键邻居也未知，则远端信息完全可能有帮助。

因此只能说该额外扰动降低约 89.4%。不能说整个生成误差降低 89.4%，也不能把接近零且区间跨零的另一项扰动包装成确定的“99% 改善”。

---

## 九、指标阅读：防止“同名不同义”

### 9.1 Raw 与 connected correlation

raw 两点关联大意是 (G(r)=\langle s_i s_j\rangle\)，connected 则扣除相应均值贡献。必须查清均值是全样本总体均值、位置依赖均值，还是逐图磁化。

B 开放窗口实现以及 A 后续开放窗口主实现采用总体均值平方的扣除；这与“每张图先减去自己的 (m^2)，再平均”不是同一个统计量。A 旧正式结果中主指标明确使用 raw G；某个通用 helper 返回 connected，不代表最终报告也使用它。

追踪方式：从最终图表／JSON 找到实际分析脚本，再找公式。不要仅凭变量叫 `correlation` 就下结论。

### 9.2 NRMSE 与 relative L2

本文涉及的这些主关联误差，在同一组 r、相同权重下使用：

\[
\frac{\sqrt{\operatorname{mean}_r(G_{model}-G_{MC})^2}}
{\sqrt{\operatorname{mean}_r G_{MC}^2}}
=\frac{\|G_{model}-G_{MC}\|_2}{\|G_{MC}\|_2}.
\]

A 后续可在 `scale_evaluation.py:correlation_band_errors` 核查。相同公式仍不意味着不同边界条件、raw／connected、r 区间、方向平均规则的数字可直接比较。

尤其不要把 A 的 short=1～8、expanded=17～32，与 B 的 short／tail 或 A 旧 L128 的 1～32／33～64 当成同一个划分。

### 9.3 能量存在一个容易漏掉的二倍归一化差异

- A 后续 `scale_evaluation.py:open_energy_density` 按开放窗口的**总键数**归一化；完全同向构型为 −1。
- B `stage1_metrics.py:open_boundary_energy` 把水平、竖直两组键乘积各取均值后相加再加负号；完全同向构型为 −2。
- A 旧周期系统的每格点能量通常也对应完全同向为 −2。

对方形开放窗口，前两者相差二倍。比较能量均值或误差前先统一定义，否则会把单位差异误读成物理错误。

### 9.4 频谱与 scaling 拟合

开放窗口需要考虑边缘／窗函数；周期格点上的 FFT 与经过 Hann 窗等处理的开放窗口低频谱不是可无条件互换的对象。

(r^{-1/4}) 是临界渐近标度参考，不是所有有限尺寸、所有距离、所有边界条件下的精确曲线。一个拟合指数更接近 1/4，不表示模型比匹配有限尺寸 MC “更正确”；应同时检查完整 G(r)、幅度、拟合区间、低频、磁化与能量等。

### 9.5 训练 loss、条件 NLL、无条件生成是三层证据

训练 loss 下降说明模型在训练目标上改善；固定遮蔽条件下 NLL 改善说明对应条件预测改善；完整生成还取决于多步误差累积、采样器和全局分布。三者相关，但不能互相替代。

同理，多个 sampling seed 反映固定模型的生成波动，不是多个独立训练。A 旧正式结果的 3×1,536=4,608 个样本，不能写成“验证了三个训练 seed”；A Stage 2B 的 512 个窗口也不是 512 个独立训练或 512 条独立 MC 链。

---

## 十、哪些资源真的在 GitHub？怎样避免下载后误判缺失？

### 10.1 A：目录里已有 LFS 归档，但部分说明过时

在本文固定 commit 的文件树中，A 包含：

- `data/level1/parents_l1024.npz` 的 Git LFS 指针，标注实际内容约 130 MB；
- `results/scale_aware_context/checkpoints/` 下 11 个 `best_val.pt` 归档入口：Level 1 四个、Stage 2A 四个、Stage 2B 三个；
- 相应运行配置、训练记录、性能记录及校验文件；
- 各阶段 summary、图表和部分逐样本／样本数组。

因此，`results/scale_aware_context/README.md` 中仍写“不包含父场／权重”的段落不能作为当前文件清单的最终依据。另一方面，**看见 LFS 指针不等于远端大文件已经在本次阅读中逐个下载、通过哈希校验**。本文核实了目录与指针，不承诺所有远端 LFS 对象均已重新验证可下载。

阅读方法与重算 NPZ 一般不需要先下载全部训练数据和权重。若仅有文本、CSV 和图表，应先完成不依赖 GPU 的阅读；需要重跑时再核查 LFS、空间、依赖和计算预算。

### 10.2 checkpoint 名字一致性是一个尚需解决的复现问题

Stage 2B `summary.json` 指向最终 `last.pt`；归档提供的是 `best_val.pt`。不能凭名称推断两者权重相同。

如果目标是精确重跑原 probe，需要核对 checkpoint 中的 step、EMA 状态和权重哈希，并与当时日志匹配。若找不到完全对应的最终权重，应明确说明“目前可复核归档输出，但尚未确认可逐位重现该次推理”，而不是随便换一个 best checkpoint 后把差异归因于方法失效。

### 10.3 B：公开结果目录不等于完整运行备份

B 的结果目录提供了配置、统计表和图，是理解结果的主要入口；`outputs/` 下完整父场、样本、checkpoint 等运行产物并非全部随仓库公开。要做严格逐样本复算或原模型重新采样，应向作者确认对应数据、checkpoint 和版本，不能把 Git clone 当成已取得全部实验资源。

### 10.4 共享范围：公开学习材料不等于完整运行备份

本共享版提供科学问题概括、固定版本阅读路线、公平对照更正与只读复算方法，不包含私人通信、服务器连接信息、凭据或完整大权重。私人原稿不属于公开材料，也不要求合作者取得它才能理解实验。

截至10月1日的新实验公开范围，以[数据可用性说明](../experiments/DATA_AVAILABILITY.md)和各实验EVIDENCE为准。本文中的旧commit是历史证据锚点；新复习文档的共享版本由Git提交记录确定，不能将这两种日期混为一谈。

---

## 十一、建议 AI 按这个顺序实际阅读

### 第一轮：快速形成准确全貌

1. 完整读本文第一节，理解老师真正要检验什么。
2. 记录两个仓库的 commit；先不运行训练、采样和服务器脚本。
3. 读 A 旧 L64／L128 summary，了解原始成功范围与外推限制。
4. 读 B 在本文固定版本中的 Stage 1 与 Stage 2 的 README、run_config、汇总 CSV。
5. 读 A 后续综合报告与三轮 decision，但立即结合本文第八节纠正解释。
6. 产出一张实验对应表：目标分布、训练尺寸、真实间距、位置机制、模型容量、checkpoint、sampler、样本量、评价范围。

### 第二轮：建立可复核证据链

按以下问题逐个找代码，而非从根目录把全部文件无差别读一遍：

| 要回答的问题 | A 优先读 | B 优先读 |
|---|---|---|
| 自旋真值从哪来？链如何分割？ | `ising_numba.py`、`generate_level1_parents.py`、数据说明 | `wolff.py`、`exact.py`、Stage 0 配置与 findings |
| 输入是周期小系统还是父场边缘？ | `scale_data.py`／`stage2_data.py` 与旧生成脚本 | `crops.py`、`stage1_data.py` |
| 坐标如何生成？是否与真实取点一致？ | `scale_data.py`、`stage2_data.py` | `masked_transformer.py` 的坐标构造 |
| 谁能与谁做 attention？ | `model.py`／`scale_model.py`／`stage2_model.py` | `masked_transformer.py` |
| MASK／PAD 是否不同？loss 怎样归一化？ | `scale_diffusion.py`、训练入口 | `masked_generation.py`、`stage1_pipeline.py` |
| 模型是 EMA 还是 raw？best 还是 last？ | 加载函数、训练日志、probe summary | 运行配置、checkpoint 选择逻辑 |
| 生成究竟使用哪个 sampler 分支？ | `diffusion.py`／`stage2_sampling.py` | `masked_generation.py` |
| 图中的误差如何计算？ | 各 `analyze_*.py`、`scale_evaluation.py` | `stage1_metrics.py`、`stage2_metrics.py` |
| 某个主要结论能否重算？ | `per_sample.npz`、raw 数据、本文 NumPy 示例 | CSV；缺逐样本原始文件则注明不可重算的部分 |

表中的包内文件均位于第四／第五节所示包目录，避免与根目录训练入口混淆。

### 第三轮：评估“早期材料验证到了哪一层”

按五个层级给结论，不要用一句“work／不 work”概括所有问题：

1. **基线层**：训练分布内能否复现目标统计？已有正面证据。
2. **失败现象层**：冻结基线直接扩展是否可能失效？两条路线都有证据，但目标分布不同。
3. **诊断改进层**：局部保护是否改善特定远端 MASK 干扰？Stage 2A 有较强筛查信号。
4. **坐标利用层**：真实匹配坐标是否帮助新几何条件预测？Stage 2B 有小幅正面筛查信号，但非 RoPE-only，且需多 seed／分层不确定性确认。
5. **完整外推层**：是否已在新方法下确认更大完整晶格的无条件生成、长程统计和跨系统迁移？目前材料不足以给出全面肯定。

以上是早期快照的证据边界；后来的G、I、Basic/NW等已针对部分缺口开展实验，应先读当前综合报告，不能将历史建议当作尚未执行的全部待办。最关键的是公平对照、坐标与邻接解耦、独立训练复现，以及匹配 MC 的完整生成评价。

---

## 十二、可直接交给另一位 AI 的任务说明

下面这段可以连同本文一起发送：

> 请阅读随附的《ISM 与 Hackathon-3 双仓库复习指南（公开共享版）》，再检查两个公开仓库：`https://github.com/zxfly666/ISM` 与 `https://github.com/zdacongming-glitch/Hackathon-3`。先解释老师关于“多尺寸、多真实距离、依据 context 中相对尺度推断”的研究设想，再整理两个仓库的实验历史和证据。请记录实际读取的 commit，区分 ISM 的旧周期模型、后续 dense RoPE 模型与 Local–Global 模型，也区分两个仓库含义不同的 Stage 2。
>
> 不要只读首页或只看图片。关键结论请追溯到运行配置、网络与数据代码、评估定义、原始表格／逐样本结果。尤其注意 ISM Stage 2B 旧对照存在输入坐标语义不匹配：主要比较应检查 Gap-Matched(correct) 对 Gap-Unit(unit)，不要直接把旧 matched_minus_data_only 当成公平主效应。还要检查坐标是否同时改变 RoPE 和局部邻接关系。
>
> 最终请输出：①研究问题的准确解释，区分科学目标与具体实现假设；②两仓库目录与建议阅读顺序；③逐阶段实验对照表；④有来源的主要结论及其局限；⑤尚缺的证据与优先级最高的下一步。不把 sampling seed 当 training seed，不把条件去噪当无条件生成，不把稀疏物理跨度当完整大尺寸 context，不把计划或 GO 字段当成已经完成的结果。文件缺失或无法访问时明确说明，不补造信息。除非另获授权，不启动训练、连接收费服务器、下载全部大文件或修改／推送仓库。

---

## 十三、交接检查清单与本说明的边界

接手者应能明确回答：

- [ ] 老师关注的是相对尺度推断，而不只是图像放大。
- [ ] 两个仓库的 Stage 编号不是跨仓库统一编号。
- [ ] A 旧模型无 RoPE，但 A 后续模型有 RoPE。
- [ ] B 的 AxialRoPE2D 不是 axial attention。
- [ ] 周期有限系统与父场开放窗口不是同一目标分布。
- [ ] Stage 2A 的训练最大宽度为 48，Stage 2B 为 64；不能抄成同一套参数。
- [ ] 89.4% 是局部 MASK 额外扰动的下降；0.93% 是修正后的条件 NLL 相对下降。
- [ ] 坐标反事实同时可能改变位置相位和局部邻接，不是纯 RoPE 干预。
- [ ] 归档 LFS 指针、真实大文件可下载性、评估 checkpoint 一致性是三件不同的事。
- [ ] 当前证据不足以宣布完整大尺度临界生成或世界模型外推已经解决。

本说明的审校原则：

| 维度 | 本文处理方式 |
|---|---|
| 研究贡献 | 将老师假设、我们实现的干预、实际结果分开 |
| 表达质量 | 先解释问题和术语，再给目录与源码阅读路线 |
| 实验强度 | 显式标出筛查、单训练 seed、采样量与公平对照 |
| 评价完整性 | 区分 loss、条件预测、无条件生成与物理统计 |
| 方法可靠性 | 提醒目标分布差异、坐标／邻接混杂、checkpoint 与 LFS 可复现性问题 |

本指南沿用此前对公开目录快照、关键报告／配置、主要模型与分析实现的核对，以及所列 Stage 2B 公平对照重算；本次共享整理没有重新执行这些计算，**没有重训两仓库，没有重跑所有 MC 或所有生成任务，也没有逐个验证全部远端权重对象**。对于未公开或未复核的部分，应保留这些边界。
