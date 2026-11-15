# 早期 ISM 与 Hackathon-3：实验设置、过程与结果对照（公开共享版）

整理日期：2026年11月15日。本文是两个固定公开版本的**历史基线比较**，不涵盖ISM后来的G、I、Basic/NW与RG全部实验。先读[完整复习计划](PROJECT_REACTIVATION_AND_ADVISOR_REVIEW_PLAN_20261104_ZH.md)与[当前综合报告](../experiments/REPORT_ZH.md)，再按需查本表；合作者的阅读路线见[双仓库指南](AI_READING_GUIDE_TWO_REPOS_PUBLIC_ZH.md)。

第15–16节保留历史研究建议，不是新实验授权、执行记录或当前待办清单。两套目标分布、训练与评价口径不同，下面的数字不能组成公平的方法排行榜。

## 1. 报告范围

本文比较两套临界二维 Ising masked discrete diffusion 实验：

- **ISM**：在完整周期 `L=64` 晶格上训练，在 `L=64` 做同尺寸生成，并将冻结模型零样本用于周期 `L=128`。
- **Hackathon-3**：从周期 `L=512` 父场中在线裁剪开放窗口，在 `W=32,48,64` 上混合训练，并将冻结模型直接用于更大的开放窗口。

版本口径固定为：

- ISM：`zxfly666/ISM@b923b037cef84daf34134a851c49a57e0bbd117b`；
- Hackathon-3：`zdacongming-glitch/Hackathon-3@3f147535389fd489775011469d338e4402eb58cb`。

下文将内容分为三类：

1. **已核实事实**：可由代码、配置或结果文件直接确认；
2. **描述性结论**：由现有指标直接支持，但不解释单一因果；
3. **机制假设**：可能解释差异，但必须通过控制变量实验验证。

## 2. 总体结论

两套实验的核心结论与此前分析一致：

1. 两边都使用 masked/absorbing discrete diffusion，而不是两类本质不同的生成范式。
2. 两边在训练尺寸内都能较好复现各自 MC reference，说明基础训练和采样链路有效。
3. 冻结模型直接扩大空间尺寸后，两边都出现长程关联偏强、衰减过慢和全局有序程度偏高。
4. ISM 在 `64→128` 上的退化明显较轻，尤其短、中距离更接近其周期 MC reference；但 ISM 仍未通过预设的长程外推检查，只能称为 **partial transfer**，不能称为正确恢复临界有限尺寸标度。
5. 当前比较同时改变了数据分布、边界条件、模型规模、attention、位置表示、训练预算、时间分布、checkpoint、采样步数和评价器，不能从现有结果确定唯一原因。

最稳妥的研究表述是：

> Masked discrete diffusion 能学习训练尺寸内的临界 Ising 分布；直接扩大 context 时，两套系统都会过度生成大尺度一致结构。在各自参考下，ISM的部分短中程描述性误差较小，但目标分布与多项配置同时不同，尚不能将差异因果归于周期几何、距离先验、轴向结构、容量或采样步数，也未解决正确的长程尺度外推。

## 3. 双方相同的实验主干

| 项目 | 共同设置 |
|---|---|
| 物理系统 | 二维、零外场、最近邻 Ising model |
| 临界逆温度 | `beta = 0.44068679350977147` |
| token | 每个格点一个二元 clean spin token，另设第三个 MASK token |
| forward corruption | 各格点独立 Bernoulli masking |
| 预测目标 | 只在 MASK 位置预测 clean spin，即 `x0` prediction |
| 核心损失 | 固定总格点数归一化、带 `1/t` 权重的 masked NELBO |
| 数据增强 | D4 旋转/反射和全局 spin flip |
| 正式采样器主干 | absorbing ancestral reverse process；已揭示 token 保持固定 |
| 正式采样温度 | logit temperature `1.0` |

因此，结果差异不能简化成“一个用了 diffusion，另一个没有”或者“一个用了 block diffusion，另一个没有”。

## 4. 研究问题与目标分布

| 维度 | ISM | Hackathon-3 | 含义 |
|---|---|---|---|
| 训练目标 | 完整周期 `L=64` Gibbs 分布 | `L=512` 父场内部开放 crop 的边缘分布 | 两者不是同一个概率分布 |
| 训练尺寸 | 单尺寸 `L=64` | 混合 `W=32,48,64` | ISM 集中拟合一个尺寸；Hackathon-3同时承担多尺寸任务 |
| 外推任务 | 周期 `L=64→128` | 开放窗口 `W=64→96,128,192,256` | 一个研究周期有限尺寸迁移，一个研究无限体积极限窗口的 context 扩展 |
| 未见距离 | `L=128` 中 `r=33…64` | 协议采用保守边界 R_seen=32，并非训练出现过的最大点对距离；保守 confirmatory tail 主要在 `W=192,256` | `W=128` 对Hackathon-3更像两倍 context 的过渡探针 |
| 边界条件 | 数据、模型先验和评价均周期一致 | 父场周期，但裁剪窗口和评价为开放边界 | 目标几何和有限尺寸修正不同 |

开放 crop 的分布是把窗口外自旋积分掉后的边缘分布：

$$
p(x_{\mathrm{crop}})=\sum_{x_{\mathrm{outside}}}
p(x_{\mathrm{crop}},x_{\mathrm{outside}}).
$$

它既不等于独立周期小晶格，也不等于简单删除边界键的有限 open-boundary Ising 模型。因此两套实验可以互相启发，但不是可直接排名的同一 benchmark。

## 5. MC 数据生成与训练数据

### 5.1 数据口径

| 项目 | ISM | Hackathon-3 |
|---|---:|---:|
| MC 对象 | 完整周期 `64×64` | 完整周期 `512×512` 父场 |
| 名义训练构型 | 15,000 个完整 `L=64` 构型 | 4,096 个选定父构型 |
| 在线训练输入 | 每次抽完整构型 | 从父场随机位置裁 `W=32,48,64` 窗口 |
| 估计在线 crop 次数 | 不适用 | 约 23.3 万次 crop 暴露 |
| 训练链 | 12 | 8 |
| 验证/测试链 | 2 / 2 | 4 / 4 |
| burn-in | 500 sweep-equivalent | 50 sweep-equivalent |
| 保存间隔 | 约 5 sweep-equivalent | 约 2 sweep-equivalent |
| 保守慢变量 ESS 代理 | 约 12,050 | 约 1,042 |

### 5.2 三种“样本数”不能混用

- `15,000` 与 `4,096` 是训练数据池中完整 MC 构型或父构型的名义数量。
- 约 `23.3 万` 是Hackathon-3在训练时反复从父场在线裁剪得到的窗口暴露次数，不代表同等数量的独立全局临界慢模样本。
- `4,608` 与 `256` 是外推完成后的模型评估样本数，与训练样本数无关。

随机 crop 会显著增加局部纹理、位置和边界环境的多样性；但多个 crop 可以共享同一个父构型的全局慢模式，所以不能把每个 crop 当成完全独立的临界样本。ESS 的约 12 倍差异只是慢变量多样性的代理诊断，也不是严格等价的有效训练集大小。

## 6. 模型结构

| 项目 | ISM | Hackathon-3 |
|---|---:|---:|
| 参数量 | 12,674,434 | 204,672 |
| 参数量比 | 约 `61.9×` | `1×` |
| block 数 | 8 | 4 |
| hidden width | 256 | 64 |
| heads | 8 | 4 |
| MLP ratio | 4 | 4 |
| dropout | 0 | 0.05 |
| attention 拓扑 | 每个 block 先做 row attention，再做 column attention | 对全部 `W²` token 做 dense full attention |
| 全局通信 | 一个 block 内经行、列两步实现 | 单层直接连接任意格点 |
| 位置/距离信息 | 周期最短距离的 learned log-distance bias | Q/K 上按横纵坐标分解的二维 RoPE |
| 局部先验 | `3×3` circular convolution stem | 无显式卷积 stem |
| RoPE | 无 | 有 |

### 6.1 必须纠正的术语

Hackathon-3的模型应描述为：

> **dense full attention + factorized/axial 2D RoPE**。

“axial”在这里描述 RoPE 分别处理横、纵坐标，不表示其 attention 只沿行列计算。真正使用 row/column axial attention 的是 ISM。

### 6.2 ISM 的周期距离先验

ISM 每个 head 的 attention bias 为：

$$
b_h(d)=-a_h\log(1+d_{\mathrm{periodic}}).
$$

忽略内容 logits 时，其指数形式为：

$$
\exp b_h(d)=(1+d)^{-a_h}.
$$

更严谨的名称是“周期最短距离上的对数距离偏置，等价于幂律形式的 attention 距离先验”。它并不直接强制最终 `G(r)` 服从某个幂律，head slope 也不等于 Ising 临界指数 `eta`。

### 6.3 context 扩大时的 normalization 差异

- ISM 从 `L=64` 到 `128` 时，每次轴向 attention 的竞争长度由 64 增至 128，即 `2×`。
- Hackathon-3的 full attention 从 `W=64` 到 `128` 时，每个 query 面对的 token 数由 4,096 增至 16,384，即 `4×`。

这使 full attention 更容易受到新增远距离 token 总量和 softmax denominator 变化的影响。不过“短程 attention 被长程污染”目前仍是合理机制假设，不是已完成的因果证明。

## 7. diffusion time 与条件注入

| 项目 | ISM | Hackathon-3 |
|---|---|---|
| 训练 `t` 范围 | `[0.01,1.0]` | `[0.2,1.0]` |
| 强制全 MASK | 2% 训练样本精确设为全 MASK | 无额外 endpoint atom |
| 时间表示 | sinusoidal embedding + MLP | 标量 mask ratio + 两层 MLP |
| 注入位置 | 每个 attention/MLP 前的 time-conditioned AdaLN | 主要在输入 residual stream 中相加 |
| 采样低端处理 | 模型条件 clamp 到 `t_min=0.01` | schedule 可进入训练未覆盖的 `t<0.2` |

ISM 训练包含精确全MASK起点及接近clean的低遮盖率条件；这不是对已生成自旋进行回改的纠错训练。Hackathon-3在正式采样的开头与末尾存在更明显的 train–sampler support mismatch。不过Hackathon-3在 `W=64` 仍表现良好，因此这不可能是全部原因，只可能在 context 扩大后放大误差。

## 8. 优化过程与训练预算

| 项目 | ISM | Hackathon-3 |
|---|---:|---:|
| optimizer updates | 60,000 | 50,000 |
| batch / accumulation | 16 个完整 `64×64`；无梯度累积 | 尺寸相关 microbatch；2 次梯度累积并轮换宽度 |
| peak LR | `3e-4` | `1e-4` |
| minimum LR | `3e-5` | `1e-5` |
| warmup | 2,000 | 1,000 |
| Adam betas | `(0.9,0.95)` | `(0.9,0.95)` |
| weight decay | 0.05 | 0.01 |
| gradient clip | 1.0 | 1.0 |
| precision | BF16 | BF16 |
| 估计 site presentations | `3.93×10^9` | `4.27×10^8` |
| site 暴露比 | 约 `9.2×` | `1×` |
| EMA | decay `0.999` | 无 |
| 最终采样权重 | EMA best checkpoint | raw best checkpoint |
| best update | 55,000 | 46,500 |

`9.2×` 是处理过的格点暴露量，不应表述为严格的“数据量 9.2 倍”。Hackathon-3的训练信号还被 `W=32,48,64` 三种尺寸分摊，而 ISM 只优化 `L=64`。

## 9. 验证与 checkpoint 选择

| 项目 | ISM | Hackathon-3 |
|---|---|---|
| validation 目标 | 固定周期 `L=64` | `W=32,48,64` 三尺寸汇总 |
| validation 数量 | 256 个固定构型 | 每个宽度 128 个固定 parents/crops |
| time grid | 13 点：`0.01…1.0`，在两端更密 | 4 点：`0.3,0.5,0.7,0.9` |
| checkpoint | 单尺寸验证最优 EMA | 多尺寸验证最优 raw 参数 |

双方 best validation 的绝对数值不能直接比较，因为目标分布、尺寸和汇总方式不同。

## 10. 正式 reverse sampling

| 项目 | ISM | Hackathon-3 |
|---|---|---|
| 初始状态 | 全 MASK | 全 MASK |
| sampler | ancestral absorbing reverse | ancestral absorbing reverse |
| 正式 steps | 128 | 64 |
| 时间 schedule | cosine-squared | cosine |
| reveal kernel | 仍为 MASK 的格点以 `1-s/t` 概率独立揭示 | 相同主干 |
| 已揭示 token | 固定，不再回改 | 固定，不再回改 |
| corrector | 正式结果未使用 | 正式 baseline 未使用 |
| checkpoint | EMA | raw |
| temperature | 1.0 | 1.0 |

因为已揭示 token 会被锁定，早期错误无法回改。128 steps 每一步揭示得更少，理论上有利于降低单步错误锁定，但对Hackathon-3的同一 checkpoint 尚未做严格 64-vs-128-step 对照，不能提前断言这一因素能修复多少误差。

## 11. 正式评估协议

| 项目 | ISM `L=128` | Hackathon-3 `W=128` |
|---|---:|---:|
| 模型样本 | 4,608 | 256 |
| sampling seeds | 3 个，每个 1,536 | 4 个，每个 64 |
| MC reference | 10,000，8 条链 | 256 target crops |
| MC noise control | chain-level bootstrap | 额外 256 independent control crops |
| correlation | 周期 FFT、toroidal radial bins，主报 raw `G(r)` | zero-padded open estimator，同时报 raw `G` 与 connected `G_c` |
| energy | 包含跨边界键的周期能量 | 不加跨窗口伪键的开放边界能量 |
| structure factor | 标准周期 FFT | Hann taper 后的 open-window low-k spectrum |
| `xi/L` | 直接计算周期 second-moment `xi/L` | 不直接使用周期公式 |
| 不确定性 | MC chain bootstrap；模型 seed/sample 两层 bootstrap | pooled estimate、seed variation、independent MC control |

较大的评估样本数只会让 ISM 的曲线更平滑、误差条更窄，不能单独解释两边系统偏差的巨大差距。Hackathon-3在 `W=128` 的 MC-vs-MC connected error 约 `0.0083`，远小于 model-vs-MC 的约 `0.4198`，因此主要失败不是 256 个样本造成的统计噪声。

## 12. 结果对照

### 12.1 训练尺寸内

| 指标 | ISM / 周期 `L=64` | Hackathon-3 / 开放 `W=64` |
|---|---:|---:|
| correlation error | raw `G`, `r=1…16` NRMSE `0.0040` | connected `G_c` short-range relative L2 `0.0181` |
| MC noise control | 由 bootstrap 给出 | MC-vs-MC `0.0164` |
| `G_model/G_MC` | `r=8,16,32`: `0.9975,0.9994,0.9978` | 未用完全相同口径报告 |
| `xi/L` | model `0.899` / MC `0.901` | open protocol 不直接计算 |
| 结论 | PASS | PASS / 接近 MC control |

两个数值不是严格排行榜，因为 periodic raw estimator 与 open connected estimator 不同。可信的共同结论只是：**两边在各自训练尺寸内的指定物理统计上达到原定质量要求；这不等于完整联合分布精确正确。**

### 12.2 两倍尺寸外推

| 指标 | ISM / 周期 `L=128` | Hackathon-3 / 开放 `W=128` |
|---|---:|---:|
| raw correlation relative error | `0.1217`，`r=1…32` | `0.4473`，`r=2…32` |
| connected correlation error | 未作为主汇总值 | `0.4198` |
| `G_model/G_MC`, `r=8` | `1.081` | `1.306` |
| `G_model/G_MC`, `r=16` | `1.137` | `1.482` |
| `G_model/G_MC`, `r=32` | `1.198` | `1.718` |
| 拟合 `eta`, `r=4…32` | model `0.126` / MC `0.198` | model `0.050` / MC `0.237` |
| `E|m|` | model `0.609` / MC `0.550` | model `0.712` / MC `0.500` |
| `xi/L` | model `1.196` / MC `0.905` | 不直接计算 |
| 正式判定 | PARTIAL；预设 zero-shot checks 未全部通过 | FAIL；明显 context extrapolation 退化 |

ISM 在真正未见距离 `r=33…64` 的 correlation NRMSE 为 `0.2209`，并且 `G_model/G_MC` 从 `r=48` 的 `1.223` 增至 `r=64` 的 `1.232`。这说明 ISM 保留了定性的长程关联，却没有恢复正确的幅度和衰减速度。

Hackathon-3在 `W=192,256` 上的预定非空tail也明显偏离各自MC参考，但失效强度不随尺寸单调增加。这仍是同一训练checkpoint的评价，不能据此声称已跨多个训练seed复现。

## 13. 结果为何呈现这种形态

两边的共同失败模式是“过度全局有序”：模型样本的 `E|m|` 高于 MC，导致许多远距离格点更容易处于同一个大尺度正/负畴中，从而整体抬高 `G(r)`。拟合 `eta` 过小则表示曲线下降过慢。

ISM 的局部能量误差很小，但 `xi/L` 偏高，说明“最近邻规律正确”不等于“非零低波数模式与全局尺度层级正确”。这也是为什么必须同时评价：

- energy；
- raw/connected `G(r)`；
- `|m|` 与 Binder cumulant；
- 零模和非零 low-k structure factor；
- `xi/L` 或适配开放几何的替代量；
- 多个尺寸上的 finite-size scaling。

## 14. 哪些结论已经成立

### 14.1 现有证据直接支持

1. 两边训练尺寸内的指定统计通过各自原定质量检查。
2. 两边直接扩大 context 后均过度生成长程一致性。
3. ISM 在各自 reference 下的短、中程误差明显小于Hackathon-3的 baseline。
4. ISM 仍未正确恢复 `L=128` 长程有限尺寸标度。
5. 256 个评估样本增加了Hackathon-3的估计噪声，但不足以解释约 42% 的系统 correlation error。

### 14.2 尚未得到因果证明

1. 周期 log-distance bias 是否是主要改进来源；
2. axial attention 是否比 dense attention 更能抵抗 context pollution；
3. 62 倍参数量究竟贡献多少；
4. 128 steps、EMA、局部卷积、`t_min` 和 endpoint atom 各贡献多少；
5. 周期完整构型是否天然优于父场 crop；
6. 多尺寸训练是否改善或损害外推；
7. RoPE 本身是否是失败原因。

因此不能声称：

- “crop pipeline 是错误的”；
- “RoPE 不适用于 Ising”；
- “distance bias 已被证明是唯一原因”；
- “ISM 已经学会正确长程临界物理”。

## 15. 差异因素的研究优先级

以下是下一步最值得控制的变量，不是已经证实的因果排名。

### 第一优先级：必须匹配

1. 数据目标和边界条件；
2. 参数量、updates 和 site-level 训练预算；
3. 正式 sampler steps、time schedule、EMA 与 `t` 支持范围；
4. 统一 raw/connected correlation 和 low-k 评价器。

### 第二优先级：检验老师提出的机制

1. dense full attention 在 context 扩大时是否发生短程输出漂移；
2. 单尺寸与多 block/window size 训练的影响；
3. RoPE coordinate spacing / physical-distance rescaling；
4. 不规则区域或随机空间 mask；
5. axial 或稀疏 attention 是否降低新增远距离 token 的污染。

### 第三优先级：模块消融

1. periodic log-distance bias；
2. `3×3` circular convolution；
3. exact full-MASK endpoint；
4. time-conditioned AdaLN；
5. EMA。

## 16. 建议的阶梯式实验

### 阶段 0：不重训的统一复算

1. 双方交换正式生成样本和 MC reference 原始数组。
2. 用同一脚本同时复算 raw `G`、connected `G_c`、`E|m|` 和 low-k spectrum。
3. 明确区分 periodic 与 open geometry，不跨边界口径比较绝对数值。

### 阶段 1：不重训的 sampler 对照

对Hackathon-3的同一 `W=128` checkpoint：

- 比较 64 与 128 steps；
- 固定 temperature、seeds 和样本数；
- 将样本数提升至至少 1,024，理想为 4,608；
- 同时报告生成耗时和显存。

这一步回答采样器和统计精度能解释多少差距，但不能证明架构因果。

### 阶段 2：匹配预算的架构消融

在同一数据、参数量、updates、EMA、`t` 范围和 128-step sampler 下比较：

| 组别 | attention | 位置/距离 | local stem | 研究问题 |
|---|---|---|---|---|
| A | dense full | 2D RoPE | 无 | matched baseline |
| B | axial | 无距离 bias | 无 | 只测试 topology |
| C | axial | periodic log-distance bias | 无 | 测试距离先验增量 |
| D | axial | periodic log-distance bias | circular conv | 测试完整 ISM 结构 |

必须从头训练 bias-on/off 模型；不能只在已训练模型上临时删除 bias 后把下降归因于该模块。

### 阶段 3：直接检验老师的多尺度想法

| 组别 | 多尺寸 | RoPE/坐标距离 rescale | 不规则 mask | 研究问题 |
|---|---:|---:|---:|---|
| T0 | 否 | 否 | 否 | 当前固定尺度 baseline |
| T1 | 是 | 否 | 否 | 多 block/window size 是否改善 context robustness |
| T2 | 否 | 是 | 否 | 模型能否按相对物理距离而非 token 间隔工作 |
| T3 | 是 | 是 | 否 | 尺寸多样性与尺度 rescale 是否协同 |
| T4 | 是 | 是 | 是 | 不规则/稀疏空间配置的泛化 |

这些是历史候选对照，不是唯一合法的研究次序。讨论下一步前，应先核对后来G、I、Basic/NW与RG已经完成的检验及其限制。

## 17. Claim–evidence 对照

| Claim | Evidence | 状态 |
|---|---|---|
| 两边训练尺寸内指定统计达标 | `L=64`/`W=64` correlation 接近各自 MC control | supported |
| 两边外推后均过关联 | `G_model/G_MC>1` 且随 `r` 增大，`eta` 偏小，`E|m|` 偏大 | supported |
| ISM 的短中程结果更接近各自 reference | raw correlation error `0.1217` vs `0.4473` | supported，但任务/几何不完全相同 |
| ISM 已正确学习长程临界标度 | 长程 NRMSE、`eta`、`xi/L` 均未通过 | contradicted |
| distance bias 是主要原因 | 多变量同时变化，尚无 matched ablation | needs experiment |
| RoPE 导致Hackathon-3的失败 | 当前只观察到关联，未控制容量、sampler 和数据 | needs experiment |
| 评估样本少导致Hackathon-3的失败 | MC control noise 远小于 model error | contradicted as primary explanation |

## 18. 最终建议用语

对老师或合作者汇报时，推荐使用：

> 两套实现共享相同的 masked discrete diffusion 主干，但实验目标和预算差异很大。ISM 使用完整周期 `L=64` 数据、周期距离先验、行列轴向 attention、局部循环卷积、约 1267 万参数、EMA 和 128-step sampler；Hackathon-3 使用 `L=512` 父场的开放 crop、多尺寸训练、dense full attention 与二维 RoPE、约 20.5 万参数、raw checkpoint 和 64-step sampler。两边在训练尺寸内均接近各自 MC reference；直接扩大尺寸后均出现长程关联过强。ISM 在 `64→128` 的短中程误差较小，但仍未恢复正确的长程有限尺寸标度。由于多项变量同时改变，下一步需要统一评价与采样预算，并分别消融 attention topology、距离表示、多尺寸训练和 RoPE 距离 rescale。

## 19. 自审结论

- **清晰性**：训练样本、在线 crop 和评估样本已分开定义。
- **术语一致性**：区分 dense full attention、axial attention 与 factorized 2D RoPE。
- **证据边界**：描述性结果与因果推测已分开。
- **实验公平性**：明确两套目标分布和评价器并不完全相同。
- **缺失证据**：关键架构结论仍需 matched-budget ablation。
