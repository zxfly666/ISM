# Level 2：尺度坐标与局部—全局物理解耦实验方案

版本：v1.0<br>
状态：Stage 2A 与单 seed Stage 2B 因果筛选已执行；完整 3-seed 正式确认待执行
前置结果：`docs/LEVEL1_RAPID_RESULT_REPORT_ZH.md`

最新机器可读结果和图表：`results/scale_aware_context/README.md`

---

# 一、方案摘要

## 1. 核心研究问题

第一级已经证明，训练时同时随机化物理采样 stride 与外部二维坐标，能够让离散扩散模型使用真实尺度信息，并显著改善中长程相关；但当前 dense attention 同时损害了最近邻能量、短程 `G(r)` 和精确 Markov 条件概率。

因此，第二级不再重复回答“尺度坐标是否有作用”，而是回答：

> 能否把局部物理预测与全局尺度修正解耦，使模型保留 scale-aware coordinates 的长程收益，同时避免全局 context 污染局部规律？

## 2. 核心方法假设

当前 dense T3 使用同一隐藏状态同时承担局部与全局推断。模型为了适应稀疏物理 stride，会调整所有 attention 层的表示；这些全局表示随后也参与最近邻预测，造成局部 posterior 偏离精确 Markov oracle。

本方案提出一个参数受控的 Local–Global Scale Denoiser：

- Local expert 只能访问有限物理半径内的 token；
- Global expert 使用 matched randomized physical coordinates 与二维 RoPE；
- timestep-conditioned gate 控制全局修正对最终 logits 的贡献；
- local logits 构成基本预测，global logits 只做尺度相关残差修正。

## 3. 实验路线

```text
Stage 0   sampler closure（不重训）
   ↓
Stage 2A  2×2 单 seed 因果修复实验
   ↓ 只有通过硬门槛
Stage 2B  3 seeds + RandomGap 因果实验 + W=96/128
   ↓ 只有通过
Stage 2C  irregular geometry + sparse/axial attention + 跨系统
```

第二级的首要目标不是追求最大的长程改进，而是证明“长程收益与短程损害可以被结构性解耦”。

---

# 二、与现有工作的关系

随机位置编码并非全新概念。Randomized Positional Encodings 会在更大的位置范围内随机选择有序位置子集，以改善 Transformer 的长度泛化；Position Coupling 和 position-index warping 也具有相近思想。

本项目必须明确区别于这些工作：我们不仅替换位置编号，还同步改变输入随机场的真实物理采样算子，即自旋来自 `x0 + s*i`，坐标也为 `s*i`。方法目标是学习跨离散尺度的生成分布，而非仅避免新的 position ID。

Neural Operator 已把 discretization invariance 与 zero-shot super-resolution 作为核心目标；近期 lattice-aware Transformer 也已研究小晶格到大晶格的复用。因此，单独在 Ising 上展示 size extrapolation 不足以构成强创新。

本方案把贡献聚焦为三点：

1. joint randomization of physical sampling and coordinates；
2. local-fidelity/global-extrapolation trade-off 的因果诊断；
3. 能显式保护局部物理的 scale-aware local–global generative architecture。

参考：

- Randomized Positional Encodings: <https://arxiv.org/abs/2305.16843>
- YaRN: <https://arxiv.org/abs/2309.00071>
- LongRoPE: <https://arxiv.org/abs/2402.13753>
- Longformer: <https://arxiv.org/abs/2004.05150>
- Neural Operator: <https://arxiv.org/abs/2108.08481>
- Scaling Autoregressive Models for Lattice Thermodynamics: <https://arxiv.org/abs/2603.14695>
- MaskGIT: <https://arxiv.org/abs/2202.04200>

---

# 三、预注册假设

## H1：坐标可辨识性

对于 held-out physical stride `s_true ∈ {3,6}`，LG-T3 的 NLL 关于输入 coordinate scale `s_coord` 应在 `s_coord≈s_true` 附近达到最小，而不是只对任意较大的坐标都给出类似结果。

## H2：局部保护

LG-T3 在 ID `W=48,s=1` 上应恢复 T0 的短程 `G(r)` 和能量，同时将 Large-MASK/Visible Markov pollution 至少降低到当前 T3 的一半；理想目标是不高于 T0。

## H3：长程保持

LG-T3 在 `W=64,s=1` 生成中应至少保留当前 T3 70% 的 expanded-context 改善，并在 `r=17...32` 明显优于 T0。

## H4：坐标的独立贡献

在相同 Local–Global 架构下，LG-T3 必须优于 LG-Punit；否则改善主要来自多尺度自旋数据，而不能归因于 matched physical coordinates。

## H5：架构—坐标交互

Local–Global 架构不应只是统一提升所有模型。它应特别减少 scale-aware 模型的局部污染，并保留正确坐标带来的 OOD 收益。

## H6：采样结论稳定

主结论不应依赖单一 sampler step。模型排名和主要物理指标应在预先冻结的 sampler 设置下稳定。

---

# 四、Stage 0：sampler closure

## 1. 动机

第一级中，T0 的 expanded `G(r)` NRMSE 从 64-step 的 `0.164` 变为 128-step 的约 `0.398`。虽然 T3 相对 T0 的排名没有反转，但 reveal-only sampler 对 step 数明显敏感。

如果不先处理这一混杂因素，第二级生成结果可能把 sampler 偏差错误归因于模型架构。

## 2. 不重训模型

使用第一级的 T0、T3、Punit checkpoint，仅在 validation MC split 上比较：

| 编号 | sampler | 说明 |
|---|---|---|
| S0 | irreversible reveal | 当前 sampler |
| S1 | confidence remasking | 每轮重新 mask 最低置信 token |
| S2 | S1 + blocked refinement | 完整生成后做少量模型 pseudo-Gibbs 修正 |

S1 采用 MaskGIT 风格的迭代：

1. 对全部 masked site 预测；
2. 采样候选 token；
3. 按置信度只保留计划数量的 token；
4. 低置信 token 重新 MASK；
5. 使用 cosine remaining-mask schedule。

S2 在 S1 后增加 `K∈{4,8,16}` 次 refinement，每次随机或按低置信度重新 mask `5%` site 并重采样。禁止使用真实 Ising Hamiltonian 做修正，否则生成质量将混入物理 oracle。

## 3. 调参和冻结

只在 validation MC 上比较：

```text
steps = {32, 64, 128, 256}
samples/model/setting = 128
seeds = {1234, 2345}
```

选择规则按优先级：

1. 64/128/256 的主要物理指标不出现系统漂移；
2. energy、short/medium/expanded `G(r)` 的综合 rank 最优；
3. 时间更短；
4. 不允许分别为不同模型选择不同 sampler。

选定 sampler 后冻结，在 test-target 上只运行一次。

## 4. Stage 0 停止条件

若三类 sampler 均无法给出稳定生成，则第二级仍可继续训练与 NLL/Probe 评估，但生成结果降为次要证据，不用于决定方法是否成功。

---

# 五、Stage 2A：核心 2×2 因果实验

## 1. 因子设计

两个实验因子：

```text
Factor A: architecture ∈ {Dense, Local–Global}
Factor B: coordinate semantics ∈ {Unit, MatchedPhysical}
```

得到四个主模型：

| 模型 | 自旋 stride | 坐标 stride | 架构 | 状态 |
|---|---|---|---|---|
| Dense-Punit | `{1,2,4}` | `1` | 当前 dense | 可复用第一级 |
| Dense-T3 | `{1,2,4}` | matched | 当前 dense | 可复用第一级 |
| LG-Punit | `{1,2,4}` | `1` | 新 Local–Global | 新训练 |
| LG-T3 | `{1,2,4}` | matched | 新 Local–Global | 新训练 |

保留以下参考，不进入主 2×2 ANOVA：

- T0：固定 `W=48,s=1` 的 ID reference；
- Pphase：判断纯坐标 phase augmentation；
- MC target/control：测量有限样本噪声。

如果新架构参数量比 dense 高出超过 10%，增加一个 parameter-matched Dense-T3+，避免把容量提升误判为架构贡献。

## 2. 为什么不同时修改数据范围

Stage 2A 严格复用第一级的父场、W 分布、stride 分布、训练步数和验证样本。唯一新因素是 Local–Global 架构。

此阶段禁止同时加入：

- `s=8/10` 训练；
- `W=64` 训练；
- 不规则 geometry；
- sparse/axial attention；
- physics oracle loss。

否则无法知道局部修复来自哪个改动。

---

# 六、Local–Global Scale Denoiser

## 1. 总体结构

每个 site 同时经过两个相互独立的预测路径：

```text
noisy spin tokens + time
        ├── Local Expert  ───────────────┐
        └── Global Scale Expert + 2D RoPE ─ gate ─┤→ logits
```

最终 logits：

\[
\ell_i
=
\ell_i^{\mathrm{local}}
+
g_i(t)\,\Delta\ell_i^{\mathrm{global}}.
\]

Local expert 给出基本条件概率；Global expert 只学习局部路径不能解释的长程残差。

## 2. Local expert

Local expert 采用物理半径限制的 window attention：

\[
M_{ij}^{\mathrm{local}}
=
\mathbf 1\{\|x_i-x_j\|_1\le R_{\mathrm{phys}}\}.
\]

推荐：

```text
d_local = 64
blocks_local = 4
heads_local = 4
R_phys = 1
external 2D relative/RoPE coordinates = on
```

为防止多层局部 attention 产生感受野扩张，Local expert 只在第一层执行一次 Manhattan 半径 1 的空间 attention；其余三层是逐 site MLP residual，不再传播空间信息。因此局部分支的总感受野严格保持为中心及四个真实最近邻，而不是随深度从 1 扩张到 4。

关键点是半径按物理坐标计算，而不是按 token index 计算：

- `s=1` 时，local expert 能看到真实最近邻；
- `s=3` 时，相邻 token 的物理距离为 3，不会被错误当成最近邻；
- irregular geometry 下同一规则仍成立。

Local stream 不接收 Global stream 的隐藏状态，从结构上避免远处 token 写入局部表示。

## 3. Global scale expert

复用第一级 CoordinateDenseDenoiser：

```text
d_global = 128
blocks_global = 6
heads_global = 4
2D RoPE base = 10000
physical coordinates = raw, not normalized by W
```

Global expert 继续使用 matched physical coordinates，不额外输入显式 stride label。这样模型必须从坐标差中学习尺度，而不是读取一个手工提供的 `s` 类别。

## 4. Gate

Gate 使用 time embedding、local hidden 和 global hidden：

\[
g_i(t)
=
\sigma\left(\mathrm{MLP}([h_i^L,h_i^G,e_t])\right).
\]

推荐初始化使 `g≈0.1`，避免训练初期 global expert 破坏局部预测。Gate 可以逐 site 变化，但第一版不加入显式 scale supervision。

Global residual 的最后一层采用 zero initialization。这样模型初始状态严格接近 local-only predictor；只有训练证据支持时，global path 才逐渐产生非零修正。第一版不使用手工 gate 标签，也不按测试尺度调 gate 阈值。

必须保存：

- 各层/各 t 的平均 gate；
- gate 对 physical stride 的条件分布；
- gate 与 mask fraction、局部邻居可见数的关系。

另预注册一个只用于诊断、不能进入主 GO 判定的 `LG-HardMarkov` 变体：当目标 site 的四个真实最近邻全部可见时，强制 `g_i=0`。如果 soft gate 失败而该诊断变体成功，说明问题主要在 gate 学习；如果二者都失败，说明 local expert 或训练目标本身尚未学好局部条件分布。该变体利用了 Ising 的已知 Markov 结构，因此不能作为跨系统方法的主结果。

## 5. 参数公平性

主表同时报告：

- 参数量；
- FLOPs/update；
- updates/s；
- peak VRAM；
- sampling seconds/sample。

如果 LG 模型增加参数，parameter-matched Dense-T3+ 使用更宽 dense hidden dimension，使总参数误差控制在 5% 内。

---

# 七、数据和训练设置

## 1. Stage 2A 数据

复用：

```text
parent lattice       L=1024
train chains         6
validation chains    2
test-target chains   2
test-control chains  2
W                    {16,24,32,48}
physical stride      {1,2,4}
held-out stride      {3,6}
```

所有模型使用相同 parent indices、origin、W、stride、D4、spin flip、t 和 MASK 随机流。对于 Unit/Matched 配对模型，只允许 coordinate tensor 不同。

## 2. 训练超参数

```text
updates                    8,000 screen / 15,000 confirm
optimizer                  AdamW
learning rate              3e-4
minimum learning rate      3e-5
warmup                     500 updates
betas                      (0.9, 0.95)
weight decay               0.05
gradient clipping          1.0
EMA                        0.999
precision                  BF16 train, FP32 mechanism probes
full-mask probability      0.02
```

Batch sizes继续保持每个 update 的有效 token 数近似一致。Loss 必须按有效 site 归一化，而不是按样本归一化。

## 3. 初始化公平性

- LG-Punit 与 LG-T3 完全相同初始化；
- Dense-Punit 与 Dense-T3 完全相同初始化；
- 记录 common/shared parameter hash；
- 新 gate/local branch 的 hash 单独记录；
- 对所有主实验固定 geometry 与 corruption trace manifest。

## 4. Stage 2A seed

第一轮仅使用 `seed=1234` 作架构筛选。单 seed 不能支持最终有效性表述，只能决定是否进入 Stage 2B。

---

# 八、评估一：训练与 ID guard

## 1. 固定验证集

扩大固定验证样本，避免第一级 16/32 个 validation crop 过少：

```text
ID W=48,s=1                    512 crops
OOD W=24,s=3                  2,048 crops
OOD W=24,s=6                  2,048 crops
t grid                        {0.2,0.5,0.8,0.95}
```

所有模型使用逐 site 相同的 clean/noisy/MASK trace。

## 2. 指标

- masked NLL；
- Brier score；
- calibration error；
- per-t NLL；
- per-geometry NLL；
- training/validation learning curves。

## 3. ID non-inferiority

LG-T3 必须满足：

```text
anchor NLL(LG-T3) - anchor NLL(T0) <= 0.003
```

同时报告 paired bootstrap CI；固定 `0.003` 是 practical margin，不用显著性检验替代效应量。

---

# 九、评估二：连续坐标响应曲线

## 1. 动机

第一级只比较 Correct、Unit 和一个 WrongScale。更强的证据应显示：模型 NLL 随输入 coordinate scale 形成可解释曲线，并在真实尺度附近达到最优。

## 2. 设置

固定 physical stride：

```text
s_true = {3,6}
W = 24
samples/stride = 2,048
t = {0.2,0.5,0.8,0.95}
s_coord sweep = {1,1.5,2,2.5,3,4,5,6,8,10}
```

相同 clean spin、MASK 和 t，仅替换坐标。

## 3. 主指标

- `NLL(s_coord)` 曲线；
- argmin scale；
- correct-scale regret；
- 曲线在 `s_true` 附近的局部曲率；
- LG-T3 vs LG-Punit；
- Dense-T3 vs LG-T3。

## 4. 坐标可辨识通过条件

对 `s_true=3,6`：

- NLL 最小值位于 true scale 或相邻 sweep point；
- CorrectCoord 优于 UnitCoord，paired CI 全部小于 0；
- CorrectCoord 优于 LG-Punit，平均 NLL 改善至少 `0.002`，或给出清楚的 practical effect 解释；
- 方向在至少 3/4 t bins 中一致。

---

# 十、评估三：Markov contamination scaling curve

## 1. 扩展第一级 Probe

不再只比较 Small 与 W=48，而是测量污染随 context size 的增长：

```text
context W = {3,8,16,24,32,48,64}
context type = {PAD, MASK, Visible}
centers = 8,192
t = {0.2,0.5,0.8,0.95}
precision = FP32
```

中心 MASK，四个最近邻强制可见，oracle：

\[
p_i^\star
=
\operatorname{sigmoid}
\left(2\beta\sum_{j\sim i}\sigma_j\right).
\]

## 2. 指标

- oracle KL；
- soft-label CE；
- Brier；
- logit drift；
- `Delta_MB(W)`；
- pollution slope 对 `log(number of context tokens)` 的回归；
- per-t 和 per-neighbour-sum 分层。

## 3. 局部保护通过条件

必须同时满足：

```text
Large-MASK Delta_MB(LG-T3) <= 0.064 + paired uncertainty
Large-Visible Delta_MB(LG-T3) <= 0.003 + paired uncertainty
```

若无法达到 T0，则最低可接受条件为：相对当前 Dense-T3 的污染下降至少 50%，且 short `G(r)` 与 energy 同时通过。

---

# 十一、评估四：W=64 生成

## 1. 设置

```text
models                     T0, Dense-Punit, Dense-T3, LG-Punit, LG-T3
W                          64
samples/model/seed         256 screen
sampler                    Stage 0 frozen sampler
sampler seeds              {1234,2345,3456}
MC target/control          1,024 crops each
boundary                   open
```

## 2. 指标

- open energy；
- signed/absolute magnetization；
- raw 与 ensemble-connected `G(r)`；
- structure factor `S(k)`；
- short `r=1...8`；
- medium `r=9...16`；
- expanded `r=17...32`；
- low-frequency power；
- observable distribution Wasserstein distance；
- sample diversity；
- sampler convergence。

## 3. 短程门槛

```text
short NRMSE(LG-T3) <= 0.040
abs energy error(LG-T3) <= 0.032
```

这两个阈值略宽于第一级 T0，但明显优于当前 Dense-T3。

## 4. 长程门槛

当前 T0 与 Dense-T3 expanded NRMSE 分别为 `0.164` 与 `0.0466`。

LG-T3 必须满足：

```text
expanded NRMSE <= 0.080
```

这相当于至少保留约 70% 的当前 T3 长程改善。理想目标为不劣于 `0.055`。

---

# 十二、评估五：gate 与 attention 机制

为避免方法只是一组黑箱数字，保存：

- gate 随 t 的曲线；
- gate 随 physical stride 的曲线；
- gate 随可见邻居数的曲线；
- local/global logit norm；
- local-only、global-only、full 的 NLL 与物理指标；
- 各 head attention mass 的物理距离径向分布。

预期机制：

- `s=1` 且局部邻居可见时，local contribution 较大；
- 高 mask fraction 或 sparse stride 时，global correction 较大；
- LG-T3 的 global contribution 对 CorrectCoord 敏感；
- local-only Markov posterior 接近 oracle，full prediction 不显著偏离。

如果观察不到这些模式，即使最终指标改善，也只能称为经验架构改进，不能宣称实现了预期解耦机制。

---

# 十三、Stage 2A 判定规则

## 1. INVALID

以下任一失败，修复后重跑：

- MC split/chain 泄漏；
- FP32 PAD invariance > `1e-3`；
- paired trace 不一致；
- 参数/初始化审计失败；
- NaN/OOM；
- sampler 在验证阶段无法冻结；
- checkpoint 或逐样本结果缺失。

## 2. STRONG GO

必须全部满足：

- ID non-inferiority 通过；
- coordinate response curve 在 true scale 附近最优；
- Markov pollution 不高于 T0；
- short NRMSE 与 energy 通过；
- expanded NRMSE `<=0.055`；
- LG-T3 优于 LG-Punit；
- sampler 结论稳定。

## 3. GO

- ID、coordinate、short/energy 全部通过；
- Markov pollution 相对 Dense-T3 至少降低 50%；
- expanded NRMSE `<=0.080`；
- LG-T3 优于 LG-Punit。

## 4. CONDITIONAL GO

只允许两种情况：

1. 局部已修复但长程只保留 40%–70%，需要调 gate；
2. 长短程均改善，但 LG-T3 与 LG-Punit 差异低于 practical threshold，说明主要贡献是多尺度数据而不是坐标。

## 5. NO-GO

- 局部污染未明显降低；或
- 局部修复完全牺牲长程收益；或
- 正确坐标没有独立贡献；或
- 只有通过大量 per-case 调参才能成功。

---

# 十四、Stage 2B：正式确认

只有 Stage 2A 为 GO/STRONG GO 才执行。

## 1. 多 seed

```text
seeds = {1234,2345,3456}
updates = 15,000
models = LG-U-Unit, LG-U-Matched, LG-Gap-Unit, LG-Gap-Matched
```

主比较使用 hierarchical bootstrap：先 resample model seed，再 resample MC chain，再 resample crop/sample。不得把同一链中的 crop 当作完全独立观测。

## 2. 正式因果设计：均匀尺度与样本内随机距离

只在每个 batch 中抽一个统一 stride，仍可能让网络记住有限的尺度类别；它不等价于项目所研究的“根据 context 中的相对距离推断”。因此 Stage 2B 在已经通过 Stage 2A 的 Local–Global 架构上再做一个 2×2：

```text
Factor G: physical geometry ∈ {UniformStride, RandomGap}
Factor C: coordinate semantics ∈ {Rank/Unit, MatchedPhysical}
```

四个模型使用相同参数量、优化步数、mask trace 和 MC chain split：

| 模型 | 自旋取样位置 | 输入坐标 | 目的 |
|---|---|---|---|
| LG-U-Unit | 每个样本统一 stride | `0,1,...,W-1` | 无真实尺度基线 |
| LG-U-Matched | 每个样本统一 stride | 真实物理位置 | 复现 Stage 2A 尺度收益 |
| LG-Gap-Unit | 同一样本内间距变化 | rank 坐标 | 区分数据增广本身 |
| LG-Gap-Matched | 同一样本内间距变化 | 真实物理位置 | matched-distance 主模型 |

`RandomGap` 不伪造连续 Ising 格点。对每个轴从父场中选严格递增的整数位置：

\[
x_{i+1}=x_i+\Delta x_i,\qquad
y_{j+1}=y_j+\Delta y_j,
\]

其中训练间距 `Δx_i,Δy_j ∈ {1,2,4,8}`，然后从真实 MC 父场的 Cartesian product `(x_i,y_j)` 读取自旋。这样位置坐标和物理采样严格匹配，而且同一 context 内同时出现多种距离。

额外加入一个不训练的错配审计：保留 UniformStride 自旋，却输入随机 gap 坐标。若它与真正的 `LG-Gap-Matched` 同样好，说明收益更可能来自位置噪声正则化，而不是对物理距离的理解。

## 3. 扩展训练尺度

训练：

```text
W_train = {16,24,32,48,64}
uniform_stride_train = {1,2,4,8}
random_gap_train = {1,2,4,8}
```

测试：

```text
W_test = 64:  uniform/random gaps {3,6,10}
W_test = 96:  uniform/random gaps {3,6,10}
W_test = 128: uniform/random gaps {3,6}
```

held-out mixture 至少包括：全 3、全 6、允许时的全 10、`{3,6}` 交替、以及从允许的 held-out gap 集合独立抽样。测试同时包含训练中见过的 gap 值组成的新排列，以区分“新组合泛化”和“新距离泛化”。

父场至少 `L_parent=2048`，并增加到至少 24 条独立链。上述分层测试是硬约束：`W=128, gap=10` 的跨度为 `1270`，超过 `L=2048` 的半宽 `1024`，因此不得运行；若一定要测试该组合，必须另行生成至少 `L_parent=4096` 的父场并重新做 MC 诊断。

## 4. 更强基线

- fixed 2D RoPE；
- Punit/no physical coordinates；
- Randomized Positional Encoding 式随机位置子集；
- Position Interpolation；
- YaRN/LongRoPE-style frequency scaling；
- relative distance bias；
- parameter-matched Dense；
- FNO/GNO 或适合生成任务的 discretization-invariant baseline；
- lattice-aware autoregressive/marginalization baseline。

其中 Randomized Positional Encoding baseline 必须使用与 `LG-Gap-Matched` 相同的位置分布，但自旋仍来自普通连续窗口；这是排除“只随机 position ID 就足够”的关键对照。

## 5. Stage 2B 论文级指标

- NLL/calibration；
- energy、magnetization、Binder cumulant；
- `G(r)` 和 `S(k)`；
- finite-size scaling；
- 临界指数拟合及区间；
- free-energy/partition-function proxy（若模型允许）；
- computational scaling；
- 三 seed effect size 与 CI。

---

# 十五、Stage 2C：泛化与论文边界

只有 Stage 2B 稳定成功后才加入：

1. 非正方形窗口；
2. 超出 separable RandomGap 的随机洞、任意稀疏点集和非 Cartesian geometry；
3. axial/sparse global attention；
4. Potts、XY、percolation 或另一类随机场；
5. PDE/材料晶格等跨任务验证。

这一阶段才有资格支持较广的论文主张：方法学习的是物理坐标与离散尺度，而不是只适配方形 Ising 图像。

---

# 十六、资源与预计时间

## 1. Stage 0

- 实现：1–2 小时；
- GPU：20–40 分钟。

## 2. Stage 2A

- 架构、单测、Probe 改造：3–6 小时；
- 两个新模型训练：预计 15–30 分钟；
- parameter-matched control（若需要）：约 10–15 分钟；
- FP32 Probe 与生成：约 30–60 分钟；
- 总计：代码完成后约 1–2 小时 GPU。

## 3. Stage 2B

- 新数据：约 1–3 小时 CPU，取决于链数；
- 12 个主模型 run：预计 2–4 小时 GPU；
- W=96/128 采样与多 seed Probe：预计 2–6 小时；
- 总计：约 4–10 小时单卡 5090 D。

这些时间必须先通过 200-step 和 W=64 sampling benchmark 更新，禁止按理论 FLOPs 直接承诺。

---

# 十七、必须保存的文件

```text
artifacts/level2/
├── environment.json
├── code_manifest.json
├── sampler_audit/
│   ├── per_sample_metrics.*
│   ├── convergence.json
│   └── figures/
├── runs/
│   ├── dense_punit/
│   ├── dense_t3/
│   ├── lg_punit/
│   └── lg_t3/
├── probes/
│   ├── coordinate_scale_sweep/
│   ├── markov_context_curve/
│   └── gate_attention/
├── generation_w64/
├── final/
│   ├── level2_summary.json
│   ├── LEVEL2_DECISION_ZH.md
│   ├── tables/
│   └── figures/
└── checksums.txt
```

每个 run 必须保存：

- config；
- git commit/dirty diff hash；
- initialization fingerprint；
- history；
- best/last checkpoint；
- performance；
- geometry/corruption trace manifest；
- random seeds；
- CUDA/PyTorch/GPU 信息。

---

# 十八、预期图表

1. Stage 0 sampler convergence；
2. 四模型 ID/OOD learning curves；
3. `NLL(s_coord)` 连续响应曲线；
4. `Delta_MB(W)` context-size scaling；
5. short/medium/expanded `G(r)` 对照；
6. energy、`|m|`、low-frequency distribution；
7. local/global/gated ablation；
8. gate vs t/stride/neighbour visibility；
9. attention mass vs physical distance；
10. 参数、吞吐、显存、采样耗时。

---

# 十九、审稿人式自审

## 1. Contribution

**问题：随机坐标是否只是已有 randomized PE 的领域迁移？**

状态：needs evidence。必须用 paired physical-sampling control、Randomized PE baseline 和跨系统实验说明区别。

**问题：Local–Global 是否只是 Longformer 的常规应用？**

状态：needs evidence。贡献不能写成“首次 local+global attention”，而应是物理尺度生成中的局部保真—全局外推机制、诊断和针对性融合。

## 2. Experimental strength

**问题：只在单 seed Ising 上是否足够？**

状态：不够。Stage 2A 只能筛选；Stage 2B/2C 才能支持论文结论。

**问题：坐标收益是否有 practical significance？**

状态：needs evidence。不能只依赖大样本带来的窄 CI，必须设 `0.002` NLL 或物理指标改善阈值。

## 3. Evaluation completeness

**问题：生成改进是否来自 sampler？**

状态：Stage 0 解决，sampler 必须在 validation 上冻结。

**问题：模型容量是否公平？**

状态：通过 parameter-matched Dense 控制。

## 4. Method soundness

**问题：硬 local radius 是否过度依赖 Ising？**

状态：部分风险。半径必须按物理坐标定义，并在不同局部相互作用系统中验证；论文应承认其局部性假设。

**问题：Gate 是否需要每个任务单独调？**

状态：needs evidence。Gate 初始化和训练规则应固定，并跨 stride/系统复用。

## 5. 当前决策

Stage 2A 的设计足以决定该方向是否值得进入论文级投入，但本身不足以构成顶会实验证据。

---

# 二十、Claim–Evidence Map

| Claim | 所需证据 | 当前状态 |
|---|---|---|
| matched physical coordinates 被模型真实使用 | 连续 scale sweep、Correct/Unit/Wrong paired NLL | 第一级部分支持，第二级需加强 |
| random physical scale 改善长程外推 | 多 seed `G(r)`,`S(k)`、强位置基线 | 第一级单 seed 支持 |
| Local–Global 结构修复局部污染 | Markov curve、energy、short `G(r)` | 尚无证据 |
| 长程收益在修复后保留 | expanded `G(r)` 与 LG-Punit/Dense-T3 对照 | 尚无证据 |
| 方法学习离散尺度而非 Ising 特例 | irregular geometry、跨系统 | 尚无证据 |
| 方法具有合理计算代价 | 参数/FLOPs/吞吐/显存 | 待 Stage 2A benchmark |

---

# 二十一、执行建议

推荐执行顺序：

1. 先实现并冻结 Stage 0 sampler；
2. 实现 LG 模型及单元测试；
3. 128-sample overfit 与 200-step benchmark；
4. 只训练 LG-Punit、LG-T3 单 seed；
5. 用当前 dense checkpoints 完成 2×2 比较；
6. 跑连续坐标曲线、Markov context curve 和 W=64 生成；
7. 按预注册规则停下讨论；
8. 只有 GO/STRONG GO 才创建 Stage 2B 数据与多 seed 队列。

禁止在 Stage 2A 结果未知时提前启动 Stage 2B。
