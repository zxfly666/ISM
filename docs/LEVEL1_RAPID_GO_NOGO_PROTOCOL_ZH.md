# ScaleField-MDLM 第一级快速 Go/No-Go 实验方案

**版本**：v0.1<br>
**日期**：2026-08-18<br>
**目标时长**：12–24 小时，单张 RTX 5090 D<br>
**性质**：探索性筛选，不是论文级确认实验
**后续协议**：[FORMAL_SCALE_AWARE_ISING_PROTOCOL_V03_ZH.md](FORMAL_SCALE_AWARE_ISING_PROTOCOL_V03_ZH.md)

---

## 0. 一句话目标

本实验只回答一个决策问题：

> 多尺寸、真实物理坐标匹配的 stride 训练，是否已经显示出足够明确的坐标利用、context 稳健性或分布外推信号，值得继续投入多 seed 和 `W=128` 正式实验？

第一级不尝试证明论文结论，不运行 Axial-Final，不运行 5 seeds，也不追求完整 `L_parent=2048` 大体积极限审计。

---

# 一、范围与硬性时间预算

## 1. 本阶段运行什么

只训练四个 Dense-Scout 模型，使用同一个初始化 seed：

| 组 | 训练数据 | 输入坐标 | 回答的问题 |
|---|---|---|---|
| T0 | 固定 `W=48,s=1` | 单位真实坐标 | 固定 context 基线 |
| T3 | 多 `W`、真实 stride 自旋 | CorrectCoord | matched-distance 方法是否有信号 |
| Pphase | 连续自旋 | 跳跃坐标 | 收益是否只是 RoPE phase coverage |
| Punit | 真实 stride 自旋 | 单位坐标 | 收益是否只来自自旋数据增强 |

只做三个结果模块：

1. 精确 Markov-blanket contamination；
2. CorrectCoord / UnitCoord / WrongCoord；
3. T0 与 T3 的小规模 `W=64` 生成比较。

## 2. 明确不运行什么

- 不运行 T1/T2 完整 factorial；
- 不运行 Axial 模型；
- 不运行 D3/D4、不规则形状或 anisotropic stride；
- 不加入 context-consistency loss；
- 不比较 YaRN、PI、VisionNTK、distance bias；
- 不运行 `L_parent=2048` 正式数据集；
- 不生成 `W=128`；
- 不使用多 training seeds；
- 不把本阶段结果写成统计显著或论文最终结论。

## 3. 时间上限

| 模块 | 目标时间 | 硬上限 |
|---|---:|---:|
| 环境、代码与单元测试 | 1–3 h | 4 h |
| 小型 MC 数据与 parent audit | 1–3 h | 4 h |
| 200-step benchmark | 0.3–0.7 h | 1 h |
| 四组训练 | 6–12 h | 14 h |
| 两个机制 probe | 0.5–1.5 h | 2 h |
| 小规模生成与分析 | 1.5–3 h | 4 h |

目标总时长为 12–24 小时。若非实现 bug 导致预计超过 24 小时，先停止扩充训练步数并给出中间判断，不自动进入完整版。

---

# 二、服务器与目录

## 4. 已确认资源

```text
GPU              NVIDIA GeForce RTX 5090 D, 32607 MiB
GPU count        1
PyTorch          2.8.0+cu128
CUDA runtime     12.8
compute cap.     12.0
BF16             supported and tested
CPU quota        about 25 cores
memory limit     about 62 GiB
data disk        /root/autodl-tmp, 50 GiB
OS               Ubuntu 22.04.5 LTS
```

## 5. 目录结构

工程目录固定为：

```text
/root/autodl-tmp/ISM
```

结果目录固定为：

```text
artifacts/level1_rapid/
├── env/
├── data_audit/
├── T0/
├── T3/
├── Pphase/
├── Punit/
├── probes/
├── generation/
└── final_summary/
```

磁盘控制：MC 父构型使用 bit-pack；每组只长期保留 `last.pt`、`best_val.pt` 和必要中间日志；所有样本压缩保存。第一级总占用目标 `<10 GiB`。

---

# 三、快速 Monte Carlo 数据

## 6. 物理参数

```text
J = 1
h = 0
beta = 0.5 * log(1 + sqrt(2))
boundary = periodic parent torus
sampler = Wolff cluster
```

核心训练与测试使用 `L_parent=1024`。`L_parent=512` 只生成小型审计集，不与 1024 混合训练。

## 7. `L_parent=1024` 链划分

快速阶段使用 12 条独立链：

| split | 链数 | 每链目标保存构型 |
|---|---:|---:|
| train | 6 | 128 |
| validation | 2 | 128 |
| test-target | 2 | 128 |
| test-control | 2 | 128 |

共 1,536 个父构型。训练时从父构型在线随机抽取多个 crop，但测试统计仍以 chain → parent → crop 为层级。

## 8. 快速热化与保存间隔

MC 时间按 lattice-equivalent Wolff sweep 计算：累计翻转自旋数除以 `L_parent²`。

操作流程：

1. 一半链从随机态、一半从有序态开始；
2. 先运行至少 10 个 equivalent sweeps；
3. 用 energy 与 `|m|` 估计初步 `τ_int`；
4. 热化至少延长到 `max(20,10τ_int)` sweeps，并检查两类初态轨迹汇合；
5. 保存间隔使用 `max(1,2τ_int)` sweeps；
6. 记录初步 split-`R_hat`、ESS 和轨迹图。

该规则用于快速审计。如果链仍明显未平衡，停止模型训练并延长 MC；不能用不可靠数据换取 24 小时时限。

## 9. 小型 parent-size audit

另生成 `L_parent=512` 的 4 条独立链，每链保存 64 个构型。用匹配数量的 1024 构型比较：

- `W=48,s=1` 的 `G(r)`；
- `W=48,s=4` 的真实距离 `G(r)`；
- patch `|m|`；
- matched windowed low-frequency power。

判读：若 512–1024 差异明显超过各自 MC–MC 波动，第一级只使用 1024，并在报告中明确 512 不足；不在本阶段追加 2048。

---

# 四、数据 geometry

## 10. 窗口与 stride

```text
W_train = {16, 24, 32, 48}
s_train = {1, 2, 4}
maximum training span = (48 - 1) * 4 = 188
```

### T0

```text
W = 48
s = 1
spin extraction = contiguous
coordinates = (i, j)
```

### T3

均匀抽取完整 `W×s` cells：

```text
spin extraction = S[x0+s*i, y0+s*j]
coordinates = (s*i, s*j)
```

### Pphase

与 T3 使用相同的 `W×s` 抽样，但：

```text
spin extraction = S[x0+i, y0+j]
coordinates = (s*i, s*j)
```

### Punit

与 T3 使用相同的真实 stride 自旋，但：

```text
coordinates = (i, j)
```

所有组共享父构型 stream、随机平移、旋转/反射、全局 spin flip、diffusion `t` 与 corruption RNG。

---

# 五、模型与训练

## 11. Dense-Scout

| 参数 | 数值 |
|---|---:|
| hidden width | 128 |
| Transformer blocks | 6 |
| heads / head dim | 4 / 32 |
| attention | bidirectional full attention |
| position | external-coordinate decoupled 2D RoPE |
| time conditioning | AdaLN-Zero |
| normalization | pre-norm + QK-Norm |
| MLP ratio | 4, GELU |
| dropout | 0 |

不加入 convolution、distance bias、global token 或 consistency loss。

## 12. 必须先通过的测试

1. PAD 外框 exact-invariance；
2. 全局 coordinate translation invariance；
3. D0 单位坐标 regression；
4. D0/T3/Pphase/Punit 的坐标 overlay；
5. 128 个固定 crop overfit；
6. spin flip augmentation；
7. 16 样本端到端采样；
8. exact Markov posterior 单元测试。

任何测试失败，不启动四组训练。

## 13. MDLM 目标

```text
alpha(t) = 1 - t
t ~ Uniform[0.01, 1]
weight = 1/t
endpoint t=1 batches = 2%
SUBS constraint = enabled
```

主目标：

\[
\mathcal L=
\mathbb E_t\left[
\frac1{|\Omega|}
\sum_i
\frac{\mathbf1[z_i=MASK]}{t}
\operatorname{CE}(p_\theta(\sigma_i\mid z_t,\Omega,t),\sigma_i)
\right].
\]

## 14. Optimizer

```text
optimizer       AdamW fused
lr              3e-4
min_lr          3e-5
betas           (0.9, 0.95)
weight_decay    0.05
warmup          500 updates
schedule        cosine
grad_clip       1.0
precision       BF16
EMA             0.999
```

所有组从同一个保存的初始化权重开始，训练 seed 固定为 `1234`。

## 15. 有效 batch

目标约为 16k–18k sites/update：

| W | micro-batch | accumulation | sites/update |
|---:|---:|---:|---:|
| 16 | 32 | 2 | 16,384 |
| 24 | 14 | 2 | 16,128 |
| 32 | 8 | 2 | 16,384 |
| 48 | 4 | 2 | 18,432 |

若 W48 micro-batch 4 超显存，降为 3 并使用 3 次 accumulation，同时在 loss 中按 site 数归一。

## 16. 训练步数与停止规则

每组：

```text
minimum updates = 5,000
target updates  = 8,000
hard cap        = 10,000
validate every  = 500
checkpoint      = 2,000
```

若连续三个 validation 点覆盖至少 1,500 updates，主 NLL 相对改善 `<1%` 且无训练不稳定，则在 5k–8k 之间提前停止。若仍稳定下降，训练至 10k 上限。四组使用相同停止规则，不允许只延长结果最好的组。

## 17. 200-step benchmark

在正式四组前，分别 benchmark `W=16/32/48`：

- updates/s；
- tokens/s；
- peak VRAM；
- GPU utilization；
- 预计 8k updates 时间。

若四组总训练 ETA 超过 14 小时，将 target updates 统一降至 5,000，而不是删除对照组。

---

# 六、两个低成本高识别力 Probe

## 18. Probe A：精确 Markov-blanket contamination

在 held-out 1024 父构型上选至少 8,192 个中心点。中心被 MASK，四个最近邻强制可见。精确 posterior：

\[
p_i^\star
=\operatorname{sigmoid}\left(2\beta\sum_{j\sim i}\sigma_j\right).
\]

对同一中心构造 Small、Large-PAD、Large-MASK、Large-Visible，评价 `t={0.2,0.5,0.8,0.95}`。

主指标：

\[
\Delta_{MB}(C)
=D_{MB}(C)-D_{MB}(Small),
\]

其中 `D_MB` 是解析 Bernoulli posterior 到模型 posterior 的 KL。另报 CE、Brier 和 logit drift。

预期：Large-PAD 必须数值不变；若 T0 的 Large-MASK/Visible 出现正污染，比较 T3 是否降低。若 T0 本身无污染，不把这一项作为 NO-GO，结论只是污染机制未出现。

## 19. Probe B：坐标使用

使用 held-out stride：

```text
s_interpolation = 3
s_extrapolation = 6
W = 24 or 32, keeping span <= 186
```

每个 stride 至少 2,048 个 paired corruptions，固定相同自旋、MASK pattern 和 `t={0.2,0.5,0.8,0.95}`。

比较：

1. T3 + CorrectCoord；
2. T3 + UnitCoord；
3. T3 + WrongScale；
4. Punit + UnitCoord；
5. Pphase + phase coords。

正确证据要求 CorrectCoord 在真实标签 NLL 上最低，而不仅是 logits 对坐标发生变化。

---

# 七、小规模生成

## 20. 只生成 T0 与 T3

```text
geometry          W=64, s=1, open patch
samples/model     128
sampler           ancestral reveal
steps             64
temperature       1.0
checkpoint        final EMA
guidance          none
MC target/control 512 matched crops each, hierarchical resampling
```

另对每个模型随机选 32 个样本运行 128 steps，检查结论是否被 sampler steps 反转。若反转，标记 sampler 未收敛，不判定模型优劣。

## 21. 快速物理指标

- energy；
- signed `m` 与 `|m|`；
- raw/ensemble-connected `G(r)`；
- short `r=1…8`；
- medium `r=9…16`；
- expanded-context `r=17…32`；
- matched windowed low-frequency power；
- 16 张随机样本阵列。

不在第一级拟合或主张精确 `η`，不把肉眼观感当结论。

---

# 八、Go/No-Go 判定

## 22. 前置有效性

以下任一失败，结论为 `INVALID PILOT`，修复后重跑，而不是 NO-GO：

- MC 未平衡；
- PAD 不变性失败；
- CUDA/NaN/OOM 未解决；
- 四组未使用相同初始化和 site-normalized loss；
- sampler 64/128 steps 给出相反结论。

## 23. ID guard

在 `W=48,s=1` 上，T3 的 validation NLL、short `G(r)` 与 `|m|` 不得出现超出 paired bootstrap/repeatability envelope 的系统恶化。该阶段不写死 5% 或 10%；报告 raw difference 与 uncertainty。

## 24. 三类信号

### A. Coordinate signal

在 `s=3` 和 `s=6` 上：

- T3 CorrectCoord NLL 同时低于 UnitCoord 与 WrongScale；
- T3 CorrectCoord 优于 Punit；
- 方向在主要 `t` bins 中一致。

### B. Pollution signal

仅当 T0 的 `Δ_MB` 高于 bootstrap noise 时评价：T3 的 `Δ_MB` 是否明显更低，且没有损害 Small oracle NLL。

### C. Generation signal

T3 的 `r=17…32` `G(r)` discrepancy 相对 T0 的改善大于 MC target-control noise，并且 `|m|` 与 low-frequency power 不出现相反方向恶化。

## 25. 最终标签

| 标签 | 判定 |
|---|---|
| STRONG GO | ID guard 通过；A成立；B和C均有正信号 |
| GO | ID guard 通过；A成立；B或C至少一项有正信号 |
| CONDITIONAL GO | C改善但A不成立；说明多尺度数据可能有用，但真实坐标作用未证明 |
| NO-GO | ID明显失败，或A/B/C均无可靠方向信号 |
| INVALID PILOT | 数据、实现、训练或 sampler 有效性失败 |

单 seed 结果只能决定是否继续，不能支持“方法稳定有效”的论文表述。

---

# 九、必须保存的结果

## 26. 文件

- 环境与代码 commit；
- MC chain/ESS/`R_hat` manifest；
- 200-step benchmark JSON；
- 四组 train/validation 日志；
- `best_val.pt` 与 `last.pt`；
- Probe A/B 逐样本输出；
- T0/T3 生成样本；
- MC target/control metrics；
- `level1_summary.json`；
- `LEVEL1_DECISION_ZH.md`。

## 27. 图表

1. 512 vs 1024 parent audit；
2. 四组 NLL 学习曲线；
3. per-`t` validation NLL；
4. Markov oracle `Δ_MB`；
5. Correct/Unit/Wrong coordinate NLL；
6. T0/T3/MC `G(r)` 与 ratio；
7. `m`、`|m|` 与 low-frequency power；
8. 随机生成样本；
9. GPU吞吐、显存与实际总耗时。

---

# 十、Claim–evidence map 与自检

| 第一级判断 | 对应证据 | 可否形成论文结论 |
|---|---|---|
| 真实坐标可能有用 | T3>Punit 且 Correct>Wrong | 否，只是单 seed 信号 |
| 长 context 可能污染局部推断 | exact Markov Probe A | 否，需要多 seed 复现 |
| T3 可能改善小规模外推 | `W=64` matched-MC 指标 | 否，需要 `W=128` 与正式样本数 |
| 值得继续 | GO/STRONG GO | 是项目决策，不是科学定论 |

自检：

- 对照组是否在同一代码、初始化和预算下运行；
- 是否把 Pphase 与真实物理 stride 清楚区分；
- 是否用真实标签 NLL 证明正确坐标更优；
- 是否把 MC 和生成样本的层级相关性纳入 bootstrap；
- 是否同时展示负面结果与失败样本；
- 是否避免把单 seed 和 128 个生成样本写成稳定结论。

---

## 执行原则

第一级的价值不在于一次得到漂亮图片，而在于用最少训练回答三个问题：模型是否使用正确坐标、是否减轻可解析的 context 污染、是否在小规模 OOD 上改善完整分布。只有出现可重复方向信号，才进入多 seed 与 `W=128` 正式实验。
