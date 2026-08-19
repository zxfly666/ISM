# 第一级 Scale-Aware Ising 快速实验结果报告

## 1. 结论摘要

本实验严格区分了四种训练设置：固定尺度基线 T0、物理 stride 与坐标同时匹配的 T3、只改变坐标相位的 Pphase，以及只改变物理采样 stride 但保持单位坐标的 Punit。四组使用相同网络、相同初始化、相同训练步数和同一组 Monte Carlo 父场数据。

按预注册 Go/No-Go 规则，最终标签为：**NO-GO for the unchanged design**。这里的含义不是“尺度随机化完全无效”，而是：当前 dense-attention 实现虽然学会了物理坐标，并显著改善了中长程相关，但没有同时守住短程物理，还加重了精确 Markov 邻域 Probe 中的 context 污染。因此，不建议直接把当前 T3 原封不动扩大到多 seed、W=128 和更大模型。

实验给出了三个同时成立的重要发现：

1. **坐标信号真实存在。** 在完全相同的自旋和 MASK 下，T3 使用正确坐标时的 NLL 稳定优于 UnitCoord 和 WrongScale，方向在两个 held-out stride 和全部四个噪声档均一致。
2. **中长程生成显著改善。** T3 的 connected `G(r)` NRMSE 在 `r=17...32` 从 T0 的 `0.1643` 降到 `0.0466`。
3. **改善伴随局部代价。** T3 的短程 NRMSE 从 T0 的 `0.0354` 升到 `0.0541`，开放边界能量误差也从 `0.0265` 升到 `0.0493`；精确 Markov Probe 中，T3 的 Large-MASK 污染比 T0 更强。

所以更准确的科学表述是：**训练时随机化真实物理坐标确实改变并改善了尺度外推行为，但单一 dense attention 没有自动学会“长程扩展时保护局部规律”，出现了短程—长程权衡。**

---

## 2. 实验有效性

### 2.1 数据

- 临界二维 Ising，`L_parent=1024`；
- 12 条独立 Wolff 链；
- 每链 128 个父构型，共 1,536 个父构型；
- train/val/test-target/test-control 按链拆分，避免同链泄漏；
- 数据以 bit-packed 形式保存，总大小约 124 MiB。

MC 诊断：

| 指标 | 数值 |
|---|---:|
| energy split-Rhat | 1.0263 |
| abs-magnetization split-Rhat | 1.0139 |
| mean energy density | -1.4150 |
| mean abs magnetization | 0.4278 |

这些数值足以支持单 seed 快速判断，但 energy Rhat 仍高于正式论文实验常用的更严格目标。后续正式实验应增加链数或链长度。

### 2.2 实现检查

- 18/18 单元测试通过；
- 四组初始化 SHA-256 完全相同；
- 四组均完成 8,000 updates；
- 无 NaN、OOM 或进程异常；
- FP32 PAD 最大绝对 logit 漂移为 `8.23e-6`；
- 64/128-step 下 T3 相对 T0 的长程排名没有反转。

初次 BF16 Probe 曾得到 `0.03125` 的 PAD logit 漂移。进一步检查发现，这是不同序列长度触发的低精度 attention 数值路径差异，而非 PAD token 泄漏。机制 Probe 随后全部以 FP32 重跑；BF16 原结果被保留作审计副本，不用于最终判定。

---

## 3. 模型与训练

### 3.1 模型

- coordinate-aware dense Transformer denoiser；
- external 2D RoPE；
- `d_model=128`；
- 4 heads；
- 6 blocks；
- 约 1.713M 参数；
- absorbing discrete diffusion；
- EMA checkpoint；
- BF16 训练、FP32 机制评估。

### 3.2 四组设置

| 组别 | 自旋数据 | 外部坐标 | 作用 |
|---|---|---|---|
| T0 | `W=48,s=1` | unit | 固定尺度基线 |
| T3 | 多 W，真实 `s=1,2,4` | matched `s` | matched-distance 核心方案 |
| Pphase | 连续自旋 `s_spin=1` | `s_coord=1,2,4` | 只改变 RoPE phase |
| Punit | 真实 `s_spin=1,2,4` | unit | 只改变数据尺度 |

### 3.3 训练性能

| 组别 | 时间 | updates/s | 峰值显存 |
|---|---:|---:|---:|
| T0 | 373.5 s | 21.42 | 0.56 GiB |
| T3 | 381.0 s | 21.00 | 0.55 GiB |
| Pphase | 378.3 s | 21.15 | 0.55 GiB |
| Punit | 378.8 s | 21.12 左右 | 0.55 GiB |

四组总训练约 25.6 分钟。尺度随机化本身没有造成明显额外训练开销。

### 3.4 最终验证 NLL

| 组别 | ID `W=48,s=1` | held-out `W=24,s=3` |
|---|---:|---:|
| T0 | 0.333514 | 0.586007 |
| T3 | 0.335042 | 0.450100 |
| Pphase | 0.334935 | 0.473426 |
| Punit | 0.336619 | 0.451478 |

T3 的 ID NLL 仅比 T0 高 `0.001527`，说明全局验证 NLL 几乎保持；held-out stride 上 T3 和 Punit 都大幅优于 T0。这已经提示：**大部分尺度外收益可能来自见过真实 stride 数据，坐标的额外贡献需要配对 Probe 才能分离。**

---

## 4. 坐标因果 Probe

对 held-out `s=3` 和 `s=6`，固定同一 clean spin、同一 MASK pattern 和同一 t，仅改变传给模型的坐标。下表报告：

`NLL(CorrectCoord) - NLL(comparator)`。

负数表示正确坐标更好；区间为 5,000 次 paired bootstrap 的 95% CI。

| stride | 对照 | 配对 NLL 差 | 95% CI |
|---:|---|---:|---:|
| 3 | T3 UnitCoord | -0.02241 | [-0.02288, -0.02193] |
| 3 | T3 WrongScale | -0.00879 | [-0.00908, -0.00849] |
| 3 | Punit UnitCoord | -0.00096 | [-0.00109, -0.00083] |
| 6 | T3 UnitCoord | -0.04471 | [-0.04535, -0.04406] |
| 6 | T3 WrongScale | -0.00220 | [-0.00236, -0.00203] |
| 6 | Punit UnitCoord | -0.00266 | [-0.00285, -0.00246] |

六项比较均为负，且每项在 `t={0.2,0.5,0.8,0.95}` 四个档中方向一致。

解释：

- 模型不是完全忽略外部坐标；CorrectCoord 对预测有真实帮助。
- CorrectCoord 相对同一个 T3 模型的 UnitCoord 改善很大，尤其在 `s=6`。
- 但 T3 相对 Punit 的增益只有约 `0.001...0.003` NLL，说明**多尺度自旋数据贡献了主要收益，matched coordinate 提供的是较小但稳定的额外收益。**

---

## 5. 精确 Markov 邻域 Probe

中心自旋被 MASK，四个最近邻强制可见，并使用二维 Ising 的解析条件概率作为 oracle：

`p(s_i=+1 | neighbours) = sigmoid(2 beta sum_j s_j)`。

定义 `Delta_MB = KL(context) - KL(Small)`。

| Context | T0 Delta_MB | T3 Delta_MB | T3 - T0 |
|---|---:|---:|---:|
| Large-MASK | 0.06383 | 0.11831 | +0.05447 |
| Large-Visible | 0.00230 | 0.01903 | +0.01674 |

所有差异的 bootstrap 区间均远离 0。

含义：

- T0 确实存在 context pollution；
- T3 没有降低污染，反而显著加重；
- T3 更积极地使用大范围 token 与坐标，这帮助了长程建模，但在局部条件概率本应只依赖四个邻居时，远处 context 对局部 posterior 产生了更强干扰。

这正是当前方案不能直接进入更大规模正式实验的主要原因之一。

---

## 6. W=64 生成结果

每个模型生成 128 个样本；MC target/control 各 512 个开放窗口。主采样器为 64 steps，另用 32 个样本检查 128 steps。

### 6.1 connected G(r) NRMSE

| 区间 | T0 | T3 | MC target-control noise |
|---|---:|---:|---:|
| short `r=1...8` | 0.03541 | 0.05412 | 0.00051 |
| medium `r=9...16` | 0.09924 | 0.05347 | 0.00182 |
| expanded `r=17...32` | 0.16426 | 0.04657 | 0.01204 |

T3 在中程和长程显著改善，expanded-context 误差约下降 72%；但 short-range 误差增加约 53%，远高于 MC target-control 噪声。

### 6.2 标量观测量

| 指标 | MC target | T0 | T3 |
|---|---:|---:|---:|
| open energy density | -0.70687 | -0.68042 | -0.65760 |
| abs magnetization | 0.52085 | 0.56266 | 0.49521 |
| low-frequency power | 0.13272 | 0.11214 | 0.11484 |

T3 的 `abs(m)` 和低频功率比 T0 更接近 MC，但能量明显更差。能量主要反映最近邻排列，因此它与 short-range `G(r)` 的退化相互印证：T3 的样本具有较好的大尺度组织，却有过多局部粗糙/不对齐的自旋。

### 6.3 sampler 敏感性

- 64 steps：expanded NRMSE，T0 `0.1643`，T3 `0.0466`；
- 128 steps：expanded NRMSE，T0 `0.3984`，T3 `0.0526`。

T3 相对 T0 的排名没有反转，但 T0 对 steps 很敏感。该结果说明 T3 的长程优势不是由 64-step 偶然采样造成的，同时也提示后续实验需要更系统的 sampler convergence 曲线，而不能只比较单个 step 数。

---

## 7. Go/No-Go 判定

### 7.1 通过的项目

- MC、初始化、训练步数、PAD 与 sampler 排名有效性全部通过；
- Coordinate signal 通过；
- Generation long-range signal 通过。

### 7.2 未通过的项目

- ID guard 未通过：T3 的 short-range `G(r)` 明显比 T0 更差；
- Pollution signal 未通过：T3 的 Markov pollution 不仅没有降低，而且显著升高；
- open energy 也显示局部物理退化。

### 7.3 严格标签

**NO-GO for unchanged T3.**

这个标签反对的是“保持当前 dense 架构不变，直接扩大训练规模”，不是反对研究问题本身。实验反而证明了问题真实存在：尺度坐标能够改善长程，却会污染局部推断，当前网络无法自动兼顾二者。

---

## 8. 建议的下一步

不建议马上做当前 T3 的多 seed 大规模重复。更高价值的下一步是设计一个最小架构修正，显式保护局部路径：

1. 保留 T3 的 matched physical coordinates；
2. 每个 block 增加严格局部 attention 或局部卷积分支；
3. 全局 dense attention 作为残差分支，并采用可学习 gate；
4. 训练时加入 Markov-oracle/local-consistency auxiliary loss；
5. 先用单 seed 比较 `T3-dense` 与 `T3-local-global`；
6. 只有在 short NRMSE、energy 和 Markov pollution 被修复，同时保留 expanded `G(r)` 改善后，再进入多 seed、W=128 和 sparse/axial attention。

这会把下一阶段的问题从“坐标随机化是否有效”推进为更精确的问题：**怎样让模型利用尺度坐标学习长程关联，同时不让长程 context 污染局部 Markov 规律。** 这一问题比单纯调 RoPE scale 更接近可发表的方法贡献。

---

## 9. 结果文件

- `artifacts/level1_rapid/formal_results/final/level1_summary.json`
- `artifacts/level1_rapid/formal_results/final/figures/training_curves.png`
- `artifacts/level1_rapid/formal_results/final/figures/coordinate_probe.png`
- `artifacts/level1_rapid/formal_results/final/figures/markov_probe.png`
- `artifacts/level1_rapid/formal_results/generation/correlation.png`
- `artifacts/level1_rapid/formal_results/generation/sample_grid.png`
- `artifacts/level1_rapid/level1_rapid_results_no_checkpoints.tar.gz`

完整逐样本 CSV、BF16 审计副本、日志和配置位于压缩包中。模型 checkpoint 仍保存在远端服务器，未放入该结果包。
