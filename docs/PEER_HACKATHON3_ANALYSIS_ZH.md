# Hackathon-3 负面结果代码分析

分析对象：
[`zdacongming-glitch/Hackathon-3`](https://github.com/zdacongming-glitch/Hackathon-3)
主分支提交 `95add7c`。

## 1. 结论摘要

该仓库的负面结果更像是“一个严谨评估框架正确拒绝了一个训练量远远不足、
采样配置不匹配的弱基线”，而不是数据生成或长程相关指标写错。

最可能的原因按优先级排序：

1. 默认配置没有启用文档描述的 NELBO/祖先扩散路径；
2. 训练严重不足，尤其 \(W=64\) 只见到约 400 个 batch-size-1 crop；
3. MaskGIT 只有 8/16 步，且 Gumbel ranking noise 为 1.0，早期揭示近似随机；
4. 模型仅 204,672 参数，没有局域相互作用偏置、EMA 或足够训练；
5. 训练长程模式的有效独立父组态数远小于表面 crop 数；
6. 全 MASK 状态的输出严格空间等价，生成高度依赖首轮随机揭示；
7. 64/128 个生成样本足够做工程 gate，但不足以精确估计临界长程尾部。

因此不能根据该实验断言“离散扩散学不到 Ising 长程相关”。它说明的是：
当前 committed config 下的小模型 + 短训练 + MaskGIT 没有通过长程 gate。

## 2. 实际运行配置与文档设计发生漂移

`STAGE1_PROTOCOL.md` 描述了：

- fixed-\(N\)、\(1/t\) 加权 NELBO；
- `sampled_t` conditioning；
- 未修改的 Bernoulli corruption；
- posterior-consistent absorbing reverse sampler。

但是 `stage1_pilot.json` 和 `stage1_local.json` 没有设置以下键：

```json
{
  "training": {
    "loss_objective": "nelbo",
    "mask_conditioning": "sampled_t",
    "ensure_at_least_one": false
  },
  "sampling": {
    "method": "absorbing_diffusion"
  }
}
```

代码默认值实际为：

- `loss_objective="masked_mean"`；
- `mask_conditioning="realized_mask_ratio"`；
- `ensure_at_least_one=True`；
- `sampling.method="maskgit"`。

并且 committed config 仍设置：

- `force_all_mask_probability=0.1`；
- pilot 只训练 120 updates；
- local 只训练 600 updates；
- pilot/local sampling steps 为 8/16；
- `confidence_noise_scale=1.0`。

所以“协议中存在祖先扩散组件”不等于“负面实验已经使用它”。

## 3. 训练量为何不足

模型只有 204,672 个参数，但小参数量并不意味着几百步足够学会临界分布。
训练尺寸为 \(32,48,64\)，gradient accumulation 为 2，尺寸按 microbatch
轮转。600 updates 共产生 1200 个 microbatches，每个尺寸约 400 个：

| 宽度 | batch size | 约 microbatches | 约 crop 数 |
|---:|---:|---:|---:|
| 32 | 4 | 400 | 1600 |
| 48 | 2 | 400 | 800 |
| 64 | 1 | 400 | 400 |

等 token budget 解决了显存和计算平衡，但没有解决独立样本数平衡。最需要学习
长程结构的 \(W=64\) 反而只见到约 400 个 crop。

作为对照，我们的 1.6M 参数 L16 模型在约 1500 updates 后才得到可用的祖先
采样器；80-step 模型的祖先采样接近随机。不能把 120/600 updates 视作充分
训练。

## 4. 数据表面很多，长程有效样本并不多

仓库的数据划分方法本身是正确的：

- 先按完整 MC chain 分 train/val/test；
- 再从父组态 crop；
- validation/test 每个父组态只取一个固定 crop；
- train 只使用 D4 和全局 spin flip 精确对称性。

但训练来源只有 `stage0_local` 的 3 条链、每链 256 个父组态。Stage 0 文档
自己报告临界 \(L=512\) pilot 的最慢 observable 最低 ESS 约 19.4，随后追加
实验才把 ESS 提升到 126–294，而追加 run 被保留给 test。

随机裁更多窗口能增加局部纹理多样性，却不能按相同比例增加全局磁化模式和
长波涨落的独立信息。对长程学习而言，独立 chain/time block/parent 的数量
比 crop 总数更关键。

## 5. 模型形式上有全局感受野，但归纳偏置偏弱

模型采用逐 spin full self-attention，因此第一层就有全局感受野；负面结果
不能简单归因于“receptive field 不够大”。

真正的问题是：

- 仅 4 层、宽度 64、4 heads；
- 没有 3×3 邻域卷积、nearest-neighbor relative bias 或显式局域能量特征；
- 2D RoPE 给出相对几何，但不会主动强调最近邻；
- 没有 EMA，最终或 best checkpoint 都是高噪声小 batch 的原始权重；
- full attention 为 \(O(W^4)\)，消耗大量计算，却未把 Ising 的局域哈密顿量
  先验编码进去。

这使模型需要先从很少的数据中学会最近邻规律，再用同一小网络学习长波模式，
样本效率较低。

该模型面向“开放窗口 marginal”，所以不能不加区分地照搬周期 circular
convolution。更合适的改进是开放边界局域卷积、局域 relative bias，或者局域
层与全局 axial/full attention 混合。

## 6. 全 MASK 对称性和首轮揭示敏感

该实现只在 Q/K 上施加 2D RoPE，不向 value 或 residual stream 加绝对位置。
全 MASK 时所有 token embedding、condition 和 value 完全相同。即使 attention
权重随相对位置变化，对相同 value 的加权和仍相同，因此所有位置 logits
相同。

直接前向检查得到：

```text
parameters = 204672
all_mask_spatial_max_delta = 6.7e-8
one_visible_spatial_std = 0.0151
```

这符合无外场平移对称性，本身不是 bug；但它意味着空间结构完全从第一批随机
揭示开始形成。采样器因此非常关键。

## 7. 当前 MaskGIT 设置为何不利于长程结构

MaskGIT 的 early ranking score 为：

\[
\log p + \text{noise\_scale}(1-\text{progress})\,G,\quad G\sim\mathrm{Gumbel}.
\]

committed config 使用 `confidence_noise_scale=1.0`。在 8-step pilot 的第一步，
噪声尺度约为 0.875，通常远大于模型位置间的 log-confidence 差异。早期提交
顺序因此接近随机。

同时：

- pilot 只有 8 次、local 只有 16 次网络调用；
- 已提交 token 不会修改；
- 首批随机自旋会成为后续全局模式的种子；
- confidence top-k 还可能产生过度有序的低温偏置。

我们自己的实验也观察到：弱模型上 confidence 看似优于祖先采样，训练充分后
却坍缩到 \(E\approx-1.98, |m|\approx0.99\)。换成 24-step 祖先采样后才恢复
临界统计。因此同学仓库的负结果很可能包含明显 sampler contribution。

## 8. 评估系统是否可信

总体可信，而且是仓库最值得学习的部分。

优点：

- open crop 使用非 wrapping 能量与相关函数；
- 同时报 raw、connected、axis、diagonal correlation；
- signed \(m\) 能发现只生成单一自旋符号的 mode collapse；
- patch-shuffle 保留 tile 内纹理，破坏跨 tile 排列，是直接针对长程结构的
  负控制；
- 使用独立 MC-A/MC-B 给出统计噪声基线；
- gate 不会把“代码跑完”写成“学到临界物理”；
- Stage 0 有 exact \(4\times4\) enumeration、IAT、ESS、split-\(\hat R\)、
  Gibbs calibration 和 crop parent-size 检查。

限制：

- pilot/local 每尺寸仅 64/128 个模型样本，临界尾部方差较大；
- correlation relative L2 对每个径向 bin 等权，没有按 pair count 或
  bootstrap uncertainty 加权；
- 仓库不提交 `outputs/`，因此无法从 GitHub 核对实际失败数值、学习曲线、
  checkpoint 或生成样本；
- 当前 gate 是工程阈值，不是训练 seed/chain-block bootstrap 的科学检验。

这些限制可能影响“失败有多严重”，但不足以把明显长程失败解释成指标 bug。

## 9. 值得移植到 ISM 的组件

### 高优先级

1. `ParentSource/ParentPool` 的 chain-first provenance 与 split 审计；
2. open-window correlation，供未来 crop-based 任务使用；
3. raw/connected/axis/diagonal 四套关联诊断；
4. patch-shuffle negative control；
5. MC-vs-MC noise baseline；
6. Stage 0 的 IAT/ESS/\(\hat R\)/exact enumeration 审计；
7. 自动 gate 与失败报告。

### 中优先级

1. 2D RoPE full-attention 可作为小尺寸 control architecture；
2. 多尺寸 token-budget batching；
3. sampled-\(t\) 与 realized-mask-ratio 同时记录；
4. 固定 validation crops 和固定 mask grid；
5. source manifest、环境记录和可重建实验产物。

### 不应直接照搬

1. 8/16-step、noise scale 1.0 的 MaskGIT 默认值；
2. 只有 120/600 updates 的训练预算；
3. 把等 token budget 当成等独立样本量；
4. 对周期 lattice 直接使用开放 crop 架构或指标；
5. 对开放 crop 直接使用我们当前的 periodic circular bias。

## 10. 最小诊断实验

不要一次同时换数据、模型、目标和采样器。建议按以下顺序：

### A. 证明是不是单纯欠训练

保持模型和数据不变，把训练延长到 5000 updates；每 500 step 固定 checkpoint，
比较 validation NELBO、能量和 \(G(r)\)。

### B. 显式启用代码中已经存在的 aligned 路径

```json
{
  "training": {
    "max_updates": 5000,
    "loss_objective": "nelbo",
    "mask_conditioning": "sampled_t",
    "ensure_at_least_one": false,
    "force_all_mask_probability": 0.0,
    "mask_ratio_min": 0.02,
    "mask_ratio_max": 1.0
  },
  "sampling": {
    "method": "absorbing_diffusion",
    "steps": 24,
    "temperature": 0.9,
    "time_schedule": "cosine"
  }
}
```

不建议协议草案中的 \(t_{\min}=0.2\)：24-step cosine reverse 会查询低于 0.2
的噪声区间，容易造成 conditioning OOD。\(t_{\min}=0.02\) 更连续。

### C. 隔离 sampler 影响

同一 checkpoint 比较：

- MaskGIT 16/24/48 steps；
- noise scale 0.05/0.2/1.0；
- absorbing 24/48 steps；
- temperature 0.9/1.0；
- 每组至少 3 sampling seeds、256 samples。

### D. 提高模型样本效率

- 宽度 128，6–8 层；
- 增加开放边界局域卷积或 nearest-neighbor attention bias；
- 加 EMA；
- 使用 BF16、fused AdamW、预载入和 `torch.compile` 提高更新数；
- full attention 只作为 \(W\le64\) control，较大窗口改用 axial/local-global
  hybrid。

### E. 增加真正独立的长程训练信息

- 重新生成更多独立 training chains；
- 用 chain/time-block ESS 决定父组态数；
- 不把当前 test extension 回流训练；
- 为正式 test 再生成独立 seeds/chains；
- 每个训练 seed 报告 MC-noise-normalized correlation error。

## 11. 最终判断

该项目最有价值的不是当前生成模型，而是它的 Monte Carlo、数据 provenance 和
物理 gate。负面结果是可信的工程失败，但尚不是对离散扩散方法的有效否定。

只做一项修改时，应先把 sampler 改为代码里已经实现的
`absorbing_diffusion`，显式使用 `sampled_t` conditioning，并把训练提高到
至少 5000 updates。若这样仍无法超过 patch-shuffle 和 MC noise baseline，
再判断模型归纳偏置或训练数据的独立长程信息不足。

