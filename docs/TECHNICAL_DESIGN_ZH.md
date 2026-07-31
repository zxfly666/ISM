# ISM：临界二维 Ising 离散扩散技术设计

## 1. 项目目标

本项目研究一个刻意收窄的问题：离散扩散模型能否从二维临界 Ising
构型中学习局域相互作用与长程临界涨落，并把在较小晶格上学到的统计规律
迁移到更大的周期晶格。

第一阶段只训练零外场、临界逆温

\[
\beta_c=\frac{1}{2}\log(1+\sqrt{2})
\]

下的方格 Ising 模型。构型为 \(x\in\{-1,+1\}^{L\times L}\)，采用周期边界，
哈密顿量为

\[
H(x)=-\sum_{i,j}\left(x_{i,j}x_{i+1,j}+x_{i,j}x_{i,j+1}\right).
\]

训练成功不能仅由“图片像不像”判断。生成分布至少需要同时匹配能量、磁化、
Binder 累积量、二阶相关长度、实空间关联函数和结构因子。

## 2. 与 `block_diffusion.py` 的关系

实现借鉴
[`nano_world_model/block_diffusion.py`](https://github.com/suning-git/nanoinfra/blob/main/exemplars/nano_world_model/block_diffusion.py)
的核心思想，而不是直接复制语言模型结构：

1. 增加吸收态 `[MASK]`；
2. 连续采样噪声强度 \(t\)；
3. 每个位置以概率 \(t\) 被替换成 `[MASK]`；
4. 网络从带噪输入直接预测干净 token；
5. 损失只在被遮挡位置计算，并使用 \(1/t\) 重要性权重；
6. 验证时固定 \(t\) 网格与随机掩码，降低估计噪声。

未采用语言模型专用的因果注意力、clean/noisy 双流、超大词表输出头和编译
kernel。Ising 去噪是非因果二维条件建模，输出类别仅为两个自旋方向。

## 3. 离散扩散过程

编码约定为 `0=-1`、`1=+1`、`2=MASK`。正向过程为吸收式遮挡：

\[
q(x_t^i\mid x_0^i) =
\begin{cases}
1-t,&x_t^i=x_0^i,\\
t,&x_t^i=\mathrm{MASK}.
\end{cases}
\]

对 batch 中每个样本独立采样 \(t\sim U[t_{\min},1]\)。训练估计量为

\[
\mathcal L=
\frac{1}{BL^2}\sum_{b,i}
\frac{\mathbf 1[x_{t,b}^i=\mathrm{MASK}]}{t_b}
\operatorname{CE}\left(x_{0,b}^i,p_\theta(\cdot\mid x_{t,b},t_b)\right).
\]

这对应连续噪声级别上的 masked denoising 估计。额外以小概率直接取 \(t=1\)，
训练无条件生成所需的全遮挡端点。该原子概率是端点正则项，不应与连续 NELBO
混为一谈。

### 风险

- \(t=1\) 时所有位置平移等价，微小类别偏差可能被采样器放大；
- 模型若在低 \(t\) 或全遮挡端点校准不足，严格祖先采样可能接近独立随机自旋；
- irreversible confidence commit 会把早期错误永久冻结；
- 训练 NELBO 下降不保证长程物理量正确。

## 4. 去噪网络

模型是尺寸可变的二维 axial Transformer：

- token embedding 表示两个自旋与 MASK；
- 3×3 circular convolution 注入周期局域邻域；
- 每个 block 依次做行注意力、列注意力和 MLP；
- 时间嵌入通过 adaptive LayerNorm 进入每个子层；
- 注意力偏置是周期距离的函数
  \(-a_h\log(1+d_{\mathrm{periodic}})\)；
- 不使用固定大小的位置 embedding，因此同一组参数可接受不同 \(L\)。

Axial attention 把完整二维注意力的高阶代价分解为行列两次注意力。模型仍能
在一个 block 内经过行列路径实现全局通信，同时保留周期平移归纳偏置。

## 5. 采样器

代码同时保留四种采样器，必须在同一 checkpoint 上比较：

### 5.1 `confidence`

从全 MASK 开始，每一步：

1. 预测每个 MASK 位置的二分类分布；
2. 从分布抽样候选 token；
3. 按抽中 token 的置信度选择本轮提交的位置；
4. 通过 cosine schedule 逐步减少 MASK 数量。

在置信度上加入随 mask fraction 衰减的 Gumbel 噪声，避免全 MASK 时按扁平索引
机械打破平局。

### 5.2 `confidence_corrector`

在 confidence 解码的前几步，随机重新遮住少量已提交位置，再由下一轮置信度
筛选重新提交。它允许修正早期错误，但后期停止回溯以保证收敛。

confidence corrector 的诊断配置：

```text
steps=12
corrector_steps=4
corrector_mask_ratio=0.10
selection_noise=0.05
temperature=1.0
```

### 5.3 `ancestral`（当前默认）

若 \(s<t\)，已在 \(t\) 时被遮挡的位置在反向一步中显露的精确概率是
\(1-s/t\)。该采样器按此概率随机选择显露位置，并从网络预测的
\(p_\theta(x_0\mid x_t,t)\) 抽样。

它在反向过程上更一致，但对网络校准要求更高。短训练模型上表现差时，不能
据此断言扩散公式错误，更可能说明网络尚不能支撑无置信度筛选的后验抽样。
2000-step pilot 的 256-sample × 3-seed 对照显示，`steps=24,
temperature=0.9` 明显优于 confidence，因此升级为当前默认。

### 5.4 `ancestral_corrector`

祖先采样结束后随机遮挡一小块并重新去噪，作为 blocked conditional corrector。
它是诊断工具，不应未经对照直接作为默认。

## 6. 数据与独立性

训练、验证、测试必须按 Monte Carlo chain 划分，不能先混合构型再随机切分。
否则相邻采样之间的自相关会造成数据泄漏。

正式数据流程：

1. 使用多个独立 Wolff chain；
2. 每条 chain 独立 burn-in；
3. 估计能量和 \(|m|\) 的 integrated autocorrelation time；
4. 保存间隔至少取若干倍自相关时间；
5. train/val/test 使用互不重叠的 chain。

D4 旋转/反射与零外场全局 spin flip 是精确对称性，可在线增强。它们不能替代
独立 Monte Carlo 样本。

## 7. 评价指标

每次实验保存完整样本和以下统计量：

- 能量密度均值及分布；
- \(m\) 与 \(|m|\) 分布；
- Binder cumulant
  \(U_4=1-\langle m^4\rangle/(3\langle m^2\rangle^2)\)；
- connected correlation \(G(r)\)；
- structure factor \(S(k)\)；
- 二阶相关长度比 \(\xi_2/L\)；
- 多采样 seed 和多训练 seed 的均值、方差与最差值。

不能把不同量纲的绝对误差简单相加作为最终科学指标；pilot 中的总误差只用于
快速排序，正式结论必须给出每个 observable 的 bootstrap 置信区间。

## 8. 1.5 小时 pilot 协议

### 阶段 A：管线与吞吐（10–15 分钟）

- 运行单元测试；
- 用小模型跑 50–100 step；
- 测量 data、forward、backward、optimizer 的总吞吐；
- 比较 eager、fused AdamW、数据预载入、`torch.compile`；
- compile 的首次编译时间必须计入总预算。

### 阶段 B：是否学到局域规律（25–35 分钟）

- 在 \(L=16\) 或 \(L=32\) 上训练小模型；
- 保存 step 0/中间/最终 checkpoint；
- 检查 validation NELBO 与能量误差是否同步改善；
- 若 NELBO 降但能量不改善，优先检查采样器与数据质量。

### 阶段 C：采样稳定性（15–20 分钟）

- 同 checkpoint 对比四种 sampler；
- 每种至少 3 个 sampling seed；
- 每 seed 至少 256 张，smoke 阶段可降为 32–64；
- 不根据单个 seed 的最佳图片选择参数。

### 阶段 D：最小尺度外推（20–30 分钟）

- 训练尺寸上先通过；
- 再测试 \(2L\)，重点看训练未覆盖的低波数区域；
- 若同尺寸尚未匹配，不解释尺度外推结果。

## 9. 单卡与双卡策略

当前服务器是一张 RTX 5090 32GB。单卡优先增大 batch 或模型宽度，避免 DDP
通信开销。若以后有两张卡：

- pilot 更推荐两卡各跑一个独立训练 seed；
- 只有单模型显存或时限成为瓶颈时再使用 DDP；
- DDP 必须使用 DistributedSampler 或独立随机 batch，并验证全局 batch 改变后
  学习率和 EMA 衰减是否仍合理；
- 多 sampling seed 不能替代多 training seed：前者测采样随机性，后者测优化与
  数据顺序稳定性。

实测完整 12.67M 参数、\(L=32\)、batch 32 模型在编译后约 29.6 step/s，
峰值 reserved 显存约 4.03 GiB。60k step 核心训练约 34 分钟，因此当前
1.5 小时 pilot 没有必要为了单模型吞吐引入双卡 DDP；若有双卡，优先并行两个
training seed。

## 10. 工程优化路线

按风险从低到高：

1. 向量化 D4 与 spin-flip 增强，消除逐样本 GPU 同步；
2. 可选 fused AdamW；
3. 小数据集预载入 GPU，消除每 step H2D 复制；
4. 可选 `torch.compile`，以实测净收益决定；
5. BF16 autocast 与高精度 matmul；
6. 若采样仍高度敏感，重新设计正向过程为 binary uniform D3PM，使终点是随机
   ±1 并允许可逆 bit transition。

物理 MCMC corrector 可以快速改善样本，但它利用了已知哈密顿量，不能作为
“模型学到了规律”的主要证据。若使用，只能单独标记为 hybrid baseline。

## 11. 验收标准

pilot 的最低成功标准：

1. validation NELBO 明显低于随机二分类基线 \(\log 2\)；
2. 生成能量相对 MC 的误差随训练下降；
3. 三个 sampling seed 不出现明显单态坍缩；
4. 默认采样器的改善在均值或最差 seed 上可复现；
5. 所有配置、seed、checkpoint、样本和指标可追踪。

达到以上标准只能说明“值得扩大实验”，不能证明已经学习完整临界标度规律。
