# Scale-aware Ising 实验方案复核意见处理记录

**对应方案**：`FORMAL_SCALE_AWARE_ISING_PROTOCOL_ZH.md` v0.3<br>
**用途**：记录每条外部复核意见是否采纳、为什么采纳，以及如何进入新协议。
**原则**：不因为意见写得完整就机械照搬；优先判断它是否改变目标分布、因果识别、统计单位或可复现性。

---

## 一、总体结论

复核意见的总体方向正确，而且指出了原方案中四个会直接影响论文结论有效性的高风险问题：目标分布混用、父场有限尺寸效应、坐标使用缺乏可识别对照、以及 context-pollution probe 没有使用 Ising 的精确 Markov 条件概率。因此，新版不再把原方案称为已经冻结的预注册协议，而改为“内部确认前协议”；只有完成数据审计和 exploratory pilot、冻结数值 margin 后，才进入 confirmatory 阶段。

处理结果如下：

- 13 条主要意见中，10 条完全采纳；
- 3 条部分采纳，并对其统计或执行细节作进一步修正；
- 没有整条拒绝的主要意见；
- 原方案中的 D3、D4、context-consistency、复杂 RoPE baseline 和 sparse attention 全部后置，不再进入第一轮确认性实验。

---

## 二、13 条主要意见逐条判断

| 编号 | 复核意见 | 决定 | 新版处理 |
|---:|---|---|---|
| 1 | 统一开放父场边缘分布与周期 Gibbs 目标 | **完全采纳** | 核心目标固定为大周期父场在开放有限坐标集合上的边缘分布；周期小系统改为独立 topology-conditioned 扩展 |
| 2 | `L_parent=512` 与大 span 不匹配 | **部分采纳** | 主候选改为 `L_parent=1024`、核心 span `≤256`，但是否足够必须由 `512/1024/2048` parent-size audit 决定，不预先称其为无限体 |
| 3 | 必须证明模型真的使用坐标 | **完全采纳并加强** | 加入 CorrectCoord、UnitCoord、WrongScale/ShuffleCoord，以及同一 corruption 的 inference-time coordinate swap；禁止额外输入 stride 标签 |
| 4 | D1 不是 PoSE 的物理对应 | **完全采纳** | 重命名为 position-phase coverage control；明确它是故意失配的相位覆盖对照，不是合法 Ising 数据分布 |
| 5 | 用精确 Markov blanket oracle 重做污染 probe | **完全采纳** | 拆成 Probe A（无用 context 不应改变局部 posterior）和 Probe B（最近邻未知时远程信息能否真正降低 NLL） |
| 6 | 限定 mask-context consistency 的理论含义 | **完全采纳** | 不进入 T0–T3；仅在确认存在 MASK 污染后加入；同时报告 PAD 精确不变性、denoiser consistency 与 sample-level projective consistency |
| 7 | `1/t` 权重需要冻结 `alpha(t)=1-t` | **完全采纳** | 明确 clean survival schedule、loss 权重和 sampler 必须成套修改；`t=1` endpoint 是额外校准项，不计入主 NELBO |
| 8 | Axial attention 不支持任意坐标集合 | **完全采纳** | Dense-Scout 才支持一般有限集合；Axial-Final 主张限制为矩形或 separable Cartesian grid，任意形状后置 |
| 9 | 开放/不规则几何的 `S(k)` 定义不清 | **完全采纳** | 稠密开放 patch 报 matched-MC windowed spectrum；不规则坐标以 `G(r)`、NLL 和 projective consistency 为主，NUFFT 仅探索性 |
| 10 | “未见尺度”应拆成三类 | **完全采纳** | 分成 token-count OOD、physical-distance OOD 和 compositional OOD；`W=128,s=1` 称 dense long-context compositional extrapolation |
| 11 | 训练预算与 anchor exposure 混淆 | **部分采纳并加强** | 主比较固定 site exposure；另给 FLOP-matched 和 anchor-exposure sensitivity。三者回答不同问题，不能合成一个“公平预算” |
| 12 | 3 个训练 seed 不足以支持稳定推断 | **完全采纳** | Dense confirmatory 使用 5 个 paired seeds；Axial 至少 3 个、理想 5 个；生成样本 bootstrap 不能替代训练 run |
| 13 | PASS 阈值过多且任意 | **部分采纳并修正** | 删除原 30%/15%/50% 等预设；用独立 pilot 冻结最小效应与非劣 margin。噪声扣除同时报告 raw discrepancy，防止简单相减或比值不稳定 |

---

## 三、关键判断的具体理由

### 1. 为什么目标分布必须只保留一种

大周期父场的开放 crop 与同尺寸小 torus 具有不同的边界条件。前者的窗口外自旋被边缘化；后者则把窗口的对边直接连接。两者的磁化、最低波数和有限尺寸修正都不同。如果把它们放在同一主表中，模型误差与目标分布差异无法区分。因此核心论文只研究

\[
p_{\Omega}^{(L_p)}
=\operatorname{Law}\{S_x:x\in\Omega,\,S\sim p_{\beta_c,L_p}^{\rm torus}\}.
\]

周期 benchmark 仍有价值，但需要显式 topology、周期长度和 wrapped distance，属于另一项实验。

### 2. 为什么 `L_p=1024` 也不能直接叫“无限体”

临界点没有有限的相关长度，所以“父场足够大”不是仅凭 `1024>128` 就能成立。新版把 `L_p=1024` 视为候选 reference，并用相同 geometry 比较 `512/1024/2048`。只有 cross-parent difference 落入预先冻结的等价范围，才采用 1024；否则增大父场或缩小 span。

复核意见中提到的“held-out denoising oracle”不作为 parent-size audit 的核心项，因为它会把数据差异与某个已训练模型的偏差混在一起。核心审计只使用由 MC 直接定义的观测量；固定诊断模型的 NLL 可作为补充。

### 3. 为什么真实 stride 仍不能单独证明坐标学习

真实 stride 会同时改变输入自旋的统计。模型可能从“相邻 token 更不相关”推断尺度，而完全忽略坐标。因此新版要求两类证据同时成立：

1. 训练消融：D2+CorrectCoord 优于相同 D2 自旋的 UnitCoord 模型；
2. 推理反事实：在同一模型、同一 D2 样本和同一 corruption 上，正确坐标的 held-out NLL 优于错误坐标。

WrongCoord 不采用任意破坏拓扑作为唯一控制，因为那会制造过于容易识别的异常。主错误条件使用仍然合理但数值错误的 stride scale，例如真实 `s=4` 输入 `s'=2`；随机坐标置乱只作为更强的 sanity check。

### 4. 为什么精确 Markov blanket probe 是最重要的新增项

二维最近邻 Ising 是 Markov random field。当目标点四个最近邻全部可见时，目标 posterior 有精确公式：

\[
P(\sigma_i=+1\mid\sigma_{\partial i})
=\operatorname{sigmoid}\!\left(2\beta J\sum_{j\sim i}\sigma_j\right).
\]

这意味着 PAD、额外 MASK 和额外真实远点都不应改变正确 posterior。它把“模型是否被长 context 污染”变成可直接与解析真值比较的问题，不再依赖模型彼此作为 teacher。

原方案中“Large-Visible 变好说明利用了合法远程信息”的解释只在最近邻没有全部可见时成立。因此新版把它放入单独的 long-range utility probe，避免互相矛盾。

### 5. 为什么 context-consistency 不应一开始加入

若基线没有可测的 MASK 污染，加入该 loss 既没有必要，也会模糊 T0–T3 的主因果比较。即便它降低 shared-site KL，也只说明 denoiser 层面更一致，并不自动证明完整联合生成分布满足 Kolmogorov consistency。因此它被移至阳性结果后的机制增强阶段，并必须与 sample-level restriction test 一起评价。

### 6. 为什么需要三种预算视角

对 dense attention，固定有效 site 数不等于固定 FLOPs，因为成本近似随 token 数平方增长；固定 FLOPs又不等于固定训练样本信息量。多尺寸组还会稀释最大训练窗口的 exposure。因此新版分别报告：

- site-exposure matched：比较相同格点监督量；
- FLOP-matched：比较相同计算成本；
- anchor-exposure matched：排除最大训练尺度样本被稀释的解释。

三种结果均报告，但 site-exposure matched 是主因果表，另外两种是敏感性分析。

### 7. 为什么不能简单依赖 `D_model-MC-D_MC-MC`

若模型样本数与两组 MC 样本数不同，三条经验曲线具有不同的估计方差，简单相减可能过度或不足扣除噪声。新版要求在 bootstrap 内把 model、target MC 和 control MC 匹配到相同有效样本量，主表同时报告 raw discrepancy、MC–MC noise floor 和 signed excess。只有 T0 excess 明显为正时才使用相对改善比值；否则使用绝对 paired difference。

---

## 四、其他细节意见的处理

| 细节意见 | 处理 |
|---|---|
| “内部初稿”与“已正式审核”矛盾 | 已改为 v0.3 内部确认前协议；冻结点写清 |
| 核心 D2 只用 isotropic stride | 采纳；anisotropic 与 irregular 后置 |
| stride 插值/外推集合冲突 | 统一训练 `{1,2,4}`、插值 `3`、外推 `{6,8}` |
| D4 变换需同步坐标与 mask | 采纳，并列为单元测试；D4 不进核心 |
| Wolff 间隔用 lattice-equivalent sweeps | 采纳；热化和保存间隔由 `E`、`|m|` 的 IAT 冻结 |
| 多 crop bootstrap 层级 | 采纳：chain → parent configuration → crop/geometry |
| raw/connected correlation 需固定公式 | 采纳；使用 ensemble-connected 主定义 |
| 不强行套周期 conformal 拟合 | 采纳；matched-MC 是主真值，`eta` 仅 secondary |
| 负控制优先高温唯一相 | 采纳；放在核心阳性后的 Phase 4 |
| 修复 Markdown/文献链接 | 采纳；新版统一使用一手论文链接 |

---

## 五、最终方案相对原方案的实质变化

新版不是简单“增加更多实验”，而是做了六项收缩：

1. 从开放与周期混合目标，收缩为一个明确的开放边缘分布；
2. 从泛称尺度外推，拆成 token、物理距离和组合外推；
3. 从 D0–D4 全部铺开，收缩为 T0–T3、Pphase、Punit 六个核心组；
4. 从相关曲线是否更好，升级为解析 Markov oracle、坐标反事实和 matched-MC 生成三条证据链；
5. 从许多任意 PASS 数字，改为独立 pilot 后冻结 margin；
6. 从一次完成所有架构，改为 Dense 因果确认后再由 Axial 复现大 context。

这使实验更慢于一次性 pilot，但显著降低了得到“曲线变好、却无法说明为什么”的风险。
