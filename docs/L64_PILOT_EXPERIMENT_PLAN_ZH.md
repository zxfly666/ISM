# L=64 临界 Ising 离散扩散 Pilot 实验方案

版本：2026-07-31

状态：执行前冻结方案

主配置：`configs/pilot_l64.json`

## 1. 结论与目标

本轮直接生成完整的周期边界 `64 x 64` Ising 构型，不从 `L=256` 裁剪，也不把小图拼接成大图。训练目标是检验吸收态 discrete diffusion 是否不仅能复现局部能量，还能学到临界点跨尺度的长程关联。

主实验使用 12.67M 参数的 axial denoiser、batch size 16、60,000 个优化 step、BF16 和 EMA。单张 RTX 5090 的实测稳态速度约为 19.6 step/s，60k step 的核心训练约 51 分钟；加上周期验证和 checkpoint，预计 55–65 分钟。

从零开始完成数据、训练、采样、物理统计和标准图表，现有脚本基线预计 80–110 分钟。第一次执行还需补齐里程碑 checkpoint、bootstrap 误差条和汇总图脚本，因此承诺完整可审计结果应按 2–3 小时计算；如果结果暴露采样器问题并需要一次针对性复跑，预留 4–6 小时。

本实验回答三个问题：

1. 数据是否真的是二维方格 Ising 模型临界分布，而非未热化、裁剪或存在链泄漏的伪样本？
2. 模型是否同时学到局部统计和 `r=O(L)` 的长程统计？
3. 结论是否对采样 seed 和合理的反向步数稳定，而不是某个采样器超参数的偶然结果？

本轮不声称证明 `L=128/256` 的零样本尺度泛化，也不以“某一张图看起来像分形”作为成功证据。

## 2. 运行时间与硬件实测

远端硬件为单张 RTX 5090（约 32 GB 显存）。基准均在 `L=64`、batch size 16、BF16、fused AdamW、`torch.compile(mode="reduce-overhead")` 下测量。

| 模型 | 参数量 | 首次编译/首轮验证 | 稳态速度 | 峰值保留显存 | 60k 核心训练 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| compact | 5.51M，192 维，6 blocks | 18.1 s | 约 34 step/s | 4.63 GiB | 约 30 min | 冒烟测试、定位故障 |
| full | 12.67M，256 维，8 blocks | 21.2 s | 约 19.6 step/s | 7.60 GiB | 约 51 min | 主 pilot |

采样实测：compact checkpoint 下，`L=64`、ancestral 24 steps、batch 16、128 张样本，包含统计和生成一张比较图共 8.3 秒。按此估计，主实验每个 seed 生成 512 张约 30–40 秒，三个 seed 约 2 分钟以内。

建议的完整时间预算：

| 阶段 | 预计耗时 |
|---|---:|
| 生成 20,000 张 MC 构型 | 1–3 min |
| 数据 QA、关联函数和链诊断 | 2–5 min |
| full 模型 60k step | 55–65 min |
| 主采样、3 seeds 和采样器小消融 | 3–8 min |
| bootstrap、全部图表和报告 | 8–15 min |
| 基线流程合计 | 80–110 min |

当前模型较小，单次训练不建议做两卡 DDP：通信和同步开销很可能抵消收益。若有两张 5090，最有效的使用方式是两卡并行跑不同训练 seed 或 sampler/配置消融，而不是拆分同一个 12.7M 模型。

## 3. 数据生成

### 3.1 物理定义

- 二维正方晶格，`L=64`，周期边界条件。
- 无外场，`J=1`。
- 精确临界逆温度：`beta_c = 0.44068679350977147`。
- Wolff cluster Monte Carlo，直接生成整张 `64 x 64` 构型。
- 16 条独立链，每条使用独立 seed；按整条链划分 train/val/test，禁止同一条链跨 split。

### 3.2 数据量和参数

| 参数 | 数值 | 理由 |
|---|---:|---|
| chains | 16 | 支持链间收敛检查和 chain bootstrap |
| train/val/test chains | 12 / 2 / 2 | 15,000 / 2,500 / 2,500 张 |
| samples per chain | 1,250 | 总计 20,000 张，pilot 足够且生成很快 |
| burn-in sweeps | 500 | 对临界 Wolff 已保守；仍以 trace/R-hat 验收 |
| saved-sample gap | 5 sweeps | 生产阶段固定间隔；独立性最终由 ESS 衡量 |
| adaptation sweeps | 3 | 将 cluster updates 标定为近似物理 sweep |
| pilot cluster steps | 128 | 固定生产间隔前估计平均 cluster 大小 |
| backend/workers | numba / 16 | CPU 并行生成独立链 |
| global seed | 20260805 | 完整记录派生 chain seeds |

执行命令：

```bash
python generate_data.py \
  --output data/ising_l64_pilot.npz \
  --lattice-size 64 --beta 0.44068679350977147 \
  --chains 16 --train-chains 12 --val-chains 2 --test-chains 2 \
  --samples-per-chain 1250 \
  --burn-in-sweeps 500 --sweeps-between 5 \
  --adaptation-sweeps 3 --pilot-cluster-steps 128 \
  --workers 16 --backend numba --seed 20260805
```

随后运行：

```bash
python plot_training_data.py \
  --data data/ising_l64_pilot.npz \
  --output-dir artifacts/pilot_l64/data_qa --max-lag 64
```

### 3.3 数据硬门槛

任一硬门槛失败就停止训练，先修数据：

- 所有自旋严格属于 `{-1,+1}`，shape 与元数据一致。
- train/val/test 的 chain id 交集为空。
- `G(0)=1`、能量—最近邻关联恒等式和结构因子 sum rule 在数值精度内成立。
- energy 与 `|m|` 的 split-R-hat 均不大于 1.05。
- 每条链 energy 与 `|m|` 的 ESS 至少为 30；低于 100 记警告并增加保存间隔。
- 局部 Gibbs 条件概率五个邻居和 bin 的最大绝对偏差不超过 0.05。
- `|<m>| <= 0.10`；若不满足，检查有限链的正负磁化驻留，并在训练中保留全局 spin-flip augmentation。
- 物理量落在宽松临界窗口：`<e>` 距 `-sqrt(2)` 小于 0.08，`U4` 距 0.61069 小于 0.10，`xi/L` 距 0.90505 小于 0.20。

此前 L=64 小样本参考值为 `<e>=-1.42849`、`<|m|>=0.6069`、`U4=0.6134`、`xi/L=0.9196`。这些只作 sanity anchor；正式 ground truth 必须由本轮独立 test chains 及其置信区间给出，不能把单次点估计当作精确答案。

## 4. 模型与离散扩散

### 4.1 模型

- 输入 token：`-1`、`+1` 和 absorbing `[MASK]`，网络输出两类 clean-spin logits。
- 周期 circular `3 x 3` stem，避免人为边界。
- 8 个 axial-attention blocks，`d_model=256`、8 heads、MLP ratio 4，共约 12.67M 参数。
- 行、列注意力让单层即可沿整行/整列交换信息；多层组合产生二维全局感受野。
- 周期 log-distance attention bias，距离按 torus 最短距离定义。
- 连续扩散时间嵌入和 adaptive normalization。
- D4 旋转/反射增强与精确的全局 spin flip 增强。

### 4.2 与 `block_diffusion.py` 的关系

采用其核心思路：离散 token、absorbing mask corruption、随机噪声时间训练、从全 mask 逐步解除遮挡，而不是把二值自旋错误地当连续 RGB 高斯扩散。当前实现针对 Ising 做了三项关键改造：周期边界、全局二维 axial mixing、物理统计驱动的 sampler 验收。

`full_mask_probability` 设为 0.02，仅用于校准无条件生成起点并压制有限数据或数值误差造成的自旋符号偏置。需要特别注意：零外场下，全 mask 输入不含任何破缺对称性的信息，每个位置的理论单点边缘严格为 50/50，因此 `t=1` 的最优逐点 CE 就是 `ln 2`，它不可能靠训练降到零。空间关联是在第一批自旋被随机解除遮挡后，通过后续条件预测逐步建立的；端点能力应主要由 `t=0.95–0.99` 的近全 mask 输入和最终联合采样判断。将精确全 mask 控制在 2%，可把绝大多数计算用于仍保留少量条件信息的噪声状态。

## 5. 训练方案

主配置已经写入 `configs/pilot_l64.json`。关键参数如下：

| 参数 | 数值 |
|---|---:|
| optimizer | AdamW，betas `(0.9,0.95)`，weight decay 0.05 |
| batch size | 16 |
| steps | 60,000 |
| peak/min LR | `3e-4 / 3e-5` |
| warmup | 2,000 steps |
| schedule | warmup + cosine decay |
| grad clip | 1.0 |
| EMA | 0.999；评估一律优先 EMA |
| precision | BF16 |
| diffusion t | `[0.01,1.0]`，另有 2% 精确全 mask |
| validation | 每 1,000 steps，固定 256 张、固定 corruption seed |
| checkpoints | 每 5,000 steps；best + last + milestone |

执行：

```bash
python train.py --config configs/pilot_l64.json
```

训练前需做一项小改进：当前 `last.pt` 会在 checkpoint interval 上覆盖，正式实验要额外保留 `step_005000.pt`、`step_015000.pt`、`step_030000.pt`、`step_045000.pt` 和 `step_060000.pt`，用来判断长程关联是在何时学到，而不仅比较 best/last。

### 5.1 训练监控和停止规则

保存每个验证点的：

- train loss、masked CE、mask fraction、mean t、gradient norm、LR 和 wall time；
- 总 validation NELBO；
- `t={0.01,0.05,0.1,0.2,0.4,0.6,0.8,0.9,0.97,1.0}` 的分层 validation loss；
- best checkpoint step 和每 5k checkpoint 的 SHA256。

立即停止并诊断的条件：NaN/Inf、连续 500 steps 梯度异常、validation 三次连续明显恶化、`t=0.97` 等近端点噪声层长期没有改善，或训练吞吐较基准下降超过 25%。`t=1` 应接近理论最优值 `ln 2 ≈ 0.693`；若明显偏离且伴随固定磁化符号偏置，才需要检查对称性或端点校准。

60k 是首轮固定终点，不依据 test set 选 checkpoint。若 45k 到 60k 的 validation 和生成指标都已平台化，结束；若仍稳定改善，可从 `last.pt` 延长到 90k，预计额外 30–40 分钟。

## 6. 采样方案

### 6.1 主结果

- checkpoint：validation 选出的 `best.pt`，EMA 权重。
- sampler：`ancestral`。
- reverse steps：24。
- temperature：0.9。
- selection noise：0.05。
- sampling seeds：1234、2345、3456。
- 每个 seed 512 张，共 1,536 张；不按视觉效果删图。
- reference：固定的 2,500 张 held-out MC test samples。

示例命令（其余 seed 改输出目录和 `--seed`）：

```bash
python sample_and_evaluate.py \
  --checkpoint artifacts/pilot_l64/best.pt \
  --output-dir artifacts/pilot_l64/eval/ancestral_s1234 \
  --lattice-size 64 --samples 512 --batch-size 16 \
  --steps 24 --temperature 0.9 --sampler ancestral \
  --selection-noise 0.05 \
  --reference-data data/ising_l64_pilot.npz \
  --device cuda --seed 1234
```

多个 seed 的价值不是“重复展示几张图片”，而是估计同一个 checkpoint 的采样方差、符号 mode balance 和低 `k` 指标稳定性。主结论报告聚合指标与 seed 间误差，不把三次结果当三套独立训练模型。

### 6.2 小规模 sampler 消融

每项先生成 128 张，只用于识别敏感性：

| ID | sampler | steps | T | 目的 |
|---|---|---:|---:|---|
| A | ancestral | 24 | 0.9 | 主设置 |
| B | ancestral | 48 | 1.0 | 检查离散化/温度依赖 |
| C | confidence | 24 | 1.0 | 检查并行置信度提交是否过度有序 |
| D | ancestral_corrector | 24 | 0.9 | 检查少量重遮挡能否修正早期错误 |

若 A/B 的能量、`U4` 或 `xi/L` 相差超过各自 MC bootstrap 95% CI，判为 sampler 敏感，不能只挑最好看的设置。此时优先校正 transition kernel、端点训练和步数收敛，再考虑调 temperature。

## 7. 评估指标与成功标准

所有指标对 model 和 held-out MC 使用同一实现。MC 置信区间按整条链 bootstrap；model 按 sampling seed 分层 bootstrap。保存点估计、标准误和 95% CI。

### 7.1 必报指标

- 每站点能量分布 `e`。
- signed `m` 和 `|m|` 分布、正负磁化样本比例。
- Binder cumulant `U4 = 1 - <m^4>/(3<m^2>^2)`。
- second-moment correlation length `xi_2/L`。
- 周期二维 raw 和 connected `G(r)`；在 `2 <= r <= 16` 拟合有效 `eta`。
- `G_model(r)/G_MC(r)-1`，重点看 `r=8..32`。
- 径向结构因子 `S(k)`，并单列 `(1,0)`、`(0,1)`、`(1,1)` 低波数 modes。
- 样本多样性、重复率和最近邻 Hamming 距离。

### 7.2 Pilot 判据

成功不要求每项都达到论文精度，但至少同时满足：

| 项目 | Pilot 通过线 |
|---|---|
| spin-flip symmetry | `|<m>| < 0.10`，正/负磁化各占 35%–65% |
| mean energy | 与 MC 差小于 0.02，或落在 MC 95% CI 的两倍宽度内 |
| `<|m|>` | 绝对误差小于 0.05 |
| Binder U4 | 绝对误差小于 0.04 |
| `xi/L` | 绝对误差小于 0.10 |
| `G(r)` | `r=1..16` 归一化 RMSE 小于 0.08，且 `r=8..16` 无系统性塌缩 |
| 有效 eta | 中间尺度拟合与 MC 相差小于 0.06；理论参考约 0.25 |
| seed stability | 三个 seed 的主要标量差异不超过 MC/采样联合 95% CI 的约两倍 |

这些是 pilot 决策阈值，不是精确物理论证。最终结论应同时展示数值误差和置信区间，不能只写 PASS/FAIL。

## 8. 必须保存的文件和图表

建议目录：

```text
artifacts/pilot_l64/
  manifest.json
  environment.json
  data_qa/
  checkpoints/
  training/
  eval/
    ancestral_s1234/
    ancestral_s2345/
    ancestral_s3456/
    sampler_ablation/
  figures/
  tables/
  REPORT_ZH.md
```

原始数据与可复现信息：

- `ising_l64_pilot.npz`、完整 metadata、16 个 chain seeds 和 split chain ids；
- 冻结后的 data/train/sample/eval 配置；
- git commit、dirty diff hash、Python/PyTorch/CUDA/驱动/GPU 信息；
- best、last、指定 milestone checkpoints，含 model、EMA、optimizer 和 step；
- `history.json` 与扁平化 `history.csv`；
- 每个 sampler/seed 的全部生成构型 `samples.npz`，不是只保存入选图片；
- 所有标量、曲线数组、bootstrap replicates、`metrics.json/csv`。

最终至少输出下列 PNG 和 PDF：

1. `01_mc_samples_by_chain`：固定索引、跨链、无挑图的数据样本网格。
2. `02_mc_trace_acf_ess`：energy/`|m|` trace、ACF、ESS 和 R-hat。
3. `03_mc_physics_sanity`：local Gibbs、`G(r)`、`S(k)`、恒等式检查。
4. `04_training_curves`：loss、validation NELBO、LR、gradient norm。
5. `05_validation_by_t`：各噪声层 loss 曲线或 heatmap，突出 `t=1`。
6. `06_gt_vs_generated_grid`：同一色标的 ground truth 与生成样本网格；固定随机索引并在图注中列出 seed。
7. `07_energy_magnetization_hist`：energy、signed `m`、`|m|` 分布和 bootstrap band。
8. `08_correlation_loglog`：raw/connected `G(r)`、MC、model、`r^-1/4` 参考斜率和 95% CI。
9. `09_correlation_relative_error`：随距离变化的相对误差。
10. `10_structure_factor`：径向 `S(k)` 和最低几个离散 mode。
11. `11_dimensionless_observables`：`U4`、`xi/L`、energy、`|m|` 的带误差条比较。
12. `12_sampler_seed_stability`：sampler × seed 的关键指标矩阵。
13. `13_checkpoint_trajectory`：5k–60k checkpoint 的局部与长程指标演化。

`06_gt_vs_generated_grid` 同时保存两种版本：一种固定随机索引，防止 cherry-picking；另一种按磁化分位数选择典型样本，用于直观展示正、负和近零磁化区域，但必须明确标注选择规则。

## 9. 预期看到的实验现象

正确 MC 数据应表现为：

- 红蓝两类自旋的全局平均近似各半，但单张有限尺寸构型不必 50/50；单张图通常具有非零甚至较大的磁化。
- 多尺度 domain 共存，边界粗糙、没有单一特征长度；不是均匀随机盐胡椒，也不是几块光滑的大色块。
- signed `m` 分布关于零对称，`<|m|>` 在本尺度约 0.60。
- `G(r)` 在中间尺度近似 `r^{-1/4}`；接近 `L/2` 时受周期 torus 和有限尺寸影响，不应强求整段直线。
- `U4` 约 0.61、`xi/L` 约 0.91，允许有限样本波动。

训练和生成通常按以下顺序改善：

1. loss 很快低于随机猜测 `ln 2`，最近邻能量先变合理；
2. 中等 `t` 的 validation loss 先下降，`t=0.97` 等近端点最慢；精确 `t=1` 的逐点 CE 应保持在 `ln 2` 附近；
3. magnetization 分布和中程 `G(r)` 改善；
4. 最低 `k` 的结构因子、`xi/L` 和大距离 `G(r)` 最后收敛。

失败模式及解释：

| 现象 | 更可能的原因 |
|---|---|
| loss 降、能量正确，但大 `r` 的 `G(r)` 明显偏低 | 只学到局部纹理，模型容量/全局 mixing/训练长度不足 |
| 同一 checkpoint 随 steps 或 T 巨变 | 反向 kernel 或端点校准问题，不能归咎于训练数据 |
| 能量过低、`|m|` 过大、样本接近全红/全蓝 | 采样过度自信，等效温度偏低或 confidence 过早提交 |
| 能量接近 0、`m` 近 0、`G(r)` 很短 | 接近随机自旋，欠训练或全 mask 端点未学到 |
| 几乎只有一种磁化符号 | mode collapse 或采样对称性被破坏 |
| validation 很好而 test/生成差 | 链泄漏、过拟合，或 denoising objective 与生成 kernel 不一致 |
| `t=1` logits 明显偏向一种符号，或 CE 与 `ln 2` 显著不符 | 有限样本对称性偏置、全局 spin-flip 处理或端点校准异常 |

## 10. 决策顺序

1. 数据 QA 全通过后才训练。
2. 先看 loss 和 per-t validation，只用于检查优化是否正常。
3. 再看 energy/局部 Gibbs，确认局部规律。
4. 最后以 `G(r)`、low-k `S(k)`、`U4`、`xi/L` 判断长程规律。
5. 若主模型失败，先区分“模型失败”还是“sampler 失败”，再决定补训或改 kernel。
6. L=64 通过后，再做不训练的 L=128/256 尺度外测试；不能用 L=64 成功直接宣称大尺度成功。

首轮只固定一个训练 seed，避免把计算预算分散在三个尚未验证的训练上；通过主判据后再增加第二个训练 seed。三个 sampling seed 从首轮就保留，因为代价很低且直接检验采样稳定性。
