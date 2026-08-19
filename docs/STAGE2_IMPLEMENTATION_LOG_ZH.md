# Stage 2 实施与审计日志

## 2026-08-19：2A 实现冻结前检查

状态：Stage 2A 与单 seed Stage 2B 因果筛选均已完成；3-seed 正式确认待执行。

### 已实现组件

- `ism_diffusion/stage2_model.py`
  - 独立 Local/Global hidden streams；
  - Local 为一次 Manhattan 半径 1 attention 加三层逐点 MLP，总空间感受野严格不扩张；
  - Global 为 dense 2D-RoPE expert；
  - final logits 为 local base 加 gated global residual；
  - global residual zero initialization，gate 初值 0.1；
  - `LG-HardMarkov` 仅作为诊断开关。
- `ism_diffusion/stage2_data.py`
  - Stage 2A uniform-stride matched/unit sampling；
  - Stage 2B real integer RandomGap sampling；
  - matched coordinates 与 rank/unit causal control。
- `ism_diffusion/stage2_sampling.py`
  - S0 irreversible reveal；
  - S1 confidence-ordered MaskGIT-style decoding；
  - S2 S1 加 checkerboard pseudo-Gibbs refinement。
- `train_stage2.py`
  - BF16/FP32、EMA、原子 checkpoint；
  - Python/NumPy/Torch/device RNG 全状态保存；
  - 精确断点续跑；
  - gate、local/global logit norm 随验证时间步记录。
- `run_stage0_sampler.py` 与 `analyze_stage0_sampler.py`
  - 按模型、方法、步数、seed 单独保存，可跳过已完成组合；
  - 仅使用 validation MC split 冻结 sampler。
- `probe_stage2_coordinates.py`
  - paired continuous coordinate response；
  - 保存完整 NLL/Brier 逐样本张量和 JSON 汇总。
- `run_stage2_markov_curve.py`
  - 多 context width 精确 Markov 污染曲线；
  - FP32 probe 与逐样本 CSV。
- `analyze_stage2a.py`
  - 预注册硬门槛；
  - paired bootstrap；
  - `STRONG_GO/GO/CONDITIONAL_GO/NO_GO`；
  - 自动形成 2B 修改建议；
  - PNG 300 dpi 与矢量 PDF 图表。

### 参数公平性

| 模型 | 参数量 |
|---|---:|
| 原 6-block Dense | 1,713,282 |
| Local–Global | 1,978,501 |
| 7-block Dense+ | 1,976,706 |

Local–Global 相对原 Dense 多 15.5%，因此正式 2A 增加 `Dense-T3+` 和 `Dense-Punit+`；Dense+ 与 Local–Global 参数误差约 0.09%。

### 本地验证

- 26/26 单元测试通过；
- Stage 2 训练 smoke 完成；
- 原子 checkpoint、EMA、history 和 RNG 状态存在；
- 从 step 2 精确恢复并完成 step 3；
- S0/S1/S2 均可输出无 MASK 的二值格点；
- 连续坐标 probe 与 Markov curve smoke 完成；
- PAD frame 有限且 FP32 invariant；
- 四层 local capacity 不扩大中心 site 的物理空间感受野。

### 已修正的方案问题

1. 原方案 `R_phys=2` 且四层 local attention 会产生多跳感受野扩张，已改为一次 Manhattan `R=1` attention 加三层 pointwise MLP。
2. `L_parent=2048, W=128, gap=10` 违反半父场跨度约束。正式 2B 改为：
   - `W=64/96` 可测试 gap 10；
   - `W=128` 最大测试 gap 6；
   - 若坚持 `W=128, gap=10`，必须生成 `L_parent>=4096`。
3. Local–Global 参数量比原 Dense 高 10% 以上，已加入几乎精确匹配的 7-block Dense+ 对照。

### 执行环境

正式运行使用单张 RTX 5090 D、PyTorch 2.8.0+cu128、BF16 训练与 FP32 机制
probe。长任务使用可恢复 checkpoint 与持久化远端会话，结果写入独立数据盘。

## 2026-08-19：Stage 2A 正式结果

- 远端 27 项测试中的 Stage 2A 版本为 26/26 通过；
- Stage 0 在 3 模型、3 sampler、4 step 数和 2 seeds 上冻结 `S0 + 256 steps`；
- 四个 8000-step 模型均完成，Dense+ 约 7.1 分钟/模型，Local--Global 约 8.65 分钟/模型；
- `LG-T3` 相对参数匹配 `Dense-T3+`：
  - W=64 distant-MASK Markov pollution 从 `0.1822` 降到 `0.0193`；
  - distant-visible pollution 从 `0.0113` 降到约 `0.00008`；
  - 生成 short/expanded `G(r)` NRMSE 为 `0.0162/0.0492`；
  - energy absolute error 为 `0.0266`；
- 正确物理坐标相对 unit 坐标平均 NLL 改善 `0.03225`，95% CI `[-0.03282,-0.03169]`；
- 但 `LG-T3 - LG-Punit=-0.00164`，统计显著但低于预设 practical threshold `0.002`；
- 最终判定：`CONDITIONAL_GO`。

Markov 后处理从逐 context、batch=4 改为三个同形大 context 合批、base batch=16。2048 行逐样本对照的概率最大差异为 `2.38e-7`；W=64 微基准中 batch=16 比 batch=32 更快，峰值显存 `13.4 GiB`，正式计算段峰值 GPU 利用率 100%。

## Stage 2B 修改：先做因果分辨 screen

`CONDITIONAL_GO` 不足以直接启动原计划的 12 个正式 run。先固定单 seed、8000 updates，新增三个完全同构且初始化/RNG 配对的模型：

| 模型 | 自旋取样 | 坐标 | 排除的替代解释 |
|---|---|---|---|
| `LG-Gap-Unit` | 样本内 RandomGap | rank/unit | data-only augmentation |
| `LG-Gap-Matched` | 同一 RandomGap | 精确物理坐标 | matched-distance 核心假设 |
| `LG-U-RandPE` | 连续 stride-1 窗口 | 同分布但与自旋不匹配的 RandomGap 坐标 | randomized positional regularization |

训练宽度 `{16,24,32,48,64}`，训练 gap `{1,2,4,8}`；测试包含 W=64 的 seen mixture、held-out `{3,6}` mixture、固定 3/6，以及 W=48 的固定 10。所有模型使用相同 clean sample、mask trace 和 probe seed。只有 `LG-Gap-Matched` 同时以至少 `0.002` NLL 优于 data-only 与 RandPE，并在 unit/shuffled coordinate swap 下显著退化，才进入三 seeds、15000-step 的完整 Stage 2B。

## 2026-08-19：Stage 2B 单 seed 因果筛选结果

状态：完成，退出码 `0`，自动判定为 **`GO_FULL_2B`**。

### 训练与有效性

- `LG-Gap-Unit`、`LG-U-RandPE`、`LG-Gap-Matched` 均完成 8,000 updates；
- 三个模型均为 1,978,501 参数；
- 初始化 SHA-256 均为
  `e9cdb0c87427e676f3790dccb38abf8139aa1f6c011b5bbe759d5bd1baf2d9f2`；
- 27/27 单元测试通过，无 NaN 或 OOM；
- probe 使用 FP32、batch 16、512 个 paired samples、
  `t={0.2,0.5,0.8,0.95}`；
- 测试几何包括 W=64 seen gap mixture、held-out `{3,6}` mixture、固定 gap 3/6，
  以及 W=48 固定 gap 10。

### 预注册因果对比

下表为 held-out `{3,6}` RandomGap 上的配对 NLL 差；负数有利于 matched physical
coordinates。

| 对比 | 均值 | 95% CI |
|---|---:|---:|
| Gap-Matched − Gap-Unit | -0.08238 | [-0.08591, -0.07890] |
| Gap-Matched − U-RandPE | -0.03142 | [-0.03187, -0.03095] |
| Gap-Matched − uniform LG-T3 | -0.00997 | [-0.01053, -0.00945] |
| CorrectCoord − UnitCoord | -0.01262 | [-0.01293, -0.01231] |
| CorrectCoord − ShuffledCoord | -0.00682 | [-0.00707, -0.00657] |

五项检查全部通过，且前三项远大于预注册 practical threshold `0.002`。这排除了
“仅仅见过稀疏数据”“只是随机位置编码正则化”以及“模型忽略坐标”三个主要替代解释。

### GPU 与异常恢复

- 单模型训练约占 34%--64% GPU；后两个独立对照并行后稳定在 91%--99%；
- 正式 probe 连续达到 99%--100%，峰值显存约 13.48 GiB，温度不高于约 67°C；
- 远端实例曾在 probe 中异常中断。三个 8,000-step checkpoint 均已原子落盘，
  重连后只重跑尚未落盘的 probe；训练没有重复；
- 恢复运行在 `2026-08-19T19:39:21+08:00` 正常完成。

### 结论边界

本结果证明单 seed 因果筛选值得进入完整 Stage 2B，但不能替代论文级重复。下一步严格
按照预注册方案运行 `{1234,2345,3456}` 三个训练 seed、15,000 updates，并使用
hierarchical bootstrap；在此之前不把当前效应写成跨 seed 的稳定方法结论。

精简结果、图表、逐样本数组、训练历史和审计文件位于
`results/scale_aware_context/`。
