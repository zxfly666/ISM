# ISM: Critical Ising masked discrete diffusion

本项目研究二维临界 Ising 构型的吸收态离散扩散模型。仓库保存了完整的
`L=64` pilot：Monte Carlo 数据、训练代码与配置、训练日志、采样器消融、
多 seed 评估、关联函数与 scaling 分析。最终书面报告按仓库所有者要求未上传。

## 快速导航

| 目标 | 入口 |
|---|---|
| 理解项目设计 | [`docs/TECHNICAL_DESIGN_ZH.md`](docs/TECHNICAL_DESIGN_ZH.md) |
| 查看正式实验方案 | [`docs/L64_PILOT_EXPERIMENT_PLAN_ZH.md`](docs/L64_PILOT_EXPERIMENT_PLAN_ZH.md) |
| 复现正式训练 | [`configs/pilot_l64.json`](configs/pilot_l64.json) 与 `train.py` |
| 查看最终数值摘要 | [`artifacts/final_l64/final_summary.json`](artifacts/final_l64/final_summary.json) |
| 查看训练与采样产物 | [`artifacts/README.md`](artifacts/README.md) |
| 理解数据文件 | [`data/README.md`](data/README.md) |
| 下载大模型权重 | [`CHECKPOINTS.md`](CHECKPOINTS.md) |

## 正式 `L=64` pilot 概览

- 临界温度：`T_c = 2.269185...`，周期边界，Wolff cluster Monte Carlo。
- 训练：60,000 steps，batch size 16，BF16，EMA，约 12.7M 参数。
- 最佳验证 NELBO：`0.361248`；60k 时为 `0.361257`。
- 正式采样：ancestral sampler，128 reverse steps，3 个采样 seed，合计 4,608 张。
- 独立参考：10,000 张 MC 构型、8 条独立链。
- 关键结果：`G(r)` 在 `r=8,16,32` 的模型/MC 比值分别约为
  `0.9975/0.9994/0.9978`；全部预设 pilot checks 通过。

## 正式复现

```bash
pip install -r requirements.txt
python train.py --config configs/pilot_l64.json
```

如果只做采样评估，可先按 [`CHECKPOINTS.md`](CHECKPOINTS.md) 下载
`best.pt`，然后执行：

```bash
python sample_and_evaluate.py \
  --checkpoint artifacts/pilot_l64/best.pt \
  --output-dir artifacts/reproduction_s1234 \
  --lattice-size 64 --samples 1536 --batch-size 16 \
  --steps 128 --temperature 1.0 --sampler ancestral \
  --reference-data data/ising_l64_reference_10k.npz \
  --device cuda --seed 1234
```

```text
ISM/
├── ism_diffusion/   # 模型、离散扩散、Ising MC 与物理指标
├── configs/         # smoke、benchmark 与正式 L=64 配置
├── data/            # 可直接加载的 MC 训练/参考数据
├── artifacts/       # 训练日志、采样结果、图表与消融实验
├── docs/            # 技术设计、实验方案、诊断与审计记录
├── scripts/         # 报告构建等辅助工具
├── tests/           # 核心单元测试
├── train.py         # 正式训练入口
└── sample_and_evaluate.py
```

项目早期先在 `L=16/32` 上完成链路和采样器 smoke test，随后将正式 pilot
改为直接训练 `L=64`；因此早期配置和消融产物仍保留用于审计，但正式结论以
`configs/pilot_l64.json` 和 `artifacts/final_l64/` 为准。

## Method

This project trains an absorbing-state discrete diffusion model on critical
two-dimensional Ising configurations.

The implementation adapts the objective in
[`block_diffusion.py`](https://github.com/suning-git/nanoinfra/blob/main/exemplars/nano_world_model/block_diffusion.py):

- the absorbing state is a third `[MASK]` token;
- each spin is independently masked with probability `t`;
- the model predicts the clean token **at the masked position**, not the next token;
- loss is computed only on masked sites and weighted by `1/t`;
- validation uses a fixed noise grid and fixed random masks.

The language-model-specific clean/noisy two-stream layout and compiled
large-vocabulary head are intentionally absent. Ising denoising is spatial and
non-causal, and its output vocabulary has only two classes.

## Model

The denoiser is a size-flexible 2D axial Transformer:

- alternating row and column attention gives global 2D communication without
  full `L^4` attention;
- circular convolutions encode local periodic neighbors;
- the attention bias is a learned function of toroidal distance,
  `-a_h log(1 + d_periodic)`, so no learned maximum position is tied to `L=32`;
- continuous diffusion time enters every block through adaptive LayerNorm;
- the same weights accept `32`, `64`, or `128` without resizing parameters.

Input tokens are `0 = -1`, `1 = +1`, `2 = MASK`. The head produces two clean-spin
logits per lattice site.

## Objective

For clean configuration `x0`, draw `t` uniformly from `[t_min, 1]` and form `xt`
by masking each site with probability `t`. The training estimator is

```text
L = (1 / (B L^2)) sum_i 1[x_t,i = MASK] CE(x0,i, p_theta(.|xt,t)) / t .
```

An optional small probability mass at exactly `t=1` trains the all-mask endpoint
used by unconditional generation. This is an endpoint regularizer in addition to
the continuous-time NELBO.

The default sampler is posterior-consistent ancestral revealing. If `s < t`, a
site masked at noise level `t` becomes visible with probability `1 - s/t`; its
spin is sampled from the denoiser prediction. This avoids the strong
low-temperature selection bias observed with confidence-ranked commits after
the denoiser is well trained.

Four samplers are exposed for controlled comparisons:

- `ancestral` (default): posterior-consistent stochastic reverse reveals;
- `ancestral_corrector`: ancestral reveals plus blocked conditional resampling;
- `confidence_corrector`: confidence decoding plus early backtracking;
- `confidence`: the irreversible confidence baseline;

Ancestral sampling requires a reasonably trained and calibrated denoiser. In an
80-step smoke run it can look nearly random, while after 2000 steps the
confidence sampler collapses to almost uniform-spin configurations and ancestral
sampling becomes substantially better. Sampler quality must therefore be
re-evaluated as training progresses and compared across several sampling seeds.

The same sampler can later be used for conditional expansion by fixing a center
patch and masking only its exterior.

## Environment

Only NumPy, PyTorch, tqdm, and Matplotlib are required. Install the pinned
project dependencies with:

```bash
pip install -r requirements.txt
```

The formal run used CUDA with BF16 on an RTX 5090. CPU is sufficient for unit
tests and the smallest smoke configuration, but not recommended for the full
`L=64`, 60k-step training run.

## Fast end-to-end smoke run

Generate a small chain-split `16 x 16` dataset:

```powershell
python generate_data.py `
  --output data/ising_l16_smoke.npz `
  --lattice-size 16 `
  --chains 4 --train-chains 2 --val-chains 1 --test-chains 1 `
  --samples-per-chain 48 --burn-in-sweeps 30 --sweeps-between 2 `
  --seed 17
```

Train the small CPU model:

```powershell
python train.py --config configs/smoke.json
```

Generate samples and compare with the held-out MC chain:

```powershell
python sample_and_evaluate.py `
  --checkpoint artifacts/smoke/best.pt `
  --output-dir artifacts/smoke_eval `
  --lattice-size 16 --samples 32 --batch-size 8 --steps 12 `
  --sampler ancestral --temperature 0.9 `
  --reference-data data/ising_l16_smoke.npz --device cpu
```

This run verifies the pipeline; it is not evidence of learned critical physics.

## Phase-one experiment

1. Generate the chain-split training set:

```powershell
python generate_data.py `
  --output data/ising_l32_phase1.npz `
  --lattice-size 32 `
  --chains 16 --train-chains 12 --val-chains 2 --test-chains 2 `
  --samples-per-chain 1250 `
  --burn-in-sweeps 500 --sweeps-between 5 `
  --seed 20260731
```

The fixed five-sweep spacing is only an initial value. Run an autocorrelation
pilot for energy and `|m|`, then increase the spacing to at least roughly five
integrated autocorrelation times before treating samples as effectively
independent.

2. Train:

```powershell
python train.py --config configs/phase1.json
```

The full config has about 12.7 million parameters and is intended for a CUDA GPU.
On the tested RTX 5090, an `L=32`, batch-32 capacity run reached about 29.6
training steps/s after compilation and used about 4.03 GiB peak reserved memory.
The 60k-step core loop is therefore roughly 34 minutes before validation,
checkpoint, and data-generation overhead.

3. Evaluate same-size fidelity first, then scale extrapolation:

```powershell
python sample_and_evaluate.py `
  --checkpoint artifacts/phase1/best.pt `
  --output-dir artifacts/eval_l32 `
  --lattice-size 32 --samples 10000 --batch-size 32 --steps 32 `
  --reference-data data/ising_l32_phase1.npz

python sample_and_evaluate.py `
  --checkpoint artifacts/phase1/best.pt `
  --output-dir artifacts/eval_l64 `
  --lattice-size 64 --samples 10000 --batch-size 8 --steps 32
```

For `L=64` and `L=128`, the evaluation command generates an independent same-size
Wolff reference when `--reference-data` is omitted.

## What constitutes success

Generated samples must match the **distribution** of same-size MC samples, not any
particular configuration. Report at least:

- full energy and magnetization distributions;
- real-space connected correlation `G(r)`, especially distances beyond the
  training window;
- structure factor `S(k)`, especially the unseen low-wave-number region;
- dimensionless finite-size quantities `xi_2 / L` and Binder cumulant `U4`;
- sample-to-sample variance and multiple training seeds to expose mode collapse.

`sample_and_evaluate.py` computes the first-pass versions of these observables and
saves both `metrics.json` and `comparison.png`. A publication experiment should
add bootstrap confidence intervals and aggregate at least three training seeds.

## Complete experiment artifacts

The repository includes the training/evaluation code, configurations, Ising
datasets, logs, plots, sampled lattices, and compact analysis artifacts from the
L=64 pilot. The seven approximately 203 MB PyTorch checkpoints are published as
assets in the `l64-pilot-checkpoints` GitHub Release; see
[`CHECKPOINTS.md`](CHECKPOINTS.md) for paths and SHA-256 checksums.

The final written report is intentionally excluded from this repository at the
owner's request. The peer `Hackathon-3` checkout used for comparison is also not
republished; its public source remains at
<https://github.com/zdacongming-glitch/Hackathon-3>.
