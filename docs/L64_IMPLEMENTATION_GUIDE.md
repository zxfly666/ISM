# 早期L64模型实现与运行说明

本页保留早期L64实现说明与工程命令，不代表后期dense模型的结构或当前科研结论。执行训练会产生新运行；整理和阅读报告不需要运行这些命令。当前入口见[综合报告](../experiments/REPORT_ZH.md)与[复现边界](../experiments/REPRODUCING.md)。

下文的历史未公开书面报告指原私人版本，不是目前已公开的独立科研讲解报告。命令按仓库根目录运行。

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

This section describes the **early L64 pilot**, not the later 1,976,706-parameter
dense coordinate model. See the [experiment map](../experiments/README.md) for the
architecture and checkpoint lineage of each subsequent campaign.

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
datasets, logs, plots, sampled lattices, and analysis artifacts from the L=64
pilot. The approximately 203 MB best checkpoint is managed with Git LFS; see
[`CHECKPOINTS.md`](../CHECKPOINTS.md) for its path and SHA-256 checksum. Resume and
milestone weights are intentionally kept out of the repository.

The final written report is intentionally excluded from this repository at the
owner's request. The peer `Hackathon-3` checkout used for comparison is also not
republished; its public source remains at
<https://github.com/zdacongming-glitch/Hackathon-3>.
