"""Aggregate and visualize the L=64 -> L=128 zero-shot experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_final_l64 import (
    BLACK,
    BLUE,
    BLUE_LIGHT,
    GREY,
    PALE_BLUE,
    PALE_GREY,
    RED,
    TEAL,
    apply_style,
    confidence_interval,
    fit_eta,
    metrics_from_means,
    model_bootstrap,
    point_metrics,
    reference_bootstrap,
    save_figure,
    spectral_features,
    sufficient_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--l64-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def nrmse(model: np.ndarray, reference: np.ndarray, selected: np.ndarray) -> float:
    error = np.sqrt(np.mean((model[selected] - reference[selected]) ** 2))
    scale = np.sqrt(np.mean(reference[selected] ** 2))
    return float(error / scale)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with np.load(args.reference, allow_pickle=False) as payload:
        reference = np.asarray(payload["test"], dtype=np.int8)
        chain_ids = np.asarray(payload["test_chain_id"], dtype=np.int16)
        reference_metadata = json.loads(str(payload["metadata"].item()))

    model_by_seed: list[np.ndarray] = []
    seed_names: list[str] = []
    for path in args.model:
        with np.load(path, allow_pickle=False) as payload:
            model_by_seed.append(np.asarray(payload["model"], dtype=np.int8))
        seed_names.append(path.parent.name.replace("formal_", ""))
    if len(model_by_seed) < 2:
        raise ValueError("At least two independent sampling seeds are required")
    model = np.concatenate(model_by_seed, axis=0)
    length = int(reference.shape[-1])
    if length != 128 or any(rows.shape[-2:] != (length, length) for rows in model_by_seed):
        raise ValueError("This analysis expects L=128 reference and model samples")

    np.savez_compressed(
        args.output_dir / f"model_samples_{len(model)}.npz",
        model=model,
        seed_id=np.concatenate(
            [np.full(len(rows), i, dtype=np.int8) for i, rows in enumerate(model_by_seed)]
        ),
    )

    reference_corr, reference_sf, reference_s0, reference_sk = spectral_features(
        reference, batch_size=128
    )
    model_features = [spectral_features(rows, batch_size=128) for rows in model_by_seed]
    model_corr_by_seed = [item[0] for item in model_features]
    model_sf_by_seed = [item[1] for item in model_features]
    reference_arrays = sufficient_arrays(reference, reference_s0, reference_sk)
    model_arrays_by_seed = [
        sufficient_arrays(rows, features[2], features[3])
        for rows, features in zip(model_by_seed, model_features, strict=True)
    ]
    model_arrays = {
        key: np.concatenate([rows[key] for rows in model_arrays_by_seed])
        for key in model_arrays_by_seed[0]
    }

    ref_boot_means, ref_boot_corr, ref_boot_sf = reference_bootstrap(
        reference_arrays,
        reference_corr,
        reference_sf,
        chain_ids,
        args.bootstrap,
        rng,
    )
    mod_boot_means, mod_boot_corr, mod_boot_sf = model_bootstrap(
        model_arrays_by_seed,
        model_corr_by_seed,
        model_sf_by_seed,
        args.bootstrap,
        rng,
    )
    ref_metrics = point_metrics(reference_arrays, length)
    mod_metrics = point_metrics(model_arrays, length)
    ref_boot_metrics = metrics_from_means(ref_boot_means, length)
    mod_boot_metrics = metrics_from_means(mod_boot_means, length)
    seed_metrics = [point_metrics(rows, length) for rows in model_arrays_by_seed]

    ref_c = reference_corr.mean(axis=0)
    mod_c = np.concatenate(model_corr_by_seed).mean(axis=0)
    ref_s = reference_sf.mean(axis=0)
    mod_s = np.concatenate(model_sf_by_seed).mean(axis=0)
    radius = np.arange(length // 2 + 1)
    wave_number = 2.0 * np.pi * radius / length
    seen = np.arange(1, 33)
    extrapolated = np.arange(33, 65)
    nrmse_seen = nrmse(mod_c, ref_c, seen)
    nrmse_extrapolated = nrmse(mod_c, ref_c, extrapolated)
    correlation_ratio = mod_c / ref_c
    structure_ratio = mod_s[1:] / ref_s[1:]
    scaling = {
        "short_to_mid_r4_32": {
            "model": fit_eta(mod_c, 4, 32),
            "reference": fit_eta(ref_c, 4, 32),
        },
        "long_r16_48": {
            "model": fit_eta(mod_c, 16, 48),
            "reference": fit_eta(ref_c, 16, 48),
        },
    }

    metric_keys = ("energy_mean", "abs_magnetization_mean", "binder_u4", "xi_over_l")
    metric_rows: list[dict] = []
    for ensemble, point, boot in (
        ("MC reference", ref_metrics, ref_boot_metrics),
        ("Model pooled", mod_metrics, mod_boot_metrics),
    ):
        for key in metric_keys:
            low, high = confidence_interval(np.asarray(boot[key]))
            metric_rows.append(
                {
                    "ensemble": ensemble,
                    "metric": key,
                    "estimate": point[key],
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(tables / "final_metrics.csv", metric_rows)
    write_csv(
        tables / "seed_metrics.csv",
        [{"seed": name} | metrics for name, metrics in zip(seed_names, seed_metrics, strict=True)],
    )
    write_csv(
        tables / "correlation.csv",
        [
            {
                "r": int(r),
                "region": "trained-scale" if r <= 32 else "extrapolated-scale",
                "model": float(mod_c[r]),
                "reference": float(ref_c[r]),
                "model_to_reference": float(correlation_ratio[r]),
            }
            for r in radius
        ],
    )

    positive = float(np.mean(model_arrays["magnetization"] > 0))
    checks = {
        "spin_flip_symmetry": abs(mod_metrics["magnetization_mean"]) < 0.10
        and 0.35 <= positive <= 0.65,
        "energy_abs_error_lt_0.03": abs(mod_metrics["energy_mean"] - ref_metrics["energy_mean"]) < 0.03,
        "abs_m_abs_error_lt_0.06": abs(mod_metrics["abs_magnetization_mean"] - ref_metrics["abs_magnetization_mean"]) < 0.06,
        "binder_abs_error_lt_0.05": abs(mod_metrics["binder_u4"] - ref_metrics["binder_u4"]) < 0.05,
        "xi_over_l_abs_error_lt_0.12": abs(mod_metrics["xi_over_l"] - ref_metrics["xi_over_l"]) < 0.12,
        "correlation_nrmse_r1_32_lt_0.08": nrmse_seen < 0.08,
        "correlation_nrmse_r33_64_lt_0.12": nrmse_extrapolated < 0.12,
    }
    l64 = json.loads(args.l64_summary.read_text(encoding="utf-8"))
    summary = {
        "question": "Can a denoiser trained only on L=64 extrapolate zero-shot to L=128?",
        "configuration": {
            "training_lattice_size": 64,
            "evaluation_lattice_size": 128,
            "checkpoint_frozen": True,
            "model_samples": int(len(model)),
            "sampling_seeds": seed_names,
            "samples_per_seed": [int(len(rows)) for rows in model_by_seed],
            "reference_samples": int(len(reference)),
            "reference_chains": int(len(np.unique(chain_ids))),
            "sampler": "ancestral",
            "reverse_steps": 128,
            "sampling_logit_temperature": 1.0,
        },
        "model": mod_metrics,
        "reference": ref_metrics,
        "absolute_error": {key: abs(mod_metrics[key] - ref_metrics[key]) for key in metric_keys},
        "mode_balance": {"positive_fraction": positive, "negative_fraction": 1.0 - positive},
        "correlation": {
            "nrmse_r1_32": nrmse_seen,
            "nrmse_r33_64": nrmse_extrapolated,
            "ratios": {f"r{r}": float(correlation_ratio[r]) for r in (8, 16, 32, 48, 64)},
            "scaling_fits": scaling,
        },
        "checks": checks,
        "all_zero_shot_checks_passed": bool(all(checks.values())),
        "reference_metadata": {
            "seed": reference_metadata.get("seed"),
            "convergence": reference_metadata.get("convergence"),
        },
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    apply_style()
    display_rng = np.random.default_rng(1287301)
    figure, axes = plt.subplots(2, 6, figsize=(14.5, 5.2))
    for row, (samples, label) in enumerate(((model, "L=128 model"), (reference, "L=128 MC"))):
        chosen = display_rng.choice(len(samples), size=6, replace=False)
        for column, index in enumerate(chosen):
            axes[row, column].imshow(
                samples[index], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
            )
            axes[row, column].set_title(f"{label}\nm={samples[index].mean():+.2f}", fontsize=8.5)
            axes[row, column].axis("off")
    save_figure(figure, figures, "01_sample_comparison")

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))
    histogram_specs = (
        ("energy", "energy density", 50),
        ("magnetization", "signed magnetization", 54),
        ("abs_magnetization", "absolute magnetization", 48),
    )
    for axis, (key, label, bins) in zip(axes, histogram_specs, strict=True):
        axis.hist(reference_arrays[key], bins=bins, density=True, histtype="step", color=BLACK, lw=2.2, label="MC reference")
        axis.hist(model_arrays[key], bins=bins, density=True, color=BLUE, alpha=0.32, edgecolor=BLUE, lw=0.8, label="zero-shot model")
        axis.set(xlabel=label, ylabel="density")
    axes[0].legend(fontsize=9)
    figure.suptitle("L=128 ensemble distributions", y=1.02, fontsize=13)
    save_figure(figure, figures, "02_ensemble_distributions")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    r = radius[1:]
    ref_low, ref_high = np.quantile(ref_boot_corr[:, 1:], (0.025, 0.975), axis=0)
    mod_low, mod_high = np.quantile(mod_boot_corr[:, 1:], (0.025, 0.975), axis=0)
    axes[0, 0].fill_between(r, ref_low, ref_high, color=PALE_GREY, alpha=0.75)
    axes[0, 0].fill_between(r, mod_low, mod_high, color=PALE_BLUE, alpha=0.75)
    axes[0, 0].loglog(r, ref_c[1:], color=BLACK, lw=2.4, label="L=128 MC")
    axes[0, 0].loglog(r, mod_c[1:], color=BLUE, lw=2.4, label="zero-shot model")
    axes[0, 0].loglog(r, ref_c[4] * (r / 4.0) ** (-0.25), "--", color=TEAL, lw=1.7, label=r"$r^{-1/4}$ guide")
    axes[0, 0].axvline(32, color=RED, ls=":", lw=1.7)
    axes[0, 0].set(xlabel="distance r", ylabel="C(r)", title="A  Two-point correlation")
    axes[0, 0].legend(fontsize=8)

    paired_ratio = mod_boot_corr[:, 1:] / ref_boot_corr[:, 1:]
    ratio_low, ratio_high = np.quantile(paired_ratio, (0.025, 0.975), axis=0)
    axes[0, 1].fill_between(r, ratio_low, ratio_high, color=PALE_BLUE, alpha=0.8)
    axes[0, 1].plot(r, correlation_ratio[1:], color=BLUE, lw=2.4)
    axes[0, 1].axhline(1.0, color=BLACK, ls="--", lw=1.5)
    axes[0, 1].axvline(32, color=RED, ls=":", lw=1.7, label="beyond L=64 half-width")
    axes[0, 1].axvspan(33, 64, color="#F6CFCB", alpha=0.22)
    axes[0, 1].set(xlabel="distance r", ylabel=r"$C_{model}/C_{MC}$", title="B  Long-range fidelity", xlim=(1, 64))
    axes[0, 1].legend(fontsize=8)

    valid_k = radius > 0
    ref_sf_low, ref_sf_high = np.quantile(ref_boot_sf[:, valid_k], (0.025, 0.975), axis=0)
    mod_sf_low, mod_sf_high = np.quantile(mod_boot_sf[:, valid_k], (0.025, 0.975), axis=0)
    axes[1, 0].fill_between(wave_number[valid_k], ref_sf_low, ref_sf_high, color=PALE_GREY, alpha=0.75)
    axes[1, 0].fill_between(wave_number[valid_k], mod_sf_low, mod_sf_high, color=PALE_BLUE, alpha=0.75)
    axes[1, 0].loglog(wave_number[valid_k], ref_s[valid_k], color=BLACK, lw=2.4, label="L=128 MC")
    axes[1, 0].loglog(wave_number[valid_k], mod_s[valid_k], color=BLUE, lw=2.4, label="zero-shot model")
    axes[1, 0].set(xlabel=r"wave number $|k|$", ylabel=r"$S(k)$", title="C  Structure factor")
    axes[1, 0].legend(fontsize=8)

    sf_ratio_boot = mod_boot_sf[:, valid_k] / ref_boot_sf[:, valid_k]
    sf_low, sf_high = np.quantile(sf_ratio_boot, (0.025, 0.975), axis=0)
    axes[1, 1].fill_between(wave_number[valid_k], sf_low, sf_high, color=PALE_BLUE, alpha=0.8)
    axes[1, 1].semilogx(wave_number[valid_k], structure_ratio, color=BLUE, lw=2.4)
    axes[1, 1].axhline(1.0, color=BLACK, ls="--", lw=1.5)
    axes[1, 1].set(xlabel=r"wave number $|k|$", ylabel=r"$S_{model}/S_{MC}$", title="D  Spectral fidelity")
    save_figure(figure, figures, "03_correlation_and_structure")

    labels = {
        "energy_mean": "energy",
        "abs_magnetization_mean": r"$\langle|m|\rangle$",
        "binder_u4": r"Binder $U_4$",
        "xi_over_l": r"$\xi_2/L$",
    }
    figure, axes = plt.subplots(1, 4, figsize=(15.7, 3.9))
    for axis, key in zip(axes, metric_keys, strict=True):
        points = (ref_metrics[key], mod_metrics[key])
        intervals = (
            confidence_interval(np.asarray(ref_boot_metrics[key])),
            confidence_interval(np.asarray(mod_boot_metrics[key])),
        )
        errors = np.asarray(
            [[points[i] - intervals[i][0] for i in range(2)], [intervals[i][1] - points[i] for i in range(2)]]
        )
        axis.errorbar((0, 1), points, yerr=errors, fmt="o", color=BLACK, ecolor=GREY, capsize=4, ms=7)
        axis.scatter((1,), (points[1],), color=BLUE, s=55, zorder=3)
        axis.set(xticks=(0, 1), xticklabels=("MC", "model"), ylabel=labels[key])
        margin = max(abs(points[1] - points[0]) * 2.2, max(errors.flatten()) * 1.8, 1e-3)
        center = 0.5 * sum(points)
        axis.set_ylim(center - margin, center + margin)
    figure.suptitle("L=128 observables with 95% bootstrap intervals", y=1.02, fontsize=13)
    save_figure(figure, figures, "04_scalar_observables")

    figure, axes = plt.subplots(1, 4, figsize=(15.7, 3.9))
    for axis, key in zip(axes, metric_keys, strict=True):
        for index, metrics in enumerate(seed_metrics):
            axis.scatter(index, metrics[key], color=BLUE_LIGHT, s=55, zorder=3)
        axis.axhline(ref_metrics[key], color=BLACK, ls="--", lw=1.8, label="MC reference")
        axis.scatter(3.3, mod_metrics[key], color=BLUE, marker="D", s=55, label="pooled model")
        axis.set(xticks=(0, 1, 2, 3.3), xticklabels=(*seed_names, "pooled"), ylabel=labels[key])
        axis.tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    figure.suptitle("L=128 sampling-seed stability", y=1.02, fontsize=13)
    save_figure(figure, figures, "05_seed_stability")

    sizes = np.asarray([64, 128])
    comparisons = (
        ("abs_magnetization_mean", r"$\langle|m|\rangle L^{1/8}$", True),
        ("binder_u4", r"Binder $U_4$", False),
        ("xi_over_l", r"$\xi_2/L$", False),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for axis, (key, ylabel, rescale) in zip(axes, comparisons, strict=True):
        mc = np.asarray([l64["reference"][key], ref_metrics[key]])
        generated = np.asarray([l64["model"][key], mod_metrics[key]])
        if rescale:
            mc = mc * sizes ** (1.0 / 8.0)
            generated = generated * sizes ** (1.0 / 8.0)
        axis.plot(sizes, mc, "o--", color=BLACK, lw=2.0, label="MC")
        axis.plot(sizes, generated, "o-", color=BLUE, lw=2.4, label="model")
        axis.set(xticks=sizes, xlabel="lattice size L", ylabel=ylabel)
    axes[0].legend(fontsize=9)
    figure.suptitle("Finite-size scaling: trained L=64 and zero-shot L=128", y=1.02, fontsize=13)
    save_figure(figure, figures, "06_cross_scale_scaling")

    status = "PASS" if summary["all_zero_shot_checks_passed"] else "PARTIAL"
    report = f"""# L=64 → L=128 零样本尺度外推报告

## 实验问题

模型仅在 L=64 的临界 Ising 样本上训练。本实验冻结 `best.pt`，不进行任何 L=128 微调，直接在 128×128 网格上采样，并与独立 Wolff Monte Carlo 参考系综比较。

## 配置

- 模型样本：{len(model):,} 张；sampling seeds：{', '.join(seed_names)}。
- MC 参考：{len(reference):,} 张，{len(np.unique(chain_ids))} 条独立链。
- 采样器：ancestral，128 个反向步，logit temperature=1.0。
- 总体预设检查：**{status}**。

## 核心结果

- 能量：model {mod_metrics['energy_mean']:.6f}，MC {ref_metrics['energy_mean']:.6f}。
- 平均绝对磁化：model {mod_metrics['abs_magnetization_mean']:.6f}，MC {ref_metrics['abs_magnetization_mean']:.6f}。
- Binder U4：model {mod_metrics['binder_u4']:.6f}，MC {ref_metrics['binder_u4']:.6f}。
- xi/L：model {mod_metrics['xi_over_l']:.6f}，MC {ref_metrics['xi_over_l']:.6f}。
- G(r) NRMSE：r=1..32 为 {nrmse_seen:.3%}；真正外推区间 r=33..64 为 {nrmse_extrapolated:.3%}。
- G_model/G_MC（r=8/16/32/48/64）：{' / '.join(f'{correlation_ratio[r]:.4f}' for r in (8, 16, 32, 48, 64))}。
- 有效幂律指数 eta（r=4..32）：model {scaling['short_to_mid_r4_32']['model']['eta']:.4f}，MC {scaling['short_to_mid_r4_32']['reference']['eta']:.4f}。
- 正/负磁化模式比例：{positive:.3f}/{1-positive:.3f}。

## 判据

""" + "\n".join(f"- {'PASS' if passed else 'FAIL'}：`{name}`" for name, passed in checks.items()) + """

## 图表阅读顺序

1. `01_sample_comparison`：先看肉眼形态是否具有跨尺度团簇，而不是逐像素匹配。
2. `02_ensemble_distributions`：比较局域能量、正负磁化对称性和磁化幅度分布。
3. `03_correlation_and_structure`：核心证据；红色虚线右侧 r>32 是训练尺度未覆盖的长距离区间。
4. `04_scalar_observables`：比较热力学量与无量纲临界指标及其 95% 区间。
5. `05_seed_stability`：检查结论是否依赖某个随机种子。
6. `06_cross_scale_scaling`：检查 L=64→128 时缩放后的磁化、Binder U4、xi/L 是否沿 MC 趋势变化。

## 解释边界

通过本实验只能支持“同一网络在两倍线性尺寸上保持临界统计”的结论，不能单凭 L=64、128 两个点精确测定普适临界指数。若结果通过，下一阶段应加入 L=96、192、256，并拟合带有限尺寸修正项的 scaling law。
"""
    (args.output_dir / "FINAL_REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
