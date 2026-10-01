"""Traceable six-figure bundle. No claim or gate is recomputed by the renderer."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import endpoint_common as c

COLORS = ["#767676", "#B64342", "#0F4D92"]
NAMES = ["Original support", "Lower time floor", "Explicit late-mask"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.5, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})


def coverage(root, fixture):
    cuts = np.array([1, 2, 8, 32])
    counts = np.zeros((4, 4)); total = np.zeros(4)
    ratios = np.zeros((4, 2))
    if fixture:
        return np.array([[.0001, .0004, .002, .01], [.0002, .0005, .003, .012], [.001, .003, .006, .016], [.04, .08, .16, .25]]), np.array([[.0001, .001], [.0001, .001], [.0003, .004], [.08, .2]])
    for path in (root/"coverage_audit").glob("s*.npz"):
        z = c.load(path); m = z["M"]; fraction = m/z["width"]**2
        counts[0] += (m[:, None] <= cuts).sum(0); total[0] += len(m)
        ratios[0] += np.array([(fraction <= .002).sum(), (fraction <= .01).sum()])
    for ai, arm in enumerate(c.ARMS):
        for seed in c.SEEDS:
            path = root/"training"/f"s{seed}_{arm}"/"log.jsonl"
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line); m = np.asarray(row["M"]); frac = m/row["width"]**2
                    counts[ai+1] += (m[:, None] <= cuts).sum(0); total[ai+1] += len(m)
                    ratios[ai+1] += np.array([(frac <= .002).sum(), (frac <= .01).sum()])
    assert np.all(total > 0)
    return counts/total[:, None], ratios/total[:, None]


def render(root, fixture=False):
    root = Path(root); out = root/"analysis"; style()
    summary = c.read(out/"summary.json")
    captions = {}
    records = []
    def export(fig, name, caption, sources):
        if fixture:
            fig.suptitle("SYNTHETIC SOFTWARE FIXTURE — NOT EXPERIMENT RESULTS", fontsize=13, color="#B64342")
        fig.tight_layout(pad=2, rect=(0, .07 if fig.legends else 0, 1, .91 if fixture else 1))
        for extension in ["png", "pdf"]:
            fig.savefig(out/f"{name}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)
        captions[name] = caption
        records.append(dict(figure=name, caption=caption, source_files=sources, plot_script="scripts/research20261001/endpoint_figures.py", fixture=fixture))
    rates, fraction = coverage(root, fixture)
    c.save(out/"coverage_plot_data.npz", cumulative_mask_rates=rates, realized_fraction_rates=fraction)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#272727", *COLORS]
    labels = ["Old I-F (sampled audit)", *NAMES]
    for i in range(4):
        axs[0].plot(range(4), rates[i], marker="o", color=colors[i], label=labels[i])
        axs[1].plot(range(2), fraction[i], marker="o", color=colors[i])
    axs[0].set(xticks=range(4), xticklabels=["M≤1", "M≤2", "M≤8", "M≤32"], ylabel="Fraction of training images", title="Realized MASK counts")
    axs[1].set(xticks=range(2), xticklabels=["M/N≤.002", "M/N≤.01"], ylabel="Fraction of training images", title="Realized fraction, not clock input")
    axs[0].legend(fontsize=9)
    export(fig, "01_mask_coverage", "Observed cumulative coverage. Historical I-F uses6144 reconstructed updates, not a full replay; new arms use all logs. Sparse-visible updates are included and are not late-mask examples.", ["coverage_audit/*.npz", "training/*/log.jsonl", "analysis/coverage_plot_data.npz"])
    z = c.load(out/"conditional_draws.npz")
    means = z["exact_per_parent"].mean(-1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ai in range(3):
        for panel in [0, 1]:
            y = means[:, ai, panel*4:panel*4+4]
            axs[panel].plot(range(4), y.mean(0), color=COLORS[ai], marker="o", label=NAMES[ai])
            for si in range(6):
                axs[panel].scatter(np.arange(4)+.025*(si-2.5), np.maximum(y[si], 1e-10), color=COLORS[ai], alpha=.35, s=12)
            axs[panel].set(yscale="log", xticks=range(4), xticklabels=[1, 2, 8, 32], xlabel="Remaining MASK count", ylabel="Exact local mean KL", title=f"W{48 if panel == 0 else 96}")
            axs[panel].axhline(.01, color="black", linestyle=":")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
    export(fig, "02_local_conditional_ability", "Exact nearest-neighbor conditional KL; dots are six inherited training lineages, lines are means. The.01 line is an accuracy threshold, not proof of correct joint generation; formal simultaneous upper bounds are in summary.json.", ["analysis/conditional_draws.npz", "analysis/summary.json"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, row) in enumerate(summary["primary"].items()):
        for off, key, color in [(-.07, "bootstrap_ci", "#0F4D92"), (.07, "paired_t_ci", "#B64342")]:
            lo, hi = row[key]; mean = row["estimate"]
            ax.plot([lo, hi], [i+off, i+off], color=color, linewidth=3)
            ax.plot(mean, i+off, "o", color=color)
        ax.scatter(row["per_seed"], np.full(6, i-.2), color="#767676", s=20)
    ax.axvline(0, color="black", linewidth=1); ax.axvline(-.05, color="black", linestyle=":")
    ax.set(yticks=[0, 1], yticklabels=["P1: late-mask − original (S256)", "P2: repair − monotone (late-mask)"], xlabel="Δ W96 long-range NRMSE (lower is better)")
    ax.plot([], [], color="#0F4D92", label="Joint bootstrap97.5% CI")
    ax.plot([], [], color="#B64342", label="Paired t97.5% CI")
    ax.legend(loc="best", fontsize=9)
    export(fig, "03_predefined_primary", "Two predeclared effects. Each requires both interval upper bounds below−.05 for practical success. Six gray points are lineages; secondary diagnostics do not replace these endpoints.", ["analysis/summary.json", "analysis/generation_primary_draws_block8.npz"])
    gp = c.load(out/"generation_point.npz")["per_seed"]
    gd = c.load(out/"generation_primary_draws_block8.npz")["metrics"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    labels = [f"{a}/{s}" for a in ["A", "L", "E"] for s in ["S", "R"]]
    for ax, metric, title in zip(axs.flat, [0, 1, 2, 3, 5, 6], ["Short-range NRMSE ↓", "Mid-range NRMSE ↓", "Long-range NRMSE ↓", "Signed m bias (target0)", "Signed |m| bias (target0)", "Signed energy/bond bias (target0)"]):
        values = gp[..., metric].mean(0).ravel()
        bounds = np.quantile(gd[..., metric], [.025, .975], axis=0).reshape(2, -1)
        for j in range(6):
            ax.plot([j, j], bounds[:, j], color=COLORS[j//2], linewidth=2)
            ax.scatter(j, values[j], color=COLORS[j//2], marker="s" if j % 2 else "o")
        ax.axhline(0, color="#767676", linewidth=.7)
        ax.set(xticks=range(6), xticklabels=labels, title=title)
    export(fig, "04_physical_tradeoffs", "W96 estimates and secondary95% joint intervals. S=monotone256, R=reveal192+repair64. Signed moments and energy target0, not uniformly lower values. Formal absolute-bias non-degradation gates are separate.", ["analysis/generation_point.npz", "analysis/generation_primary_draws_block8.npz", "analysis/summary.json"])
    rep = c.load(out/"repair_diagnostics.npz")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ai in range(3):
        axs[0].plot(np.arange(1, 65), rep["on_policy_local_kl"][:, ai].mean(0), color=COLORS[ai], label=NAMES[ai])
        y = rep["per_seed"][:, ai, :, 2]
        axs[1].plot(range(3), y.mean(0), color=COLORS[ai], marker="o", label=NAMES[ai])
    axs[0].set(xlabel="Repair network call (not a full sweep)", ylabel="Local KL on generated context", title="Model conditional error during repair")
    axs[1].set(xticks=range(3), xticklabels=["Prefix192", "Model repair", "Oracle repair"], ylabel="Long-range NRMSE", title="Paired prefix branches; oracle is diagnostic")
    axs[0].legend(fontsize=9)
    export(fig, "05_committed_repair_diagnostics", "Model and analytic Gibbs branches share the same prefix, position schedule and uniforms. Oracle is not a learned-model result or an upper bound. The prefix comparison spends additional compute and cannot replace equal256-call P2.", ["analysis/repair_diagnostics.npz", "evaluation/*/generation/w96_reveal192-repair64/shard_*.npz"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, view in zip(axs, ["majority32", "decimation32"]):
        zz = c.load(out/f"secondary_w96_{view}.npz")
        reference = zz["reference_mean"][:-4]; curves = zz["generated_mean"][..., :-4]
        r = np.arange(1, len(reference))
        ax.plot(r, reference[1:], color="black", linewidth=2, label="Same-transformed MC")
        for ai in range(3):
            ax.plot(r, curves[:, ai, 0].mean(0)[1:], color=COLORS[ai], label=NAMES[ai]+" / S")
        ax.plot(r, curves[:, 2, 1].mean(0)[1:], color=COLORS[2], linestyle="--", label="Late-mask / R")
        ax.set(xlabel="Coarse lag (nominal fine lag =3×)", ylabel="G(r)",
               title="3×3 majority: own MC target" if view=="majority32" else "Stride-3 decimation: own MC target")
    axs[0].legend(fontsize=8)
    export(fig, "06_true_coarse_graining", "All model/MC fields receive the same operator within each panel. Curves are means; inferential intervals are in secondary arrays. Majority is not decimation or a same-beta nearest-neighbor Ising law; this figure is not RG-training validation.", ["analysis/secondary_w96_majority32.npz", "analysis/secondary_w96_decimation32.npz", "analysis/secondary_w96_majority30_origins.npz"])
    c.write(out/"figure_sources.json", dict(figures=records, script_sha256=c.sha(Path(__file__)), fixture=fixture))
    lines = ["# Figure evidence and captions", "", "Synthetic fixtures are not scientific results." if fixture else "Predefined primary effects remain separate from all secondary diagnostics.", ""]
    for name, caption in captions.items():
        lines += [f"## {name}", "", caption, "", f"![{name}]({name}.png)", ""]
    path = out/"FIGURES.md"
    with path.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return records
