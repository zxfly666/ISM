"""Post-hoc, CPU-only display of the frozen G experiment's W96 correlations.

No training, generation, checkpoint selection, or new hypothesis test.
Once the small sufficient-statistics package is present, run this file with no
arguments. To rebuild that package from verified local archives, supply
--from-backup PATH. Original scientific inputs are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ARMS = ("G00", "G10", "G01", "G11", "G11S")
SEEDS = tuple(range(92501, 92507))
COLORS = ("#767676", "#B64342", "#42949E", "#0F4D92", "#9A4D8E")
STYLES = ("--", "-.", ":", "-", (0, (5, 2, 1, 2)))
ROOT = "artifacts/core_geometry_factorial_20260925/"
RANGES = ((1, 8), (9, 24), (25, 48))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_out(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_verified(backup, archive_name, wanted, source_manifest):
    """Check the archive and selected members against both historical manifests."""
    archive = backup / archive_name
    manifest_file = backup / (archive_name + ".manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    with archive.open("rb") as f:
        archive_hash = hashlib.file_digest(f, "sha256").hexdigest()
    assert archive.stat().st_size == manifest["bytes"]
    assert archive_hash == manifest["sha256"]
    entries = {x["path"]: x for x in manifest["files"]}
    formal = json.loads((EXPERIMENT / "evidence/manifest.json").read_text(encoding="utf-8"))
    result = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for name in wanted:
            data = bundle.extractfile(name).read()
            item = entries[name]
            assert len(data) == item["bytes"] and sha(data) == item["sha256"]
            assert formal[name.removeprefix(ROOT)] == {"bytes": len(data), "sha256": sha(data)}
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                result[name] = {k: z[k].copy() for k in z.files}
            source_manifest["members"].append(dict(item, archive=archive_name))
    source_manifest["archives"].append({
        "name": archive_name, "bytes": archive.stat().st_size, "sha256": archive_hash,
        "manifest_sha256": sha(manifest_file.read_bytes()),
    })
    return result


def make_package(backup, target):
    source = {"scope": "Post-hoc descriptive W96 curves, original experiment unchanged",
              "archives": [], "members": []}
    name = ROOT + "reference/generation_reference.npz"
    reference = read_verified(backup, "reference_complete_v1.tar.gz", [name], source)[name]
    model = {k: [] for k in ("pair_sum", "pair_count", "m", "m2", "abs_m", "energy")}
    for seed in SEEDS:
        per_seed = {k: [] for k in model}
        for arm in ARMS:
            prefix = ROOT + f"evaluation/s{seed}_{arm}/generation/"
            names = [prefix + "statistics.npz"] + [prefix + f"shard_{i:05d}.npz" for i in range(0, 64, 16)]
            data = read_verified(backup, f"evaluation_s{seed}_{arm}_v1.tar.gz", names, source)
            stats = data[names[0]]
            for k in model:
                if k == "energy":
                    spins = np.concatenate([data[n]["spins"] for n in names[1:]]).astype(np.float64)
                    assert spins.shape == (64, 96, 96) and np.isin(spins, [-1, 1]).all()
                    bonds = (spins[:, 1:, :] * spins[:, :-1, :]).sum((1, 2))
                    bonds += (spins[:, :, 1:] * spins[:, :, :-1]).sum((1, 2))
                    value = -bonds / (2 * 96 * 95)
                else:
                    value = stats["model_" + k]
                per_seed[k].append(value)
            reconstructed = stats["model_pair_sum"].sum(0) / stats["model_pair_count"].sum(0)
            np.testing.assert_allclose(reconstructed, stats["model_G"], rtol=0, atol=1e-14)
        for k in model:
            model[k].append(per_seed[k])
    arrays = {"model_" + k: np.asarray(v) for k, v in model.items()}
    arrays.update({"mc_" + k: v for k, v in reference.items() if k != "origin"})
    arrays.update(arms=np.array(ARMS), seeds=np.array(SEEDS), r=np.arange(96))
    np.savez_compressed(target / "sufficient_statistics.npz", **arrays)
    source["package"] = {"name": "sufficient_statistics.npz",
                         "sha256": sha((target / "sufficient_statistics.npz").read_bytes()),
                         "note": "Scientific numeric subset only; no weights, images or private metadata"}
    json_out(target / "source_manifest.json", source)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(z, target):
    assert tuple(z["arms"]) == ARMS and tuple(z["seeds"]) == SEEDS
    assert z["model_pair_sum"].shape == (6, 5, 64, 96)
    assert z["mc_pair_sum"].shape == (1024, 96)
    assert np.array_equal(np.unique(z["mc_chain"], return_counts=True)[1], np.full(8, 128))
    expected_count = np.r_[96**2, 2 * 96 * (96 - np.arange(1, 96))]
    assert np.all(z["model_pair_count"] == expected_count)
    assert np.all(z["mc_pair_count"] == expected_count)
    np.testing.assert_allclose(z["model_energy"], -z["model_pair_sum"][..., 1] / z["model_pair_count"][..., 1], atol=0, rtol=0)
    np.testing.assert_allclose(z["mc_energy"], -z["mc_pair_sum"][..., 1] / z["mc_pair_count"][..., 1], atol=0, rtol=0)
    for v in z.values():
        if v.dtype.kind in "fiu":
            assert np.isfinite(v).all()
    curves = z["model_pair_sum"].sum(2) / z["model_pair_count"].sum(2)
    ref = z["mc_pair_sum"].sum(0) / z["mc_pair_count"].sum(0)
    mean = curves.mean(0)
    chain_curves = np.array([
        z["mc_pair_sum"][z["mc_chain"] == c].sum(0) /
        z["mc_pair_count"][z["mc_chain"] == c].sum(0)
        for c in np.unique(z["mc_chain"])
    ])
    point = np.empty((6, 5, 7))
    for k, (lo, hi) in enumerate(RANGES):
        sel = slice(lo, hi + 1)
        point[:, :, k] = np.sqrt(np.mean((curves[:, :, sel] - ref[sel])**2, axis=-1)) / np.sqrt(np.mean(ref[sel]**2))
    for k, field in enumerate(("m", "m2", "abs_m", "energy"), 3):
        point[:, :, k] = z["model_" + field].mean(2) - z["mc_" + field].mean()
    frozen = np.load(EXPERIMENT / "evidence/analysis/generation_point.npz", allow_pickle=False)["values"]
    np.testing.assert_allclose(point, frozen, atol=1e-12, rtol=0)
    assert np.all(curves[:, :, 0] == 1) and ref[0] == 1

    def eta(c, lo, hi):
        return float(-np.log(c[hi] / c[lo]) / np.log(hi / lo))

    summary = {
        "setting": "continuous W96, physical unit spacing, beta_critical, open x/y axis pairs",
        "formal_distance_ranges": RANGES, "full_saved_range": [0, 95],
        "status": "Post-hoc descriptive display, no new scientific samples or formal tests",
        "original_210_scores_reproduction_max_abs_error": float(np.max(np.abs(point - frozen))),
        "mc": {"G": {str(r): float(ref[r]) for r in (1, 8, 16, 24, 25, 32, 48, 64, 95)},
               "long_range_rms": float(np.sqrt(np.mean(ref[25:49]**2))),
               "eta_secant_8_24": eta(ref, 8, 24), "eta_secant_25_48": eta(ref, 25, 48),
               **{k: float(z["mc_" + k].mean()) for k in ("m", "m2", "abs_m", "energy")}},
        "arms": {},
        "slope_note": "Fixed-endpoint descriptive log slopes, NOT fitted critical exponents or inferential tests",
        "guide_note": "G_MC(8)*(r/8)^(-1/4); amplitude anchored only to MC; no model fitting",
        "uncertainty_note": "Main panels are point estimates; seed/MC-chain spread is separately displayed, not a CI",
    }
    for a, arm in enumerate(ARMS):
        residual = mean[a] - ref
        sel = slice(25, 49)
        agg_nrmse = np.sqrt(np.mean(residual[sel]**2)) / np.sqrt(np.mean(ref[sel]**2))
        summary["arms"][arm] = {
            "nrmse_mean_per_seed": point[:, a, :3].mean(0).tolist(),
            "nrmse_25_48_of_mean_curve": float(agg_nrmse),
            "mean_per_seed_rmse_25_48": float(point[:, a, 2].mean() * summary["mc"]["long_range_rms"]),
            "G": {str(r): float(mean[a, r]) for r in (1, 8, 16, 24, 25, 32, 48, 64, 95)},
            "signed_residual_25_48_min_max": [float(residual[sel].min()), float(residual[sel].max())],
            "relative_residual_r48": float(residual[48] / ref[48]),
            "eta_secant_8_24": eta(mean[a], 8, 24), "eta_secant_25_48": eta(mean[a], 25, 48),
            "bias_m_m2_abs_m_energy": point[:, a, 3:].mean(0).tolist(),
            "seed_nrmse_25_48": point[:, a, 2].tolist(),
        }
    json_out(target / "curve_summary.json", summary)
    rows = []
    for r in range(96):
        row = {"r": r, "MC": ref[r], "pairs_per_image": int(expected_count[r]),
               "MC_chain_min": chain_curves[:, r].min(), "MC_chain_max": chain_curves[:, r].max()}
        for a, arm in enumerate(ARMS):
            row.update({arm: mean[a, r], arm + "_residual": mean[a, r] - ref[r],
                        arm + "_relative_residual": (mean[a, r] - ref[r]) / ref[r]})
        rows.append(row)
    write_csv(target / "curves_by_distance.csv", list(rows[0]), rows)
    rows = [dict(seed=seed, arm=arm, r=r, G=curves[s, a, r], MC=ref[r],
                 signed_residual=curves[s, a, r] - ref[r])
            for s, seed in enumerate(SEEDS) for a, arm in enumerate(ARMS) for r in range(96)]
    write_csv(target / "curves_per_seed.csv", list(rows[0]), rows)
    return curves, mean, ref, chain_curves, summary


def apply_publication_style():
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Arial", "sans-serif"],
                        "font.size": 13, "axes.linewidth": 1.5, "axes.spines.top": False,
                        "axes.spines.right": False, "legend.frameon": False,
                        "pdf.fonttype": 42, "svg.fonttype": "none"})


def finalize_figure(fig, path, top=.87):
    fig.tight_layout(pad=1.4, rect=(0, 0, 1, top))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mark_ranges(ax, full=False):
    ax.axvspan(24.5, 48.5, color="#DCE8F3", alpha=.5, zorder=0)
    ax.axvline(8.5, color="#BBBBBB", lw=.8, ls=":")
    ax.axvline(24.5, color="#BBBBBB", lw=.8, ls=":")
    if full:
        ax.axvspan(48.5, 95, color="#EEEEEE", alpha=.7, zorder=0)
        ax.axvline(48.5, color="#BBBBBB", lw=.8, ls=":")


def plot_curves(curves, mean, ref, chain_curves, target):
    apply_publication_style()
    figure_dir = target / "figures"
    figure_dir.mkdir(exist_ok=True)
    r = np.arange(1, 96)
    guide = ref[8] * (r / 8) ** (-.25)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.5))
    for ax in axes:
        for a, arm in enumerate(ARMS):
            ax.plot(r, mean[a, 1:], color=COLORS[a], ls=STYLES[a], lw=2.8 if arm == "G11" else 1.9, label=arm)
        ax.plot(r, ref[1:], color="black", lw=2.6, label="MC reference", zorder=5)
        ax.set_xlabel("Physical separation $r$ (lattice units)")
        ax.set_ylabel("Raw axis-pair correlation $G(r)$")
        mark_ranges(ax, full=ax is axes[0])
    axes[0].set(xlim=(1, 95), ylim=(0, .78), title="(a) Continuous W96: full saved range 1-95")
    axes[0].set_xticks([1, 8, 24, 48, 64, 80, 95])
    axes[0].text(65, .08, "$r>48$: descriptive only", fontsize=10, color="#555555")
    axes[1].plot(r, guide, color="#A07018", lw=1.7, ls=(0, (2, 2)), label="$r^{-1/4}$ guide (MC at $r=8$)")
    axes[1].set(xscale="log", yscale="log", xlim=(1, 48), ylim=(.24, .82), title="(b) Scored range: log-log view")
    axes[1].set_xticks([1, 2, 4, 8, 16, 24, 48])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set_yticks([.25, .3, .4, .5, .7])
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].minorticks_off()
    axes[1].text(.91, .90, "Scored tail\n25-48", fontsize=10, transform=axes[1].transAxes, ha="center")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.01), fontsize=12)
    finalize_figure(fig, figure_dir / "A_W96_correlations", top=.86)

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.3))
    for a, arm in enumerate(ARMS):
        residual = mean[a, 1:] - ref[1:]
        for ax, values in zip(axes, (residual, 100 * residual / ref[1:])):
            ax.plot(r, values, color=COLORS[a], ls=STYLES[a], lw=2.8 if arm == "G11" else 1.9, label=arm)
    for ax in axes:
        ax.axhline(0, color="black", lw=1.3)
        ax.set_xlabel("Physical separation $r$ (lattice units)")
        mark_ranges(ax, full=ax is axes[0])
    axes[0].set(xlim=(1, 95), ylabel="$G_{\\mathrm{gen}}(r)-G_{\\mathrm{MC}}(r)$", title="(a) Continuous W96: signed error")
    axes[0].set_xticks([1, 8, 24, 48, 64, 80, 95])
    axes[1].set(xlim=(1, 48), ylim=(-15, 130), ylabel="Pointwise relative error (%)", title="(b) Relative error: scored range 1-48")
    axes[1].set_xticks([1, 8, 16, 24, 32, 40, 48])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(.5, 1.01))
    finalize_figure(fig, figure_dir / "B_W96_signed_errors", top=.88)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True, sharey=True)
    for a, ax in enumerate(axes.flat):
        if a == 5:
            ax.set_axis_off()
            handles = [Line2D([], [], color="black", lw=2.5, label="MC pooled mean"),
                       Line2D([], [], color="#999999", lw=8, alpha=.22, label="MC chain min-max (8 chains)"),
                       Line2D([], [], color=COLORS[3], lw=2.5, label="Arm mean (6 training seeds)"),
                       Line2D([], [], color=COLORS[3], lw=.9, alpha=.5, label="Individual training seed")]
            ax.legend(handles=handles, loc="upper left", fontsize=12)
            ax.text(.02, .37, "One setting: continuous W96\n64 images / seed / arm\nMC: 8 chains × 128 crops\n\nSpread is descriptive, not a CI.\nNo seed or checkpoint is selected.",
                    transform=ax.transAxes, fontsize=12, va="top", linespacing=1.7)
            continue
        ax.fill_between(r[:48], chain_curves[:, 1:49].min(0), chain_curves[:, 1:49].max(0), color="#999999", alpha=.18)
        for s in range(6):
            ax.plot(r[:48], curves[s, a, 1:49], color=COLORS[a], lw=.85, alpha=.45)
        ax.plot(r[:48], mean[a, 1:49], color=COLORS[a], lw=2.5)
        ax.plot(r[:48], ref[1:49], color="black", lw=2.2)
        mark_ranges(ax)
        ax.set(title=ARMS[a], xlim=(1, 48), ylim=(.2, .8), xlabel="Physical separation $r$", ylabel="$G(r)$")
        ax.set_xticks([1, 8, 24, 48])
        ax.tick_params(labelbottom=True, labelleft=True)
    finalize_figure(fig, figure_dir / "C_W96_seed_spread", top=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-backup", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if args.from_backup:
        make_package(args.from_backup, target)
    source = json.loads((target / "source_manifest.json").read_text(encoding="utf-8"))
    assert sha((target / "sufficient_statistics.npz").read_bytes()) == source["package"]["sha256"]
    with np.load(target / "sufficient_statistics.npz", allow_pickle=False) as stored:
        z = dict(stored)
    curves, mean, ref, chains, summary = summarize(z, target)
    plot_curves(curves, mean, ref, chains, target)
    outputs = [target / name for name in ("curve_summary.json", "curves_by_distance.csv", "curves_per_seed.csv", "sufficient_statistics.npz", "source_manifest.json")]
    outputs += sorted((target / "figures").glob("*"))
    receipt = {"status": "passed", "script_sha256": sha(Path(__file__).read_bytes()),
               "input_package_sha256": source["package"]["sha256"],
               "frozen_metric_sha256": sha((EXPERIMENT / "evidence/analysis/generation_point.npz").read_bytes()),
               "max_abs_error_all_210_frozen_metrics": summary["original_210_scores_reproduction_max_abs_error"],
               "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
               "files": [{"path": p.relative_to(target).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in outputs]}
    json_out(target / "reproduction_receipt.json", receipt)
    print(json.dumps({"status": "passed", "max_metric_error": receipt["max_abs_error_all_210_frozen_metrics"],
                      "mc": summary["mc"], "arms": summary["arms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
