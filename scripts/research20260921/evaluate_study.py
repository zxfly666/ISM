"""Resumable paired evaluations; preserves raw samples and MC identities."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from ism_diffusion.geometry_study import (
    ARMS, TEST_DEFS, atomic_json, balanced_risk, coordinate_arrays,
    evaluate, file_hash, make_batch, rng, stream_seed,
)
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model, open_energy_density
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion.ising import BETA_CRITICAL


GEN_DEFS = [dict(name="continuous48", width=48, kind="continuous", gaps=[1], samples=128),
            dict(name="continuous64", width=64, kind="continuous", gaps=[1], samples=256),
            dict(name="held_gap57_w32", width=32, kind="gap", gaps=[5, 7], samples=256),
            dict(name="held_s10_w48", width=48, kind="gap", gaps=[10], samples=128)]


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(path.with_suffix(".csv.tmp"), path)


def atomic_npz(path, **arrays):
    path = Path(path)
    with path.with_name(path.name + ".tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(path.with_name(path.name + ".tmp"), path)


def physical_axis_statistics(spins, axes):
    """Open, x/y-axis pair-weighted G at physical integer separations.

    This is intentionally NOT the earlier radial-displacement average. Save
    per-sample numerators/denominators so clustering and reweighting are possible.
    """
    width = spins.shape[-1]
    max_r = int(max(axes[0].max(), axes[1].max()))
    sums = np.zeros((len(spins), max_r + 1), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    u, v = np.triu_indices(width, 1)
    for row, ss in enumerate(spins.astype(np.float64)):
        for axis in (0, 1):
            dist = axes[axis][row, v] - axes[axis][row, u]
            product = ((ss[u, :] * ss[v, :]).sum(1) if axis == 0
                       else (ss[:, u] * ss[:, v]).sum(0))
            sums[row] += np.bincount(dist, weights=product, minlength=max_r + 1)
            counts[row] += np.bincount(dist, minlength=max_r + 1) * width
        sums[row, 0] = width**2
        counts[row, 0] = width**2
    m = spins.mean((1, 2), dtype=np.float64)
    return dict(pair_sum=sums, pair_count=counts, m=m, m2=m*m, abs_m=np.abs(m))


def merge_statistics(shards):
    keys = ("pair_sum", "pair_count", "m", "m2", "abs_m")
    max_len = max(s["pair_sum"].shape[1] for s in shards)
    out = {}
    for key in keys:
        arrays = [s[key] for s in shards]
        if key.startswith("pair_"):
            arrays = [np.pad(a, ((0, 0), (0, max_len-a.shape[1]))) for a in arrays]
        out[key] = np.concatenate(arrays)
    return out


def correlation_summary(stats, reference, width, kind):
    size = max(stats["pair_sum"].shape[1], reference["pair_sum"].shape[1])
    def curve(source):
        s = np.pad(source["pair_sum"].sum(0), (0, size-source["pair_sum"].shape[1]))
        c = np.pad(source["pair_count"].sum(0), (0, size-source["pair_count"].shape[1]))
        return np.divide(s, c, out=np.full(size, np.nan), where=c > 0), c
    generated, count = curve(stats)
    mc, mc_count = curve(reference)
    r = np.arange(size)
    valid = (count > 0) & (mc_count > 0) & (r > 0)
    ranges = {"short_1_8": (1, 8), "medium_9_24": (9, 24), "context_25_plus": (25, width//2)} if kind == "continuous" else {
        "physical_1_32": (1, 32), "physical_33_128": (33, 128), "physical_129_plus": (129, size-1)}
    errors = {}
    for name, (low, high) in ranges.items():
        selected = valid & (r >= low) & (r <= high)
        if selected.any():
            delta = generated[selected] - mc[selected]
            errors[name] = dict(n_distances=int(selected.sum()), rmse=float(np.sqrt(np.mean(delta**2))),
                nrmse=float(np.sqrt(np.mean(delta**2))/max(np.sqrt(np.mean(mc[selected]**2)), 1e-12)))
    return dict(definition="open physical-axis G; all available x/y pairs equally weighted within each distance",
                errors=errors, model_m=float(stats["m"].mean()), mc_m=float(reference["m"].mean()),
                model_m2=float(stats["m2"].mean()), mc_m2=float(reference["m2"].mean()),
                model_abs_m=float(stats["abs_m"].mean()), mc_abs_m=float(reference["abs_m"].mean())), (r, generated, mc)


@torch.inference_mode()
def geometry_probe(model, parent, arm, out):
    """Two visible spins in a real W16 grid; all other sites remain MASK.

    Matched moments are estimated from the SAME MC triples and then spin-flip
    symmetrized. Four visible-value patterns are weighted by their event mass.
    Six additional grids keep span and gap histogram fixed while permuting gaps.
    """
    width, center = 16, 7
    layouts = []
    for gap in (1, 2, 4, 5, 7, 10):
        layouts.append((f"uniform{gap}", np.full(15, gap), np.full(15, gap)))
    base_x = np.resize(np.array([1, 2, 4, 8]), 15)
    base_y = np.resize(np.array([8, 4, 2, 1]), 15)
    for j in (7, 0, 1, 2, 4, 6):
        gx, gy = base_x.copy(), base_y.copy()
        gx[7], gx[j] = gx[j], gx[7]
        layouts.append((f"same_summary_swap7_{j}", gx, gy))
    ids = np.repeat(np.arange(len(parent.spins)), 256)
    origin = rng(777012, 0, "probe_sites").integers(parent.lattice_size, size=(len(ids), 2))
    qspin = parent.spins[ids, origin[:, 0], origin[:, 1]].astype(float)
    rows, per_parent = [], []
    for li, (name, gx, gy) in enumerate(layouts):
        ox, oy = np.r_[0, gx.cumsum()], np.r_[0, gy.cumsum()]
        a, b = int(gx[7]), int(gy[7])
        js = parent.spins[ids, (origin[:, 0]+a) % parent.lattice_size, origin[:, 1]].astype(float)
        ks = parent.spins[ids, origin[:, 0], (origin[:, 1]+b) % parent.lattice_size].astype(float)
        moments = np.stack([qspin*js, qspin*ks, js*ks], 1).reshape(len(parent.spins), 256, 3).mean(1)
        per_parent.append(moments)
        gqj, gqk, gjk = moments.mean(0)
        cc = np.stack(np.meshgrid(ox, oy, indexing="ij"), -1).astype(np.float32)
        cc -= cc[center, center].copy()
        if arm == "B":
            rank = np.arange(width)-center
            cc = np.stack(np.meshgrid(rank, rank, indexing="ij"), -1).astype(np.float32)
        elif arm == "C":
            cc = coordinate_arrays(1, width, "continuous" if name == "uniform1" else "gap", (1, 2, 4, 8), 12059, li)[0]["C"][0]
            cc -= cc[center, center].copy()
        for bv, cv in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            mass = (1+bv*cv*gjk)/4
            if mass <= 1e-10:
                continue
            truth = .5*(1+(bv*gqj+cv*gqk)/(1+bv*cv*gjk))
            tokens = torch.full((1, width, width), 2, device="cuda", dtype=torch.long)
            tokens[:, center+1, center] = (bv+1)//2
            tokens[:, center, center+1] = (cv+1)//2
            logits = model(tokens, torch.tensor([1-2/width**2], device="cuda"),
                           torch.tensor(cc[None], device="cuda"))
            pred = float(logits.float().softmax(1)[0, 1, center, center])
            pred = float(np.clip(pred, 1e-7, 1-1e-7))
            pp = float(np.clip(truth, 1e-12, 1-1e-12))
            kl = pp*np.log(pp/pred)+(1-pp)*np.log((1-pp)/(1-pred))
            rows.append(dict(layout=name, gap_x=a, gap_y=b, visible_j=bv, visible_k=cv,
                             event_mass=float(mass), reference_p=float(truth), model_p=pred, kl=float(kl)))
    write_csv(out / "geometry_probe.csv", rows)
    atomic_npz(out / "geometry_reference_moments.npz", moments=np.stack(per_parent), chain=parent.chain_ids)
    return dict(weighted_kl=float(sum(r["event_mass"]*r["kl"] for r in rows)/len(layouts)),
                layouts=len(layouts), context="W16, two visible spins, time matched to mask fraction",
                reference="zero-field symmetrized triple moments from the named finite MC parent pool")


@torch.inference_mode()
def natural_markov(model, parent, arm, out):
    records = []
    for width in (16, 64):
        batch = make_batch(parent, 32, width, "continuous", (1,), 901011, width, False)
        c = width//2
        clean = batch["clean"]
        nn_sum = sum(2*clean[:, c+dx, c+dy]-1 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        p = 1/(1+np.exp(-2*BETA_CRITICAL*nn_sum))
        for t in (.2, .5):
            mask = rng(985, width, f"mask{t}").random(clean.shape) < t
            mask[:, c, c] = True
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                mask[:, c+dx, c+dy] = False
            for start in range(0, 32, 4):
                sl = slice(start, start+4)
                logits = model(torch.tensor(np.where(mask[sl], 2, clean[sl]), device="cuda"),
                    torch.full((4,), t, device="cuda"), torch.tensor(batch["coords"][arm][sl], device="cuda"))
                qs = logits.float().softmax(1)[:, 1, c, c].cpu().numpy()
                for j, q in enumerate(qs):
                    truth = p[start+j]
                    q = np.clip(q, 1e-7, 1-1e-7)
                    kl = truth*np.log(truth/q)+(1-truth)*np.log((1-truth)/(1-q))
                    records.append(dict(width=width, t=t, sample=start+j, reference_p=float(truth),
                        model_p=float(q), kl=float(kl), chain=int(batch["chain"][start+j]), parent=int(batch["parent"][start+j])))
    write_csv(out / "natural_markov.csv", records)
    return dict(mean_kl=float(np.mean([r["kl"] for r in records])),
                masking="iid mask conditioned on center hidden and four neighbors visible; value blind")


def generate_and_score(model, parent, arm, seed, definition, output, deadline, shard_size=16):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    width = definition["width"]
    geometry_tag = int.from_bytes(__import__("hashlib").sha256(definition["name"].encode()).digest()[:3], "little")
    diffusion = CoordinateAbsorbingDiffusion()
    model_shards, mc_shards, exemplar_model, exemplar_mc = [], [], [], []
    all_spins, all_mc = [], []
    elapsed = 0.
    for start in range(0, definition["samples"], shard_size):
        if time.time() >= deadline:
            raise TimeoutError("evaluation budget reached; completed shards remain resumable")
        path = output / f"shard_{start:05d}.npz"
        batch = make_batch(parent, min(shard_size, definition["samples"]-start), width,
            definition["kind"], definition["gaps"], 918073, geometry_tag+start, False)
        if not path.exists():
            coord = torch.tensor(batch["coords"][arm], device="cuda")
            valid = torch.ones(batch["clean"].shape, device="cuda", dtype=torch.bool)
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = diffusion.sample(model, coord, valid, steps=256, temperature=1.,
                    generator=torch.Generator(device="cuda").manual_seed(stream_seed(seed, geometry_tag+start, "generation")))
            torch.cuda.synchronize()
            duration = time.perf_counter()-began
            if not bool(((tokens == 0) | (tokens == 1)).all()):
                raise RuntimeError("unresolved MASK/invalid spin after generation")
            spins = (2*tokens.cpu().numpy()-1).astype(np.int8)
            atomic_npz(path, spins=spins, mc=(2*batch["clean"]-1).astype(np.int8),
                axis_x=batch["axes"][0], axis_y=batch["axes"][1], input_coordinates=batch["coords"][arm],
                parent=batch["parent"], chain=batch["chain"], origin=batch["origin"],
                seconds=np.array(duration), seed=np.array(seed), sample_start=np.array(start))
        with np.load(path) as z:
            spins, mc = z["spins"], z["mc"]
            axes = [z["axis_x"], z["axis_y"]]
            elapsed += float(z["seconds"])
            model_shards.append(physical_axis_statistics(spins, axes))
            mc_shards.append(physical_axis_statistics(mc, axes))
            if len(exemplar_model) < 4:
                exemplar_model.extend(list(spins[:4-len(exemplar_model)]))
                exemplar_mc.extend(list(mc[:4-len(exemplar_mc)]))
            if definition["kind"] == "continuous":
                all_spins.append(spins)
                all_mc.append(mc)
    stats, reference = merge_statistics(model_shards), merge_statistics(mc_shards)
    summary, (r, generated, mc) = correlation_summary(stats, reference, width, definition["kind"])
    summary.update(definition=definition, arm=arm, seed=seed, sampling_seconds=elapsed,
                   sampler="S0_cos_squared_256_temperature1_no_MC_correction", reference_split="test_target")
    if all_spins:
        summary.update(model_energy_per_bond=float(open_energy_density(np.concatenate(all_spins)).mean()),
                       mc_energy_per_bond=float(open_energy_density(np.concatenate(all_mc)).mean()))
    atomic_npz(output / "statistics.npz", **{f"model_{k}": v for k, v in stats.items()},
        **{f"mc_{k}": v for k, v in reference.items()}, r=r, model_G=generated, mc_G=mc,
        exemplar_model=np.stack(exemplar_model), exemplar_mc=np.stack(exemplar_mc))
    atomic_json(output / "complete.json", summary)
    return summary


def evaluate_one(model_path, reference_path, output, arm, seed, config, deadline):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    model, _ = load_scale_model(model_path, torch.device("cuda"))
    parent = load_parent_split(reference_path, "test_target")
    meta = dict(checkpoint_sha256=file_hash(model_path), reference_sha256=file_hash(reference_path),
                arm=arm, seed=seed, protocol_hash=config["protocol_hash"])
    atomic_json(output / "provenance.json", meta)
    if not (output / "conditional.csv").exists():
        all_records = []
        for i, definition in enumerate(TEST_DEFS):
            if time.time() >= deadline:
                raise TimeoutError("conditional evaluation deadline")
            primary = i < 2
            all_records.extend(evaluate(model, parent, arm, [definition],
                samples=config.get("test_samples", 512) if primary else 128,
                seed=730000+i*1000, masks=2 if primary else 1, microbatch=8))
        write_csv(output / "conditional.csv", all_records)
    else:
        with (output / "conditional.csv").open() as f:
            all_records = [dict(r, ce=float(r["ce"]), t=float(r["t"]), chain=int(r["chain"]), parent=int(r["parent"])) for r in csv.DictReader(f)]
    primary_rows = [r for r in all_records if r["geometry"].startswith("held_")]
    conditional = balanced_risk(primary_rows)
    torch.set_float32_matmul_precision("highest")
    diagnostics = dict(geometry=geometry_probe(model, parent, arm, output),
                       markov=natural_markov(model, parent, arm, output))
    torch.set_float32_matmul_precision("high")
    definitions = list(GEN_DEFS)
    if config.get("include_w96"):
        definitions.append(dict(name="continuous96", width=96, kind="continuous", gaps=[1], samples=128))
    if config.get("smoke"):
        definitions = [dict(GEN_DEFS[0], samples=4)]
    generations = {}
    for definition in definitions:
        generations[definition["name"]] = generate_and_score(model, parent, arm, seed, definition,
            output / "generation" / definition["name"], deadline, shard_size=4 if config.get("smoke") else 16)
    result = dict(**meta, conditional=conditional, diagnostics=diagnostics, generation=generations)
    atomic_json(output / "complete.json", result)
    del model, parent
    torch.cuda.empty_cache()
    return result


def aggregate(study, config, allow_partial=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "savefig.dpi": 180})
    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    results = []
    for seed in config["seeds"]:
        for arm in ARMS:
            path = study / "evaluation" / f"s{seed}_{arm}" / "complete.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
    full = len(results) == 3*len(config["seeds"])
    if not full and not allow_partial:
        raise RuntimeError("Cannot declare completion with missing experimental cells")
    colors = {"A": "#1764a1", "B": "#ce7028", "C": "#268578"}
    names = {"A": "Matched", "B": "Rank / unit", "C": "Independent geometry"}
    rows, effects = [], []
    for seed in config["seeds"]:
        selected = {r["arm"]: r for r in results if r["seed"] == seed}
        if set(selected) != set(ARMS):
            continue
        for arm in ARMS:
            rows.append(dict(seed=seed, arm=arm, primary_masked_CE=selected[arm]["conditional"]["mean_ce"]))
        effects.append(dict(seed=seed, A_minus_B=selected["A"]["conditional"]["mean_ce"]-selected["B"]["conditional"]["mean_ce"],
                            A_minus_C=selected["A"]["conditional"]["mean_ce"]-selected["C"]["conditional"]["mean_ce"]))
    if rows:
        write_csv(study / "primary_metrics.csv", rows)
        write_csv(study / "paired_seed_effects.csv", effects)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, key in enumerate(("A_minus_B", "A_minus_C")):
            x = np.array([r[key] for r in effects])
            ax.scatter(x, np.arange(len(x))+.14*i, label=key.replace("_", " "), s=45)
        ax.axvline(0, color="0.5", ls="--")
        ax.set(xlabel="Paired masked-CE difference (nat); negative favors Matched", ylabel="Independent training seed")
        ax.legend()
        fig.tight_layout(); fig.savefig(figures / "paired_conditional_effects.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for arm in ARMS:
        for si, seed in enumerate(config["seeds"]):
            path = study / "training" / f"s{seed}_{arm}" / "train.jsonl"
            if not path.exists():
                continue
            points = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            points = [p for p in points if "validation" in p]
            if points:
                ax.plot([p["step"] for p in points], [p["validation"]["mean_ce"] for p in points],
                        color=colors[arm], alpha=.5, lw=1, label=names[arm] if si == 0 else None)
    ax.set(xlabel="Optimizer updates", ylabel="Validation masked CE (nat)", title="All training seeds; EMA evaluation")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "validation_loss.png"); plt.close(fig)
    definitions = list(GEN_DEFS)
    if config.get("include_w96"):
        definitions.append(dict(name="continuous96", width=96, kind="continuous"))
    for definition in definitions:
        name = definition["name"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        first_mc = None
        for arm in ARMS:
            curves, mcs = [], []
            for seed in config["seeds"]:
                path = study / "evaluation" / f"s{seed}_{arm}" / "generation" / name / "statistics.npz"
                if not path.exists():
                    continue
                with np.load(path) as z:
                    curves.append(z["model_G"]); mcs.append(z["mc_G"])
            if not curves:
                continue
            # Fixed geometry bank makes length/reference identical across seeds.
            values = np.stack(curves)
            valid = np.isfinite(values).all(0) & np.isfinite(mcs[0])
            r = np.arange(values.shape[1])
            valid &= r > 0
            if definition["kind"] == "continuous":
                valid &= r <= definition["width"]//2
            rv, mean, ref = r[valid], values[:, valid].mean(0), mcs[0][valid]
            axes[0].plot(rv, mean, color=colors[arm], label=names[arm])
            axes[0].fill_between(rv, values[:, valid].min(0), values[:, valid].max(0), color=colors[arm], alpha=.12)
            nonzero = np.abs(ref) > 1e-8
            axes[1].plot(rv[nonzero], mean[nonzero]/ref[nonzero], color=colors[arm])
            if first_mc is None:
                first_mc = (rv, ref)
        if first_mc:
            axes[0].plot(*first_mc, color="0.15", ls="--", label="Matched-geometry MC")
        axes[0].set(xlabel="Physical axial distance r", ylabel="G(r)", title=name+" (bands: seed min-max)")
        axes[0].legend(fontsize=8)
        axes[1].axhline(1, color="0.5", ls="--")
        axes[1].set(xlabel="Physical axial distance r", ylabel="Model / MC", title="Correlation ratio")
        fig.tight_layout(); fig.savefig(figures / (name+"_correlation.png")); plt.close(fig)
    seed = config["seeds"][0]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ri, arm in enumerate(("MC", "A", "B", "C")):
        path = study / "evaluation" / f"s{seed}_{'A' if arm == 'MC' else arm}" / "generation/continuous64/statistics.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            images = z["exemplar_mc" if arm == "MC" else "exemplar_model"]
            for ci, image in enumerate(images):
                axes[ri, ci].imshow(image, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
                if ci == 0:
                    axes[ri, ci].set_ylabel(arm if arm == "MC" else names[arm])
    fig.suptitle("W=64: predetermined first four samples (not selected for appearance)")
    fig.tight_layout(); fig.savefig(figures / "sample_comparison_w64.png"); plt.close(fig)
    summary = dict(status="complete" if full else "partial", protocol_hash=config["protocol_hash"],
        training_seeds=config["seeds"], completed_models=len(results), expected_models=3*len(config["seeds"]),
        seed_effects=effects, metrics=rows, reference=config.get("reference_selected"),
        warnings=["Training seeds are independent repeats; MC parents/crops/masks are not extra training repeats.",
                  "Correlation shading is seed min-max, not a confidence interval.",
                  "This G is physical-axis, open-window G; do not compare it numerically to earlier radial G without harmonization."])
    atomic_json(study / "final_summary.json", summary)
    lines = ["# 几何对齐研究：运行结果", "", f"状态：{summary['status']}；完成 {len(results)}/{summary['expected_models']} 个模型。",
             "", "主比较是同一数据、相同训练预算下的 Matched / Rank / Independent geometry。",
             "本报告只描述本轮测得效应，不把条件优势自动解释为联合生成优势。", "",
             "## 每个独立训练 seed 的配对 CE 差（负值有利于 Matched）", "",
             "| Seed | A−B (nat) | A−C (nat) |", "|---|---:|---:|"]
    lines.extend(f"| {r['seed']} | {r['A_minus_B']:.7f} | {r['A_minus_C']:.7f} |" for r in effects)
    lines += ["", "## 阅读图表", "", "- validation_loss.png：每条线代表一个独立训练，纵轴是 masked CE，不是精确 NLL。",
              "- paired_conditional_effects.png：逐训练 seed 的 A−B / A−C，零线左侧有利于真实几何。",
              "- *_correlation.png：按真实轴向距离的原始 G 与同几何 MC；阴影是 seed 极值范围，不是 CI。",
              "- sample_comparison_w64.png：固定前四个样本，不是挑选外观最好的图片。",
              "", "## 限制", "", "条件风险、几何诊断和完整生成分开解释。",
              "独立 MC 链和训练 seed 有限；本轮不自动给出充分功效的显著性、创新性或发表保证。",
              "更完整的人工结果解读需结合 evaluation/ 下的诊断与 generation 统计。"]
    (study / "RESULTS_ZH.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary
