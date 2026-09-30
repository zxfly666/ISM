"""Frozen-checkpoint diagnostics: independent G, geometry probes, cavity KL, steps.

No optimizers, training, or writes to the source study. Diagnostic interventions
are not new physical data and cannot be scored as an improved production model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
import torch.nn.functional as F

from diagnostic_math import (uniform_axis_fft, cavity_distribution, conditional_stages,
                             joint_from_conditionals, divergence, kl_decomposition)
from ism_diffusion.geometry_study import atomic_json, file_hash, make_batch, corrupt_batch, array_hash, stream_seed
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion.ising import BETA_CRITICAL
from evaluate_study import atomic_npz, write_csv


def check_time(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Diagnostic time budget reached; saved units can resume")


def checkpoint(study, seed, arm):
    return study / "training" / f"s{seed}_{arm}" / "final.pt"


def rows_read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_status(out, phase, **extra):
    row = dict(phase=phase, updated=time.time(), pid=os.getpid(), **extra)
    atomic_json(out / "status.json", row)
    print(json.dumps(row), flush=True)


def read_shards(directory, limit=None):
    arrays = {k: [] for k in ("spins", "mc", "axis_x", "axis_y", "input_coordinates", "parent", "chain", "origin")}
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as z:
            for k in arrays:
                arrays[k].append(z[k])
        if limit is not None and sum(len(a) for a in arrays["spins"]) >= limit:
            break
    return {k: np.concatenate(v)[:limit] for k, v in arrays.items()}


def g_metrics(spins, mc, stride, width):
    gs, counts = uniform_axis_fft(spins)
    refs, _ = uniform_axis_fft(mc)
    g, ref = gs.mean(0), refs.mean(0)
    r = np.arange(width)*stride
    bands = {"short": (1, 32), "medium": (33, 128), "long": (129, r[-1])} if stride > 1 else {
        "short": (1, 8), "medium": (9, 24), "long": (25, width//2)}
    errors = {}
    for key, (lo, hi) in bands.items():
        select = (r >= lo) & (r <= hi)
        errors[key] = float(np.linalg.norm(g[select]-ref[select])/np.linalg.norm(ref[select]))
    return dict(nrmse=errors, model_m2=float(np.mean(spins.mean((1, 2))**2)),
                mc_m2=float(np.mean(mc.mean((1, 2))**2))), dict(r=r, model_G=g, mc_G=ref,
                model_per_sample_G=gs, mc_per_sample_G=refs, pair_count_per_sample=counts)


def audit(study, out, models):
    dest = out / "audit"
    dest.mkdir(exist_ok=True)
    rows = []
    for seed, arm in models:
        for geom, stride in (("held_s10_w48", 10), ("continuous96", 1)):
            old = study / "evaluation" / f"s{seed}_{arm}" / "generation" / geom
            a = read_shards(old)
            if not np.isin(a["spins"], [-1, 1]).all():
                raise AssertionError("Invalid spin in original samples")
            for key in ("axis_x", "axis_y"):
                if not np.all(np.diff(a[key], axis=1) == stride):
                    raise AssertionError("uniform-grid audit received nonuniform coordinates")
            gs, counts = uniform_axis_fft(a["spins"])
            ms, _ = uniform_axis_fft(a["mc"])
            r = np.arange(gs.shape[1])*stride
            with np.load(old / "statistics.npz") as z:
                delta = float(np.max(np.abs(gs.mean(0)-z["model_G"][r])))
                mc_delta = float(np.max(np.abs(ms.mean(0)-z["mc_G"][r])))
            if max(delta, mc_delta) > 1e-10:
                raise AssertionError(f"G mismatch {seed} {arm} {geom}: {delta} {mc_delta}")
            row = dict(seed=seed, arm=arm, geometry=geom, samples=len(gs), max_model_error=delta,
                       max_mc_error=mc_delta, spin_values_ok=bool(np.isin(a["spins"], [-1, 1]).all()))
            if stride == 10:
                residual = gs.mean(0)-ms.mean(0)
                for peak in (190, 380):
                    j = peak//10
                    row[f"residual_curvature_r{peak}"] = float(residual[j]-.5*(residual[j-1]+residual[j+1]))
            atomic_npz(dest / f"s{seed}_{arm}_{geom}.npz", r=r, model_per_sample_G=gs,
                       mc_per_sample_G=ms, pair_count=counts, chain=a["chain"], parent=a["parent"])
            rows.append(row)
    # Heterogeneous audit rows are saved as JSON rather than an irregular CSV.
    atomic_json(dest / "complete.json", dict(status="passed", records=rows,
        interpretation="Exact FFT/slicing parity checks calculation, not MC precision or physical correctness."))


def single_visible_reference(parent, out, smoke):
    path = out / "single_visible_reference.npz"
    if path.exists():
        with np.load(path) as z:
            return z["G_per_parent"]
    count = 16 if smoke else 128
    origins = np.random.default_rng(922187).integers(parent.lattice_size, size=(len(parent.spins), count, 2))
    ids = np.arange(len(parent.spins))[:, None]
    xx, yy = origins[..., 0], origins[..., 1]
    spin = parent.spins[ids, xx, yy].astype(float)
    moments = np.empty((len(parent.spins), 2, 47))
    for axis in (0, 1):
        for j in range(1, 48):
            other = parent.spins[ids, (xx+10*j*(axis == 0)) % parent.lattice_size,
                                 (yy+10*j*(axis == 1)) % parent.lattice_size]
            moments[:, axis, j-1] = (spin*other).mean(1)
    atomic_npz(path, G_per_parent=moments, origins=origins, chain=parent.chain_ids,
               r=np.arange(1, 48)*10)
    return moments


@torch.inference_mode()
def probes_one(model, parent, study, dest, seed, arm, moments, smoke, deadline):
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        return
    torch.set_float32_matmul_precision("highest")
    old = study / "evaluation" / f"s{seed}_{arm}" / "generation/held_s10_w48"
    a = read_shards(old, 16 if smoke else 128)
    base_coords = a["input_coordinates"][0]
    # Fixed full W48 valid set. A query at [0,0] sees one spin at physical axial r.
    # The MC reference is an independent spin-flip-symmetrized two-point estimate.
    rows = []
    for scale in (.9, 1., 1.1):
        cases = [(axis, j, sign) for axis in (0, 1) for j in range(1, 48) for sign in (-1, 1)]
        for start in range(0, len(cases), 4):
            check_time(deadline)
            chunk = cases[start:start+4]
            tokens = np.full((len(chunk), 48, 48), 2, dtype=np.int64)
            for i, (axis, j, sign) in enumerate(chunk):
                tokens[i, j if axis == 0 else 0, j if axis == 1 else 0] = (sign+1)//2
            coords = np.repeat(base_coords[None], len(chunk), axis=0)*scale
            logits = model(torch.tensor(tokens, device="cuda"),
                torch.full((len(chunk),), 1-1/2304, device="cuda"), torch.tensor(coords, device="cuda"))
            pp = logits.float().softmax(1)[:, 1, 0, 0].cpu().numpy()
            for case, pred in zip(chunk, pp):
                axis, j, sign = case
                p = .5*(1+sign*moments[:, axis, j-1].mean())
                q = np.clip(pred, 1e-7, 1-1e-7)
                kl = p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
                rows.append(dict(axis=axis, physical_r=10*j, supplied_coordinate_scale=scale,
                    visible_sign=sign, reference_p=float(p), model_p=float(pred), kl=float(kl)))
    write_csv(dest / "single_visible.csv", rows)
    ce_rows = []
    clean = (a["mc"] > 0).astype(np.int64)
    for tv in (.2, .5, .8, .95):
        t, mask, noisy = corrupt_batch(clean, 922233, int(tv*100), tv)
        for scale in (.9, 1., 1.1):
            for start in range(0, len(clean), 4):
                check_time(deadline)
                sl = slice(start, start+4)
                x = torch.tensor(noisy[sl], device="cuda")
                logits = model(x, torch.tensor(t[sl], device="cuda"),
                               torch.tensor(a["input_coordinates"][sl]*scale, device="cuda"))
                target = torch.tensor(clean[sl], device="cuda")
                ce = F.cross_entropy(logits.float(), target, reduction="none").cpu().numpy()
                pp = logits.float().softmax(1)[:, 1].cpu().numpy()
                for k in range(len(ce)):
                    j = start+k
                    ce_rows.append(dict(sample=j, chain=int(a["chain"][j]), parent=int(a["parent"][j]),
                        t=tv, coordinate_scale=scale, ce=float(ce[k][mask[j]].mean()),
                        brier=float(((pp[k]-clean[j])**2)[mask[j]].mean()), masked=int(mask[j].sum())))
    write_csv(dest / "teacher_forced.csv", ce_rows)
    atomic_json(dest / "complete.json", dict(seed=seed, arm=arm, samples=len(clean),
        checkpoint_sha256=file_hash(checkpoint(study, seed, arm)), clean_hash=array_hash(clean),
        single_visible_mean_kl={str(s): float(np.mean([r["kl"] for r in rows if r["supplied_coordinate_scale"] == s])) for s in (.9, 1., 1.1)},
        teacher_forced_mean_ce={str(s): float(np.mean([r["ce"] for r in ce_rows if r["coordinate_scale"] == s])) for s in (.9, 1., 1.1)},
        warnings=["Scale 0.9/1.1 is wrong-coordinate intervention, not matched new physical sampling.",
                  "Single-visible is a sparse-evidence diagnostic; teacher-forced labels are genuine MC, never unrelated free-trajectory labels.",
                  "C uses one prespecified false geometry for the single-visible response curve."]))


@torch.inference_mode()
def cavity_one(model, parent, dest, seed, arm, smoke, deadline):
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        return
    torch.set_float32_matmul_precision("highest")
    ncase = 1 if smoke else 8
    batch = make_batch(parent, ncase, 16, "continuous", (1,), 922417, 0)
    # All exterior values, including the complete nearest-neighbour boundary,
    # are genuine MC values and stay fixed throughout the conditional experiment.
    atomic_npz(dest / "cases.npz", clean=batch["clean"], coords=batch["coords"][arm],
               parent=batch["parent"], chain=batch["chain"], origin=batch["origin"])
    records = []
    for side in (2, 3):
        sites = [(6+i, 6+j) for i in range(side) for j in range(side)]
        n = len(sites)
        plans = {"sequential_row": [[i] for i in range(n)],
                 "sequential_reverse": [[i] for i in range(n-1, -1, -1)],
                 "parallel_rows": [list(range(i*side, (i+1)*side)) for i in range(side)],
                 "parallel_all": [list(range(n))]}
        for case in range(ncase):
            base = batch["clean"][case]
            bits, reference, _ = cavity_distribution(2*base-1, sites, BETA_CRITICAL)
            for plan_name, plan in plans.items():
                check_time(deadline)
                saved = dest / f"side{side}_case{case}_{plan_name}.npz"
                metric_path = saved.with_suffix(".json")
                if metric_path.exists() and saved.exists():
                    records.append(json.loads(metric_path.read_text()))
                    continue
                stages = conditional_stages(bits, reference, plan)
                oracle = joint_from_conditionals(bits, stages, [s["probability"] for s in stages])
                learned = []
                for stage in stages:
                    pp = []
                    for start in range(0, len(stage["assignments"]), 64):
                        check_time(deadline)
                        assignment = stage["assignments"][start:start+64]
                        tokens = np.repeat(base[None], len(assignment), axis=0)
                        for x, y in sites:
                            tokens[:, x, y] = 2
                        for k, index in enumerate(stage["prior"]):
                            x, y = sites[index]
                            tokens[:, x, y] = assignment[:, k]
                        tv = max((n-len(stage["prior"]))/256, .01)
                        coords = np.repeat(batch["coords"][arm][case:case+1], len(tokens), axis=0)
                        logits = model(torch.tensor(tokens, device="cuda"),
                            torch.full((len(tokens),), tv, device="cuda"), torch.tensor(coords, device="cuda"))
                        q = logits.float().softmax(1)[:, 1]
                        pp.append(torch.stack([q[:, sites[i][0], sites[i][1]] for i in stage["group"]], 1).cpu().numpy())
                    learned.append(np.concatenate(pp))
                q, parts = kl_decomposition(bits, reference, stages, learned, oracle)
                if plan_name.startswith("sequential") and divergence(reference, oracle)["kl"] > 1e-10:
                    raise AssertionError("Sequential exact oracle must reproduce the joint")
                row = dict(seed=seed, arm=arm, side=side, case=case, plan=plan_name,
                    chain=int(batch["chain"][case]), parent=int(batch["parent"][case]),
                    **parts, learned_tv=divergence(reference, q)["tv"], oracle_tv=divergence(reference, oracle)["tv"])
                atomic_npz(saved, states=bits, reference=reference, oracle_joint=oracle, learned_joint=q)
                atomic_json(metric_path, row)
                records.append(row)
    write_csv(dest / "cavity_metrics.csv", records)
    atomic_json(dest / "complete.json", dict(seed=seed, arm=arm, cases=ncase, records=len(records),
        warnings=["Fixed plans: these joint KLs are not random-order-mixture KLs.",
                  "t=max(remaining_cavity/256,0.01); this diagnostic clock is not a replay of the production cos-squared schedule.",
                  "Boundary configurations are MC-weighted examples, not independent training repeats.",
                  "No sparse-grid nearest-neighbour Ising oracle is assumed."]))


@torch.inference_mode()
def sample_one(model, study, dest, seed, arm, geometry, steps, nsample, deadline):
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        return
    width, stride = (48, 10) if geometry == "held_s10_w48" else (96, 1)
    old = study / "evaluation" / f"s{seed}_{arm}" / "generation" / geometry
    a = read_shards(old, nsample)
    if steps == 256:
        spins = a["spins"]
        elapsed = 0.
        # No copy of original large raw shards; provenance identifies exact prefix.
    else:
        diffusion = CoordinateAbsorbingDiffusion()
        torch.set_float32_matmul_precision("high")
        tag = int.from_bytes(hashlib.sha256(geometry.encode()).digest()[:3], "little")
        shards, elapsed, nfe = [], 0., []
        for start in range(0, len(a["spins"]), 16):
            check_time(deadline)
            path = dest / f"shard_{start:05d}.npz"
            sl = slice(start, start+16)
            if not path.exists():
                coord = torch.tensor(a["input_coordinates"][sl], device="cuda")
                valid = torch.ones(coord.shape[:-1], dtype=torch.bool, device="cuda")
                generator = torch.Generator(device="cuda").manual_seed(stream_seed(seed, tag+start, "generation"))
                torch.cuda.synchronize()
                began = time.perf_counter()
                counter = [0]
                handle = model.register_forward_hook(lambda *_: counter.__setitem__(0, counter[0]+1))
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens = diffusion.sample(model, coord, valid, steps=steps, temperature=1., generator=generator)
                finally:
                    handle.remove()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-began
                if not bool(((tokens == 0) | (tokens == 1)).all()):
                    raise AssertionError("Unresolved generated tokens")
                generated = (2*tokens.cpu().numpy()-1).astype(np.int8)
                atomic_npz(path, spins=generated, seconds=np.array(seconds), nfe=np.array(counter[0]),
                    input_coordinates=a["input_coordinates"][sl], mc=a["mc"][sl],
                    parent=a["parent"][sl], chain=a["chain"][sl], origin=a["origin"][sl])
            with np.load(path) as z:
                shards.append(z["spins"])
                elapsed += float(z["seconds"])
                nfe.append(int(z["nfe"]))
            print(json.dumps(dict(sampling=dest.name, seed=seed, arm=arm, steps=steps,
                                  complete_samples=start+len(shards[-1]), seconds=elapsed)), flush=True)
        spins = np.concatenate(shards)
    metrics, arrays = g_metrics(spins, a["mc"], stride, width)
    atomic_npz(dest / "statistics.npz", **arrays, chain=a["chain"], parent=a["parent"])
    atomic_json(dest / "complete.json", dict(seed=seed, arm=arm, geometry=geometry, steps=steps,
        samples=len(spins), sampling_seconds=elapsed, nfe_per_shard=None if steps == 256 else nfe, old_source=str(old),
        checkpoint_sha256=file_hash(checkpoint(study, seed, arm)), **metrics,
        warnings=["Same geometry and seed tags, but different step counts do not yield identical trajectories.",
                  "256-step baseline is the exact same-sized prefix of the original run."]))


def aggregate(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"A": "#1764a1", "B": "#ce7028", "C": "#268578"}
    samples = [json.loads(p.read_text()) for p in (out / "sampling").glob("s*/*/steps*/complete.json")]
    if samples:
        write_csv(out / "sampling_summary.csv", [dict(seed=r["seed"], arm=r["arm"], geometry=r["geometry"],
            steps=r["steps"], samples=r["samples"], short=r["nrmse"]["short"], medium=r["nrmse"]["medium"],
            long=r["nrmse"]["long"], m2=r["model_m2"], mc_m2=r["mc_m2"]) for r in samples])
        for geom in ("held_s10_w48", "continuous96"):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            for ax, steps in zip(axes, (256, 512, 1024)):
                ref = None
                for arm in "ABC":
                    paths = sorted((out / "sampling").glob(f"s*_{arm}/{geom}/steps{steps}/statistics.npz"))
                    if not paths:
                        continue
                    curves = []
                    for path in paths:
                        with np.load(path) as z:
                            curves.append(z["model_G"]); ref=z["mc_G"]; r=z["r"]
                    curves=np.stack(curves)
                    select=(r > 0) & ((r <= 48) if geom == "continuous96" else True)
                    ax.plot(r[select], curves.mean(0)[select], color=colors[arm], label=arm)
                    ax.fill_between(r[select], curves.min(0)[select], curves.max(0)[select], color=colors[arm], alpha=.12)
                if ref is not None:
                    ax.plot(r[select], ref[select], "k--", label="MC")
                ax.set(title=f"{steps} steps", xlabel="Physical axial distance r", ylabel="G(r)")
                ax.legend(fontsize=8)
            fig.suptitle(geom+"; shading = seed range, not CI")
            fig.tight_layout(); fig.savefig(figures / f"{geom}_step_comparison.png", dpi=160); plt.close(fig)
    cavity=[]
    for p in (out / "cavity").glob("s*/cavity_metrics.csv"):
        cavity.extend(rows_read(p))
    if cavity:
        write_csv(out / "cavity_summary.csv", cavity)
    probes=[]
    for path in (out / "probes").glob("s*/complete.json"):
        probes.append(json.loads(path.read_text()))
    if probes:
        atomic_json(out / "probe_summary.json", probes)
        fig, axes=plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, arm in zip(axes, "ABC"):
            for scale, style in ((.9, ":"), (1., "-"), (1.1, "--")):
                profiles=[]
                for path in (out / "probes").glob(f"s*_{arm}/single_visible.csv"):
                    rows=rows_read(path)
                    profiles.append([np.mean([float(v["model_p"]) for v in rows if float(v["supplied_coordinate_scale"]) == scale
                        and int(v["physical_r"]) == r and int(v["visible_sign"]) == 1]) for r in range(10, 471, 10)])
                if profiles:
                    ax.plot(range(10, 471, 10), np.mean(profiles, axis=0), style, label=f"coords x {scale}")
            with np.load(out / "single_visible_reference.npz") as z:
                ref=.5*(1+z["G_per_parent"].mean((0,1)))
            ax.plot(range(10, 471, 10), ref, "k--", label="MC reference")
            ax.set(title=arm, xlabel="Physical axial distance", ylabel="P(query + | visible +)")
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figures / "single_visible_distance_response.png", dpi=160); plt.close(fig)
    return dict(probe_models=len(probes), cavity_models=len(list((out / "cavity").glob("s*/complete.json"))), sampling_cells=len(samples))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "artifacts/geometry_alignment_20260921")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/existing_diagnostics_20260922")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hours", type=float, default=4.)
    parser.add_argument("--phase", choices=("all", "audit", "probes", "cavity", "sampling", "aggregate"), default="all")
    args=parser.parse_args()
    out, study=args.out.resolve(), args.study.resolve()
    if out == study or study in out.parents:
        raise ValueError("diagnostics must not be nested inside the immutable source study")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock=(out / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seeds=[91001] if args.smoke else list(range(91001,91007))
    arms="A" if args.smoke else "ABC"
    models=[(s,a) for s in seeds for a in arms]
    prior=json.loads((study / "final_summary.json").read_text())
    reference=Path(prior["reference"]["path"])
    config=dict(version="existing_diagnostics_v1", smoke=args.smoke, models=models,
        source_study=str(study), source_protocol=prior["protocol_hash"], reference_sha256=file_hash(reference),
        checkpoint_sha256={f"s{s}_{a}":file_hash(checkpoint(study,s,a)) for s,a in models},
        source_sha256={p.name:file_hash(p) for p in (Path(__file__), Path(__file__).with_name("diagnostic_math.py"))},
        coordinate_scales=[.9,1.,1.1], single_visible_distances=list(range(10,471,10)),
        teacher_forced_samples=16 if args.smoke else 128, cavity_cases=1 if args.smoke else 8,
        cavity_sides=[2,3], primary_generation=dict(geometry="held_s10_w48", steps=[256,512,1024],samples=16 if args.smoke else 128),
        secondary_generation=dict(geometry="continuous96", seeds=[91001,91002],steps=[256,512,1024],samples=32),
        warning="Exploratory diagnostics on already-inspected tasks; not a new blinded confirmatory test.")
    config["hash"]=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=out / "frozen_diagnostic_protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["hash"] != config["hash"]:
        raise RuntimeError("Frozen diagnostic protocol differs; use a new output directory")
    atomic_json(manifest,config)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    began=time.time(); deadline=began+args.hours*3600
    try:
        if args.phase in ("all","audit"):
            log_status(out,"audit");audit(study,out,models)
        if args.phase in ("all","probes","cavity"):
            parent=load_parent_split(reference,"test_target")
            moments=single_visible_reference(parent,out,args.smoke)
            for seed,arm in models:
                check_time(deadline)
                model,payload=load_scale_model(checkpoint(study,seed,arm),torch.device("cuda")); del payload
                if args.phase in ("all","probes"):
                    log_status(out,"probes",seed=seed,arm=arm)
                    probes_one(model,parent,study,out/"probes"/f"s{seed}_{arm}",seed,arm,moments,args.smoke,deadline)
                if args.phase in ("all","cavity"):
                    log_status(out,"cavity",seed=seed,arm=arm)
                    cavity_one(model,parent,out/"cavity"/f"s{seed}_{arm}",seed,arm,args.smoke,deadline)
                del model;torch.cuda.empty_cache()
            del parent
        if args.phase in ("all","sampling"):
            for seed,arm in models:
                check_time(deadline)
                model,payload=load_scale_model(checkpoint(study,seed,arm),torch.device("cuda"));del payload
                geometries=[("held_s10_w48",16 if args.smoke else 128)]
                if not args.smoke and seed in (91001,91002):
                    geometries.append(("continuous96",32))
                for geom, count in geometries:
                    for steps in (256,512,1024):
                        log_status(out,"sampling",seed=seed,arm=arm,geometry=geom,steps=steps)
                        sample_one(model,study,out/"sampling"/f"s{seed}_{arm}"/geom/f"steps{steps}",seed,arm,geom,steps,count,deadline)
                del model;torch.cuda.empty_cache()
        totals=aggregate(out)
        after={f"s{s}_{a}":file_hash(checkpoint(study,s,a)) for s,a in models}
        if after != config["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint modified during diagnostics")
        expected_sampling=len(models)*3+(0 if args.smoke else 18)
        complete=(out/"audit/complete.json").exists() and totals==dict(probe_models=len(models),cavity_models=len(models),sampling_cells=expected_sampling)
        final=dict(status="complete" if complete else "partial",elapsed_hours=(time.time()-began)/3600,
            **totals,expected_models=len(models),expected_sampling_cells=expected_sampling,protocol_hash=config["hash"],
            checkpoint_unchanged=True,local_backup="not_yet_verified",peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(out/"final_summary.json",final)
        log_status(out,final["status"],**{k:v for k,v in final.items() if k != "status"})
    except Exception as error:
        log_status(out,"budget_stopped" if isinstance(error,TimeoutError) else "failed",error=repr(error))
        raise


if __name__ == "__main__":
    main()
