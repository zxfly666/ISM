"""J science stages, invoked only by the guarded single-run driver."""
from __future__ import annotations
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import ParentSplit, load_parent_split, pack_spins
import intervention_common as c
import intervention_data as d
import intervention_training as tr


def status(root, stage, **fields):
    value = dict(time=time.time(), stage=stage, **fields)
    c.write(Path(root)/"status.json", value, exclusive=False)
    print(json.dumps(value), flush=True)


def copy_blueprint(root):
    src = c.ROOT/"artifacts/geometry_identification_20260926/design/low_layouts.npz"
    if c.sha(src) != "dd82967bad896a785038892988b9a2fa6cfc9cd7327370cdfa570574cf5e1e55":
        raise RuntimeError("Frozen I low-K blueprint changed")
    dst = Path(root)/"design/low_layouts.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("xb") as f:
        f.write(src.read_bytes())
    c.write(Path(root)/"design/blueprint_source.json", dict(path=src.relative_to(c.ROOT).as_posix(), sha256=c.sha(src)))


def reference(root, check):
    from ism_diffusion.ising import generate_independent_chains, BETA_CRITICAL, energy_density, magnetization
    from ism_diffusion.diagnostics import integrated_autocorrelation_time, split_rhat
    from evaluate_study import physical_axis_statistics
    root = Path(root); out = root/"reference"; out.mkdir(exist_ok=False)
    cfg = c.CONFIG["mc"]
    started = time.time()
    seeds = [int(x.generate_state(1, dtype=np.uint32)[0]) for x in np.random.SeedSequence(c.CONFIG["seeds"]["mc"]).spawn(16)]
    args = dict(lattice_size=1024, chain_seeds=seeds, samples_per_chain=256,
        burn_in_sweeps=40, sweeps_between=4, beta=float(BETA_CRITICAL), adaptation_sweeps=3,
        pilot_cluster_steps=128, workers=4, backend="numba", initial_states=["random"]*8+["plus"]*4+["minus"]*4,
        return_chain_metadata=True)
    c.write(out/"protocol.json", dict(args, master_seed=c.CONFIG["seeds"]["mc"], role="evaluation_only_no_adaptive_extension"))
    spins, chain, metadata = generate_independent_chains(**args)
    check()
    if spins.shape != (4096, 1024, 1024) or not np.array_equal(chain, np.repeat(np.arange(16), 256)):
        raise RuntimeError("Wrong independent MC identities")
    md = dict(lattice_size=1024, beta=float(BETA_CRITICAL), chain_metadata=metadata,
              chain_seeds=seeds, seed=c.CONFIG["seeds"]["mc"], frames_per_chain=256)
    c.save(out/"fresh_l1024.npz", test_target_packed=pack_spins(spins), test_target_chain_id=chain, metadata=np.array(json.dumps(md)))
    parent = ParentSplit(spins, chain, 1024, md)
    # Chunk diagnostics: never promote all 4096 fields to float64 at once.
    traces = {key: [] for key in ("energy", "m", "abs_m", "m2", "G25_48")}
    crop_stats, origins = [], []
    for start in range(0, 4096, 16):
        check()
        ids = np.arange(start, start+16)
        a = np.broadcast_to(np.arange(96), (16, 2, 96)).copy()
        origin = np.stack([c.rng(c.CONFIG["seeds"]["evaluation"], int(i), "reference_QA_crop").integers(1024, size=2) for i in ids])
        crop = 2*d.sampled(parent, ids, a, origin)-1
        cs = physical_axis_statistics(crop, [a[:, 0], a[:, 1]])
        crop_stats.append(cs); origins.append(origin)
        m = magnetization(spins[ids])
        for key, value in dict(energy=energy_density(spins[ids]), m=m, abs_m=np.abs(m), m2=m*m,
                G25_48=(cs["pair_sum"][:, 25:49]/cs["pair_count"][:, 25:49]).mean(1)).items():
            traces[key].append(value)
    traces = {key: np.concatenate(value) for key, value in traces.items()}
    diag = {key: dict(split_rhat=float(split_rhat([value[chain == i] for i in range(16)])),
                     chains=[integrated_autocorrelation_time(value[chain == i]) for i in range(16)]) for key, value in traces.items()}
    passed = all(np.isfinite(diag[k]["split_rhat"]) and diag[k]["split_rhat"] <= 1.1 and
                 all(x["ess"] >= 16 for x in diag[k]["chains"]) for k in ("energy", "abs_m", "m2", "G25_48"))
    c.save(out/"chain_traces.npz", chain=chain, frame=np.tile(np.arange(256), 16), **traces)
    c.save(out/"qa_crop_statistics.npz", **{key: np.concatenate([p[key] for p in crop_stats]) for key in crop_stats[0]},
           origin=np.concatenate(origins), parent=np.arange(4096), chain=chain)
    c.write(out/"qa.json", dict(status="passed" if passed else "qa_failed", diagnostics=diag,
        parents=4096, chains=16, elapsed_seconds=time.time()-started, production_not_extended=True))
    if not passed:
        raise RuntimeError("Reference QA failed; no resampling until pass")
    low = d.low_reference(parent, c.load(root/"design/low_layouts.npz"), check)
    c.save(out/"low_joint_counts.npz", **low)
    c.write(out/"complete.json", dict(status="passed", parents=4096, chains=16, elapsed_seconds=time.time()-started,
        files={p.name: c.sha(p) for p in out.glob("*.npz")}, qa_sha256=c.sha(out/"qa.json")))


def save_validation(root, ident, step, model, banks, check):
    out = Path(root)/"training"/ident["cell"]/"validation"/f"step_{step}"
    out.mkdir(parents=True, exist_ok=False)
    for name, bank in banks.items():
        c.save(out/(name+".npz"), **tr.predict(model, bank, check=check))


def train(root, protocol, check):
    root = Path(root); started = time.time()
    parent = load_parent_split(c.DATA, "train")
    val = load_parent_split(c.DATA, "val")
    banks = d.validation_banks(val)
    for name, b in banks.items():
        d.validate_bank(b); c.save(root/"validation_banks"/(name+".npz"), **b)
    del val
    for ident, start, end in c.block_order():
        check()
        if (root/"reference_failure.json").exists():
            raise RuntimeError("Reference failure; no further training")
        out = root/"training"/ident["cell"]; out.mkdir(parents=True, exist_ok=True)
        if start == 0:
            base = c.base_checkpoint(ident["pair_index"]) if ident["cohort"] == "continuation" else None
            m, e, o, meta = tr.initialize(ident, "cuda", base)
            c.write(out/"initial.json", dict(identity=ident, metadata=meta))
            save_validation(root, ident, 0, e, banks, check)
        else:
            m, e, o, meta = tr.restore(out/"last.pt", ident, protocol["protocol_hash"], "cuda")
        if meta["step"] != start:
            raise RuntimeError("Wrong interleaved checkpoint step")
        status(root, "training", cell=ident["cell"], start=start, target=end)
        began = time.perf_counter()
        with (out/"train.jsonl").open("a", encoding="utf-8", buffering=65536) as f:
            for step in range(start+1, end+1):
                check()
                if step % 20 == 0 and (root/"reference_failure.json").exists():
                    raise RuntimeError("Reference failed during training")
                b = d.training_batch(parent, ident, step)
                row = tr.update(m, e, o, b, ident["cohort"], step)
                meta["step"] = step; meta["global_step"] = ident["base_step"]+step
                for dest, src in (("paired_data_digest", "paired_data_hash"), ("actual_input_digest", "actual_input_hash")):
                    meta[dest] = hashlib.sha256((meta[dest]+row[src]).encode()).hexdigest()
                f.write(json.dumps(dict(time=time.time(), step=step, global_step=meta["global_step"], **row))+"\n")
                if step % 20 == 0:
                    f.flush()
        meta["elapsed_seconds"] += time.perf_counter()-began
        tr.checkpoint(out/"last.pt", m, e, o, meta, protocol["protocol_hash"], immutable=False)
        spec = c.COHORTS[ident["cohort"]]
        if end in spec["validation_steps"]:
            save_validation(root, ident, end, e, banks, check)
        if end in spec["diagnostic_ema_steps"]:
            path = out/f"ema_{end}.pt"
            with path.open("xb") as f:
                torch.save(dict(ema=e.state_dict(), identity=ident, step=end, global_step=meta["global_step"],
                                protocol_hash=protocol["protocol_hash"], config=dict(model=c.MODEL)), f)
        if end == ident["updates"]:
            tr.checkpoint(out/"final.pt", m, e, o, meta, protocol["protocol_hash"])
            c.write(out/"complete.json", dict(status="complete", identity=ident, steps=end,
                global_step=meta["global_step"], metadata=meta, final_sha256=c.sha(out/"final.pt"),
                raw_hash=gs.model_hash(m), ema_hash=gs.model_hash(e)))
        c.write(out/"status.json", dict(step=end, identity=ident, time=time.time()), exclusive=False)
        del m, e, o; torch.cuda.empty_cache()
    for cohort in c.COHORTS:
        for index in range(6):
            rows = [c.read(root/"training"/c.identity(cohort, index, a)["cell"]/"complete.json") for a in ("D", "O")]
            for key in ("initial_raw_hash", "initial_ema_hash", "paired_data_digest", "actual_input_digest"):
                if rows[0]["metadata"][key] != rows[1]["metadata"][key]:
                    raise RuntimeError("Unpaired complete training: "+key)
    locked = {row["cell"]: c.sha(c.ROOT/row["checkpoint"]) for row in c.evaluation_identities()}
    c.write(root/"finals_locked.json", dict(time=time.time(), checkpoints=locked, trained=24,
                                           training_seconds=time.time()-started, all_finals_saved_before_any_formal_prediction=True))
    status(root, "training_complete", models=24)
    return time.time()-started


def make_banks(parent, root, check, *, scratch=False):
    root = Path(root)
    core_dir = root/"banks"; core_dir.mkdir(exist_ok=False)
    all_ids = np.arange(len(parent.spins))
    subset = all_ids if scratch else d.parent_subset(parent, np.arange(0, 256, 16))
    oracle_ids = all_ids if scratch else d.parent_subset(parent, [0, 64, 128, 192])
    build_timings = {}
    for gi, spec in enumerate(c.CONFIG["core_banks"]):
        began = time.perf_counter()
        check()
        ids = all_ids if spec["parents"] == 4096 else subset
        # The two continuous banks must share geometry/query/observation ordering.
        key = 2 if spec["kind"] == "continuous" else gi
        b = d.bank(parent, ids, c.CONFIG["seeds"]["evaluation"], key, spec["width"], spec["kind"], spec["k"])
        d.validate_bank(b)
        c.save(core_dir/(spec["name"]+".npz"), **b)
        p = (b["noisy"].reshape(len(ids), -1) == 1).sum(1)
        q = (p+.5)/(spec["k"]+1)
        y = b["labels"]
        risks = []
        for prob in (np.full(len(ids), .5), q):
            pc = prob[:, None]
            risks.append(np.stack([-(y*np.log(pc)+(1-y)*np.log1p(-pc)), (pc-y)**2], -1))
        c.save(core_dir/(spec["name"]+"_cpu_baselines.npz"), risks=np.array(risks), labels=y,
               parent=ids, chain=b["chain"], names=np.array(["uniform", "visible_mean_Jeffreys"]))
        build_timings[spec['name']] = time.perf_counter()-began
        del b
    for category in ('mechanism', 'padding', 'oracle'):
        began = time.perf_counter()
        if category == 'mechanism': banks = d.mechanism_banks(parent, subset)
        elif category == 'oracle': banks = d.oracle_banks(parent, oracle_ids)
        else:
            banks = {}
            for k in (32, 512):
                banks.update({f'k{k}_{name}': b for name,b in d.padding_banks(parent,subset,k).items()})
        for name, b in banks.items():
            check(); d.validate_bank(b)
            c.save(root/"diagnostic_banks"/category/(name+".npz"), **b)
        build_timings[category] = time.perf_counter()-began
        del banks
    began = time.perf_counter()
    low = d.low_inputs(c.load(root/"design/low_layouts.npz"))
    c.save(root/"diagnostic_banks/low/low_k.npz", **low)
    build_timings['low'] = time.perf_counter()-began
    c.write(core_dir/"complete.json", dict(status="complete", core_banks=4, mechanism=8, padding=6, oracle=9, low=1,
                                           software_scratch=scratch, build_timings=build_timings))


def evaluate_model(model, ident, root, check):
    root = Path(root); out = root/"evaluation"/ident["cell"]; out.mkdir(parents=True, exist_ok=False)
    categories = [("core", root/"banks")]
    if ident["kind"] != "intermediate":
        categories += [(x, root/"diagnostic_banks"/x) for x in ("mechanism", "padding", "oracle", "low")]
    count, input_count, inference_seconds, write_seconds = 0, 0, 0., 0.
    model_hash = gs.model_hash(model)
    for category, directory in categories:
        for path in sorted(directory.glob("*.npz")):
            if path.stem.endswith("_cpu_baselines"):
                continue
            check(); b = c.load(path)
            torch.cuda.synchronize() if next(model.parameters()).is_cuda else None
            began = time.perf_counter()
            pred = tr.predict(model, b, check=check)
            pred.update(model_cell=np.array(ident["cell"]), snapshot=np.array(ident["snapshot"]),
                        attention_mode=np.array(ident["attention_mode"]), model_hash=np.array(model_hash),
                        software_scratch=np.array(bool(ident.get("software_scratch", False))))
            torch.cuda.synchronize() if next(model.parameters()).is_cuda else None
            inference_seconds += time.perf_counter()-began
            began = time.perf_counter()
            c.save(out/category/path.name, **pred)
            write_seconds += time.perf_counter()-began
            input_count += len(b["t"]); count += 1
            del b, pred
        status(root, "evaluation", cell=ident["cell"], category=category, files=count)
    expected = 4 if ident["kind"] == "intermediate" else 28
    if count != expected:
        raise RuntimeError("Missing prediction family")
    c.write(out/"complete.json", dict(status="complete", identity=ident, prediction_files=count,
        input_forwards=input_count, inference_seconds=inference_seconds, write_seconds=write_seconds,
        model_hash=model_hash, software_scratch=bool(ident.get("software_scratch", False))))
    return inference_seconds, write_seconds


def evaluate(root, check):
    root = Path(root); locked = c.read(root/"finals_locked.json")
    qa = c.read(root/"reference/complete.json")
    if locked["trained"] != 24 or qa["status"] != "passed":
        raise RuntimeError("Formal evaluation requires all finals and passed MC")
    for name, digest in qa["files"].items():
        if c.sha(root/"reference"/name) != digest:
            raise RuntimeError("Reference changed")
    parent = load_parent_split(root/"reference/fresh_l1024.npz", "test_target")
    began = time.perf_counter(); make_banks(parent, root, check)
    bank_seconds = time.perf_counter()-began
    del parent; gc.collect()
    totals = [0., 0.]
    for ident in c.evaluation_identities():
        check(); path = c.ROOT/ident["checkpoint"]
        if c.sha(path) != locked["checkpoints"][ident["cell"]]:
            raise RuntimeError("Evaluation checkpoint changed")
        p = torch.load(path, map_location="cpu", weights_only=False)
        model = tr.new_model(ident["seed"], ident["attention_mode"], "cuda")
        model.load_state_dict(p["ema"], strict=True); del p
        seconds = evaluate_model(model, ident, root, check)
        totals = [a+b for a, b in zip(totals, seconds)]
        if c.sha(path) != locked["checkpoints"][ident["cell"]]:
            raise RuntimeError("Checkpoint mutated during inference")
        del model; torch.cuda.empty_cache()
    return dict(bank_build_seconds=bank_seconds, inference_seconds=totals[0], prediction_write_seconds=totals[1])
