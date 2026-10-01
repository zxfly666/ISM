"""Reference, banks, frozen predictions and auditable generation products."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import endpoint_common as c
import endpoint_data as d


def new_reference():
    c.import_old()
    from ism_diffusion.ising import generate_independent_chains, energy_density, magnetization
    from ism_diffusion.diagnostics import integrated_autocorrelation_time, split_rhat
    assert (c.OUT/"run.lock").exists()
    protocol = c.read(c.OUT/"run_protocol.json"); c.check_sources(protocol["sources"])
    out = c.OUT/"reference"; out.mkdir(exist_ok=False)
    start = time.time()
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in np.random.SeedSequence(2026100111).spawn(16)]
    args = dict(lattice_size=1024, chain_seeds=seeds, samples_per_chain=128,
                burn_in_sweeps=40, sweeps_between=4, beta=float(c.BETA), adaptation_sweeps=3,
                pilot_cluster_steps=128, workers=4, backend="numba",
                initial_states=["random"]*8+["plus"]*4+["minus"]*4, return_chain_metadata=True)
    c.write(out/"protocol.json", dict(**args, master_seed=2026100111, role="evaluation_only"))
    spins, chains, metadata = generate_independent_chains(**args); c.check()
    md = dict(lattice_size=1024, beta=float(c.BETA), seed=2026100111, chain_metadata=metadata, chain_seeds=seeds)
    c.save(out/"fresh_l1024.npz", test_target_packed=d.pack(spins), test_target_chain_id=chains, metadata=np.array(json.dumps(md)))
    parent = d.Parent(spins, chains, 1024, md)
    crop_chunks, origins = [], []
    stats = {}
    for start_i in range(0, 2048, 16):
        c.check(); ids = np.arange(start_i, start_i+16)
        origin = np.stack([c.rng(c.CFG["role_seeds"]["banks"], int(i), "reference_crop96").integers(1024, size=2) for i in ids])
        axes = np.broadcast_to(np.arange(96), (16, 2, 96)).copy()
        crop = (2*d.sample(parent, ids, axes, origin)-1).astype(np.int8)
        crop_chunks.append(crop); origins.append(origin)
        for view, values in d.multiscale_stats(crop).items():
            stats.setdefault(view, []).append(values)
        stats.setdefault("w48", []).append(d.physical_stats(crop[:, 24:72, 24:72]))
    crop96 = np.concatenate(crop_chunks)
    c.save(out/"crop96.npz", spins=crop96, parent=np.arange(2048), chain=chains, origin=np.concatenate(origins))
    for view, parts in stats.items():
        c.save(out/f"stats_{view}.npz", **d.concatenate(parts), parent=np.arange(2048), chain=chains)
    full = d.concatenate(stats["full"])
    m = magnetization(spins)
    trace = dict(energy=energy_density(spins), m=m, abs_m=np.abs(m), m2=m*m,
                 G25_48=(full["pair_sum"][:, 25:49]/full["pair_count"][:, 25:49]).mean(1))
    diag = {k: dict(split_rhat=float(split_rhat([v[chains == i] for i in range(16)])),
                   chains=[integrated_autocorrelation_time(v[chains == i]) for i in range(16)]) for k, v in trace.items()}
    passed = all(np.isfinite(diag[k]["split_rhat"]) and diag[k]["split_rhat"] <= 1.1 and all(v["ess"] >= 16 for v in diag[k]["chains"])
                 for k in ["energy", "abs_m", "m2", "G25_48"])
    c.save(out/"chain_traces.npz", chain=chains, **trace)
    c.write(out/"qa.json", dict(status="passed" if passed else "failed", diagnostics=diag, elapsed_seconds=time.time()-start))
    if not passed:
        raise RuntimeError("Reference QA failed: no resampling to pass")
    make_banks(parent)
    c.write(out/"complete.json", dict(status="passed", parents=2048, chains=16, elapsed_seconds=time.time()-start,
                                      files={p.name: c.sha(p) for p in out.glob("*.npz")}))


def make_banks(parent):
    out = c.OUT/"banks"; out.mkdir(exist_ok=False)
    ids = d.select_parents(parent, 256)
    for w in [48, 96]:
        for m in [1, 2, 8, 32]:
            bank = d.local_bank(parent, ids, w, m, "exact_test")
            d.validate_bank(bank); c.save(out/f"exact_w{w}_m{m}.npz", **bank)
        bank = d.stress_bank(parent, w); d.validate_bank(bank)
        c.save(out/f"stress_w{w}.npz", **bank)
    for label, kind, k in [("C48_K115", "continuous", 115), ("C48_K1152", "continuous", 1152), ("H48_K512", "held_gap", 512)]:
        bank = d.retention_bank(parent, ids, kind, k); d.validate_bank(bank)
        c.save(out/f"{label}.npz", **bank)
    small = d.select_parents(parent, 16)
    for w in [48, 96]:
        for m in [1, 2, 8, 32]:
            for floor in [.01, .002]:
                bank = d.local_bank(parent, small, w, m, "phase0", clock=floor)
                d.validate_bank(bank); c.save(out/f"phase0_w{w}_m{m}_floor{floor}.npz", **bank)


def validation_banks():
    out = c.OUT/"validation_banks"; out.mkdir(exist_ok=False)
    val = d.load_parent(c.DATA, "val")
    ids = d.select_parents(val, 128)
    for m in [1, 2, 8, 32]:
        bank = d.local_bank(val, ids, 48, m, "learning_validation")
        d.validate_bank(bank); c.save(out/f"exact_w48_m{m}.npz", **bank)
    c.write(out/"split.json", dict(parent_count=128, chain_ids=np.unique(val.chain_ids).tolist(), original_split=val.metadata["split_manifest"]))


def old_coverage_audit(parent):
    c.import_old()
    import identification_data as old
    out = c.OUT/"coverage_audit"; out.mkdir(exist_ok=False)
    cycles = np.linspace(0, 374, 32, dtype=int)
    records = []
    for seed in c.SEEDS:
        parts = {k: [] for k in ["step", "width", "sparse", "M", "t"]}
        for cycle in cycles:
            for offset in range(32):
                c.check(); step = int(cycle*32+offset+1)
                b = old.training_batch(parent, seed, step, "I-F")
                size = len(b["t"])
                parts["step"].append(np.full(size, step)); parts["width"].append(np.full(size, b["width"]))
                parts["sparse"].append(np.full(size, b["sparse"])); parts["M"].append(b["mask"].sum((1, 2))); parts["t"].append(b["t"])
        merged = {k: np.concatenate(v) for k, v in parts.items()}
        c.save(out/f"s{seed}.npz", **merged)
        records.append(dict(seed=seed, reconstructed_updates=1024, images=len(merged["t"]),
                            ordinary_low_M_counts={str(m): int(((merged["M"] <= m) & ~merged["sparse"]).sum()) for m in [1, 2, 8, 32]}))
    c.write(out/"summary.json", dict(historical_sampled_not_full_replay=True, total_updates=6144, cycles=cycles.tolist(), records=records))


def generate_model(model, seed, out, width, sampler, images, source_sha, floor=.002, phase0=False):
    import torch
    import endpoint_sampling as sam
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    parts = {}
    durations = []
    for first in range(0, images, 16):
        c.check(); ids = np.arange(first, first+16) + (1000000 if phase0 else 0)
        torch.cuda.synchronize(); start = time.perf_counter()
        result = sam.sample(model, seed, width, ids, sampler, floor)
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        sam.audit_repair(result)
        result.update(source_sha256=np.array(source_sha), seed=np.array(seed), sampling_seconds=np.array(seconds),
                      phase0=np.array(phase0), width=np.array(width))
        c.save(out/f"shard_{first:05d}.npz", **result)
        durations.append(seconds)
        for image_type, key in [("final", "spins"), ("prefix", "prefix_spins"), ("oracle", "oracle_spins")]:
            if key not in result:
                continue
            for view, stat in d.multiscale_stats(result[key]).items():
                parts.setdefault(f"{image_type}_{view}", []).append(stat)
        c.log("generation_shard", seed=seed, width=width, sampler=sampler, phase0=phase0, output=str(out.relative_to(c.ROOT)), completed=first+16, total=images)
    for name, rows in parts.items():
        c.save(out/f"stats_{name}.npz", **d.concatenate(rows), image_ids=np.arange(images)+(1000000 if phase0 else 0), seed=np.array(seed))
    c.write(out/"complete.json", dict(status="complete", images=images, width=width, sampler=sampler, clock_floor=floor,
                                      sampling_seconds=sum(durations), source_sha256=source_sha, per_image_network_calls=256))


def evaluate_all(protocol):
    import torch
    import endpoint_training as tr
    c.check_sources(protocol["sources"])
    finals = c.read(c.OUT/"final_lock.json")
    assert len(finals["files"]) == 18
    for path, expected in finals["files"].items():
        assert c.sha(c.ROOT/path) == expected
    all_banks = sorted((c.OUT/"banks").glob("*.npz"))
    formal_banks = [p for p in all_banks if not p.name.startswith("phase0")]
    assert len(formal_banks) == 13
    for seed in c.SEEDS:
        for arm in c.ARMS:
            c.check(); key = f"s{seed}_{arm}"
            cell = c.OUT/"evaluation"/key; cell.mkdir(parents=True, exist_ok=False)
            base = c.OUT/"training"/key
            path = base/"final.pt"; source = c.sha(path)
            model = tr.load_ema(path, seed)
            for bankpath in formal_banks:
                bank = c.load(bankpath)
                prediction = tr.predict(model, bank)
                c.save(cell/"conditional"/bankpath.name, **prediction, source_sha256=np.array(source), bank_sha256=np.array(c.sha(bankpath)))
            for step in [4000, 6000, 8000]:
                snapshot = path if step == 8000 else base/f"ema_{step}.pt"
                sm = model if step == 8000 else tr.load_ema(snapshot, seed)
                for bankpath in sorted((c.OUT/"validation_banks").glob("*.npz")):
                    prediction = tr.predict(sm, c.load(bankpath))
                    c.save(cell/"learning"/f"step{step}_{bankpath.name}", **prediction, source_sha256=np.array(c.sha(snapshot)), bank_sha256=np.array(c.sha(bankpath)))
                if sm is not model:
                    del sm; torch.cuda.empty_cache()
            for w, images in [(96, 128), (48, 64)]:
                for sampler in ["monotone-256", "reveal192-repair64"]:
                    generate_model(model, seed, cell/"generation"/f"w{w}_{sampler}", w, sampler, images, source)
            assert c.sha(path) == source
            c.write(cell/"complete.json", dict(status="complete", source_sha256=source, formal_predictions=13,
                                               learning_predictions=12, final_images=384, prefix_images=192, oracle_images=192))
            del model; torch.cuda.empty_cache()
    # Original six models are frozen diagnostics, never new training replications.
    for seed in c.SEEDS:
        base = c.base_path(seed); source = c.sha(base)
        model = tr.load_ema(base, seed)
        out = c.OUT/"phase0"/f"s{seed}"; out.mkdir(parents=True, exist_ok=False)
        for bankpath in all_banks:
            if bankpath.name.startswith("phase0"):
                prediction = tr.predict(model, c.load(bankpath))
                c.save(out/"conditional"/bankpath.name, **prediction, source_sha256=np.array(source), bank_sha256=np.array(c.sha(bankpath)))
        for floor in [.01, .002]:
            generate_model(model, seed, out/f"floor{floor}", 96, "monotone-256", 32, source, floor, phase0=True)
        c.write(out/"complete.json", dict(status="complete", predictions=16, images=64, source_sha256=source))
        del model; torch.cuda.empty_cache()
