"""Bounded, resumable study queue. Launch after preflight and smoke pass."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from ism_diffusion.geometry_study import (
    ARMS, MODEL, GAPS, TOKENS_PER_UPDATE, atomic_json, file_hash, train_one,
)
from ism_diffusion.scale_data import load_parent_split
from evaluate_study import evaluate_one, aggregate

STUDY = ROOT / "artifacts/geometry_alignment_20260921"


def freeze():
    path = STUDY / "frozen_protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    pre = STUDY / "preflight"
    if not (pre / "complete.json").exists() or not json.loads((pre / "s0.json").read_text())["gate_i_pass"]:
        raise RuntimeError("preflight did not pass")
    if not (STUDY / "smoke/complete.json").exists():
        raise RuntimeError("new trainer/evaluator end-to-end smoke did not pass")
    if not (STUDY / "resume_test/complete.json").exists():
        raise RuntimeError("interrupted-versus-uninterrupted resume test did not pass")
    bench = json.loads((pre / "benchmark.json").read_text())
    steps = 24000
    seconds_train_one = bench["training_seconds_per_update"] * steps * 1.35
    sp = bench["sample_seconds_per_256step"]
    seconds_gen_one = 128*sp["48"] + 256*sp["64"] + 256*sp["32"] + 128*sp["48"] + 128*sp["96"]
    seconds_gen_one *= 1.25
    # Reserve one hour for conditional probes/reference analysis and an hour
    # for packaging/variance in runtime. Counts are chosen before any test result.
    forecast18 = 18*(seconds_train_one + seconds_gen_one) + 7200
    seeds = [91001, 91002, 91003, 91004, 91005, 91006]
    if forecast18 > 14.25*3600:
        seeds = seeds[:3]
    forecast = len(seeds)*3*(seconds_train_one+seconds_gen_one)+7200
    if forecast > 14.25*3600:
        raise RuntimeError("Even nine balanced runs exceed the budget; do not start blindly")
    sources = list((ROOT / "ism_diffusion").glob("*.py"))
    sources += [p for p in (ROOT / "scripts/research20260921").glob("*.py") if p.name != "remote_transport.py"]
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    for p in sources:
        dst = STUDY / "source_snapshot" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    config = dict(version="geometry_alignment_v1", created_unix=time.time(), steps=steps,
        seeds=seeds, arms=list(ARMS), microbatch=bench["microbatch"], model=MODEL,
        tokens_per_update=TOKENS_PER_UPDATE, valid_tokens_per_model=steps*TOKENS_PER_UPDATE,
        gaps=list(GAPS), widths=[16,24,32,48], mixture="equal valid tokens over 8 class/width cells",
        rng_strategy="stateless named streams; shared clean/time/MASK; independent geometry stream",
        precision="bf16", ema=.999, include_w96=True, test_samples=512,
        sampler=dict(method="S0", steps=256, schedule="cos_squared", temperature=1., mc_correction=False),
        primary="paired EMA-final masked CE on gap57/W32 and s10/W48; macro task,t,chain,parent weights",
        budget_hours=15, forecast_hours=forecast/3600, training_reserve_hours=2.5,
        dataset_sha256=file_hash(ROOT / "data/level1/parents_l1024.npz"),
        benchmark=bench, source_hashes=source_hashes,
        reference_rule="fresh 4 independent chains if complete and both split-Rhat<=1.1 and all ESS>=16; else historical test_target with explicit limitation",
        fresh_reference=dict(chains=4, samples_per_chain=128, burn_in_sweeps=40, sweeps_between=4, seed=20260922))
    config["protocol_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(path, config)
    print(json.dumps(dict(frozen_protocol=config["protocol_hash"], seeds=seeds, steps=steps, forecast_hours=forecast/3600)), flush=True)
    return config


def launch_reference():
    reference = STUDY / "reference"
    reference.mkdir(exist_ok=True)
    path = reference / "fresh_l1024.npz"
    if path.exists():
        return None
    pid_path = reference / "pid.json"
    if pid_path.exists():
        pid = json.loads(pid_path.read_text())["pid"]
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists() and b"generate_level1_parents.py" in cmdline.read_bytes():
            return None
    command = [sys.executable, "-u", str(ROOT / "generate_level1_parents.py"), "--output", str(path),
        "--lattice-size", "1024", "--train-chains", "0", "--val-chains", "0", "--target-chains", "4",
        "--control-chains", "0", "--samples-per-chain", "128", "--burn-in-sweeps", "40",
        "--sweeps-between", "4", "--workers", "4", "--seed", "20260922"]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with (reference / "generation.log").open("a") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    atomic_json(pid_path, dict(pid=proc.pid, command=command, launched=time.time()))
    return proc


def select_reference():
    decision_path = STUDY / "reference/selection.json"
    if decision_path.exists():
        return json.loads(decision_path.read_text())
    fresh = STUDY / "reference/fresh_l1024.npz"
    selected = ROOT / "data/level1/parents_l1024.npz"
    reason = "fresh pool incomplete; historical two test chains only"
    diagnostics = None
    if fresh.exists():
        try:
            with np.load(fresh, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"].item()))
                diagnostics = meta["diagnostics"]
            ok = all(np.isfinite(diagnostics[k]) and diagnostics[k] <= 1.1 for k in ("energy_split_rhat", "abs_magnetization_split_rhat"))
            ok &= all(x["ess"] >= 16 for k in ("energy", "abs_magnetization") for x in diagnostics[k])
            if ok:
                selected = fresh
                reason = "predeclared MC diagnostic gates passed; independent four-chain reference"
            else:
                reason = "fresh diagnostic gate failed; historical reference only; no retry to select favorable model result"
        except (OSError, ValueError, KeyError) as error:
            reason = "fresh reference incomplete/unreadable: " + str(error)
    decision = dict(path=str(selected), sha256=file_hash(selected), reason=reason, diagnostics=diagnostics)
    atomic_json(decision_path, decision)
    return decision


def verify_pairing(seed, config):
    records = []
    for arm in ARMS:
        folder = STUDY / "training" / f"s{seed}_{arm}"
        hashes = [json.loads(line) for line in (folder / "paired_hashes.jsonl").read_text().splitlines()]
        init = json.loads((folder / "run_config.json").read_text())["initialization_hash"]
        records.append((init, hashes))
    if records[0] != records[1] or records[0] != records[2]:
        raise RuntimeError(f"paired initialization/data/MASK mismatch for seed {seed}")
    atomic_json(STUDY / "training" / f"paired_seed_{seed}.json", dict(pass_=True, checked_updates=100,
                protocol_hash=config["protocol_hash"]))


def smoke():
    output = STUDY / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    parent = load_parent_split(ROOT / "data/level1/parents_l1024.npz", "train")
    val = load_parent_split(ROOT / "data/level1/parents_l1024.npz", "val")
    bench = json.loads((STUDY / "preflight/benchmark.json").read_text())
    config = dict(steps=8, microbatch=bench["microbatch"], protocol_hash="smoke-v1", smoke=True, test_samples=4)
    result = train_one(parent, val, output / "training", "A", 808801, config, time.time()+600)
    del parent, val
    evaluation = evaluate_one(output / "training/final.pt", ROOT / "data/level1/parents_l1024.npz",
                             output / "evaluation", "A", 808801, config, time.time()+600)
    atomic_json(output / "complete.json", dict(training=result["status"], evaluated=True,
                primary_ce=evaluation["conditional"]["mean_ce"], note="8-update plumbing smoke, not a research result"))
    print("END_TO_END_SMOKE_PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(); return
    lock = (STUDY / "queue.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = freeze()
    if args.freeze_only:
        return
    deadline = config["created_unix"] + config["budget_hours"]*3600
    reference_process = launch_reference()
    parent = load_parent_split(ROOT / "data/level1/parents_l1024.npz", "train")
    validation = load_parent_split(ROOT / "data/level1/parents_l1024.npz", "val")
    finished = []
    try:
        for index, seed in enumerate(config["seeds"]):
            order = list(ARMS[index % 3:]+ARMS[:index % 3])
            for arm in order:
                atomic_json(STUDY / "status.json", dict(phase="training", seed=seed, arm=arm,
                    completed_training=finished, updated=time.time(), deadline=deadline, pid=os.getpid()))
                result = train_one(parent, validation, STUDY / "training" / f"s{seed}_{arm}", arm, seed, config,
                                   deadline-config["training_reserve_hours"]*3600)
                if result["status"] != "trained":
                    raise TimeoutError("training reserve/deadline reached; checkpoint saved")
                finished.append(dict(seed=seed, arm=arm, seconds=result["elapsed_seconds"]))
            verify_pairing(seed, config)
        del parent, validation
        torch.cuda.empty_cache()
        selected = select_reference()
        config = dict(config, reference_selected=selected)
        for index, seed in enumerate(config["seeds"]):
            for arm in ARMS:
                atomic_json(STUDY / "status.json", dict(phase="evaluation", seed=seed, arm=arm,
                    completed_training=finished, updated=time.time(), deadline=deadline, pid=os.getpid()))
                evaluate_one(STUDY / "training" / f"s{seed}_{arm}/final.pt", Path(selected["path"]),
                    STUDY / "evaluation" / f"s{seed}_{arm}", arm, seed, config, deadline-900)
        summary = aggregate(STUDY, config)
        checksums = {str(p.relative_to(STUDY)): file_hash(p) for p in STUDY.rglob("*")
                     if p.is_file() and p.name not in {"checksums.json", "queue.log", "queue.lock", "status.json"}}
        atomic_json(STUDY / "checksums.json", checksums)
        atomic_json(STUDY / "status.json", dict(phase="complete", models=summary["completed_models"],
            elapsed_hours=(time.time()-config["created_unix"])/3600, updated=time.time(), pid=os.getpid(),
            local_backup="pending_download_and_verification"))
    except Exception as error:
        traceback.print_exc()
        atomic_json(STUDY / "status.json", dict(phase="budget_stopped" if isinstance(error, TimeoutError) else "failed",
            error=str(error), completed_training=finished, updated=time.time(), pid=os.getpid()))
        raise
    finally:
        if reference_process is not None and reference_process.poll() is None:
            # This subprocess is owned by this queue, not another user's task.
            os.killpg(reference_process.pid, signal.SIGTERM)
            try:
                reference_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(reference_process.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
