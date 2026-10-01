"""Bounded validation-only generation precision revision and full-shard timing.

No training, MC, scientific image IDs or test statistics are run here. Previous
failed gates remain failed. Every candidate uses the same original 256 inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research20261001"))
import endpoint_common as c
import endpoint_data as d
import endpoint_training as tr
import endpoint_sampling as sam
from endpoint_preflight import setup_torch
from precision_adapter import CANDIDATES, precision_mode

TRAIN = c.OUT
EXPORT = ROOT / "artifacts/endpoint_mask_sampler_evaluation_20261001_exports"
DEADLINE = 1790863200.0  # 2026-10-01 22:00 UTC+8; user-approved evaluation window.


def check():
    if time.time() >= min(DEADLINE, STARTED + 720):
        raise TimeoutError("Bounded precision check deadline")


def compare(model, banks, folder, mode, reference=None):
    records, ces, arrays = [], [], {}
    for label, bank in banks:
        check()
        fp = tr.predict(model, bank, check=check) if reference is None else reference[label]
        with precision_mode(model, mode):
            candidate = tr.predict(model, bank, amp=True, check=check)
        dp = float(np.max(np.abs(fp["probability"] - candidate["probability"])))
        dc = candidate["ce"].mean(1) - fp["ce"].mean(1)
        ces.extend(dc.tolist())
        records.append(dict(bank=label, max_probability_difference=dp,
                            mean_ce_difference=float(dc.mean())))
        c.save(folder / (label + ".npz"), fp32=fp["probability"],
               candidate=candidate["probability"], target=bank["target"],
               bank_hash=np.array(d.bank_hash(bank)), mode=np.array(mode))
        arrays[label] = fp
    result = dict(mode=mode, rows=records, inputs=256,
                  max_probability_difference=max(r["max_probability_difference"] for r in records),
                  absolute_mean_ce_difference=abs(float(np.mean(ces))))
    result["passed"] = result["max_probability_difference"] <= .005 and result["absolute_mean_ce_difference"] <= .001
    c.write(folder / "result.json", result)
    print(json.dumps(dict(stage="precision", output=str(folder.relative_to(ROOT)), **result)), flush=True)
    return result, arrays


def run():
    setup_torch()
    out = EXPORT / "precision_revision_v1"
    out.mkdir(parents=True, exist_ok=False)
    c.check_sources(c.read(TRAIN / "run_protocol.json")["sources"], enforce_deadline=False)
    c.write(out / "plan.json", dict(started=STARTED, hard_deadline=DEADLINE,
        technical_timeout_seconds=720, candidates=list(CANDIDATES),
        selection="first candidate passing original gate on all 24 frozen final/base models",
        max_probability_gate=.005, mean_ce_gate=.001,
        scientific_generation=False, training=False))
    val = d.load_parent(c.DATA, "val")
    ids = d.select_parents(val, 32)
    banks = [(f"w{w}_m{m}", d.local_bank(val, ids, w, m, "preflight_validation"))
             for w in (48, 96) for m in (1, 2, 8, 32)]
    for label, bank in banks:
        c.save(out / "banks" / (label + ".npz"), **bank)
    checks = []
    selected = None
    base = c.base_path(c.SEEDS[0])
    model = tr.load_ema(base, c.SEEDS[0])
    reference = None
    for mode in CANDIDATES:
        result, reference = compare(model, banks, out / mode / "base_s92601", mode, reference)
        if not result["passed"]:
            checks.append(dict(mode=mode, passed=False, failure="original_256_inputs"))
            continue
        # No tuning against test data: all remaining inherited/final identities.
        identities = [(s, "base", c.base_path(s)) for s in c.SEEDS[1:]]
        identities += [(s, arm, TRAIN / "training" / f"s{s}_{arm}" / "final.pt")
                       for s in c.SEEDS for arm in c.ARMS]
        passed = True
        for seed, arm, path in identities:
            check()
            candidate_model = tr.load_ema(path, seed)
            r, _ = compare(candidate_model, banks, out / mode / f"s{seed}_{arm}", mode)
            checks.append(dict(mode=mode, seed=seed, arm=arm, source_sha256=c.sha(path), **r))
            del candidate_model
            if not r["passed"]:
                passed = False
                break
        if passed:
            selected = mode
            break
    if selected is None:
        c.write(out / "complete.json", dict(status="no_candidate_passed", checks=checks,
            started=STARTED, completed=time.time(), formal_generation_started=False))
        return
    timings = []
    with precision_mode(model, selected):
        for width in (48, 96):
            for sampler in ("monotone-256", "reveal192-repair64"):
                check(); torch.cuda.synchronize(); start = time.perf_counter()
                z = sam.sample(model, c.SEEDS[0], width, np.arange(2000200, 2000216),
                               sampler=sampler, amp=True, check=check)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                sam.audit_repair(z)
                c.save(out / f"technical_w{width}_{sampler}.npz", **z)
                row = dict(width=width, sampler=sampler, seconds=elapsed, images=16,
                           network_calls=int(z["network_calls"]), precision=selected)
                timings.append(row); print(json.dumps(dict(stage="timing", **row)), flush=True)
    c.write(out / "complete.json", dict(status="passed", selected=selected, checks=checks,
        timings=timings, started=STARTED, completed=time.time(),
        source_sha256=c.sha(base), frozen_training_sources_unchanged=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), formal_generation_started=False))


if __name__ == "__main__":
    STARTED = time.time()
    try:
        run()
    except BaseException as error:
        c.write(EXPORT / "precision_revision_failure_v1.json",
                dict(time=time.time(), error=repr(error), traceback=traceback.format_exc()))
        raise
