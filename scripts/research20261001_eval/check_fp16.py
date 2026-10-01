"""One fixed FP16 candidate, unchanged precision gates, validation only."""
from pathlib import Path
import sys
import time
import traceback
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research20261001"))
import endpoint_common as c
import endpoint_data as d
import endpoint_training as tr
import endpoint_sampling as sam
from endpoint_preflight import setup_torch
from fp16_adapter import fp16_mode

EXPORT = ROOT / "artifacts/endpoint_mask_sampler_evaluation_20261001_exports"
DEADLINE = 1790863200.


def check():
    if time.time() >= min(DEADLINE, STARTED + 720):
        raise TimeoutError("FP16 candidate bounded check expired")


def run():
    setup_torch()
    out = EXPORT / "fp16_revision_v1"
    out.mkdir(exist_ok=False)
    c.check_sources(c.read(c.OUT / "run_protocol.json")["sources"], enforce_deadline=False)
    assert c.read(EXPORT / "precision_revision_v1/complete.json")["status"] == "no_candidate_passed"
    c.write(out / "plan.json", dict(started=STARTED, deadline=DEADLINE,
        fixed_candidate="FP16_autocast_FP32_parameters_output_head_probabilities",
        selection="one candidate, no threshold relaxation", training=False,
        test_data=False, max_probability_gate=.005, mean_ce_gate=.001,
        candidate_source_sha256=c.sha(Path(__file__).with_name("fp16_adapter.py"))))
    banks = [(p.stem, c.load(p)) for p in sorted((EXPORT / "precision_revision_v1/banks").glob("*.npz"))]
    assert len(banks) == 8
    identities = [(s, "base", c.base_path(s)) for s in c.SEEDS]
    identities += [(s, a, c.OUT / "training" / f"s{s}_{a}" / "final.pt") for s in c.SEEDS for a in c.ARMS]
    results = []
    for seed, arm, path in identities:
        check(); model = tr.load_ema(path, seed); rows, ces = [], []
        for name, bank in banks:
            fp = tr.predict(model, bank, check=check)
            with fp16_mode(model):
                half = tr.predict(model, bank, check=check)
            dp = float(np.max(np.abs(fp["probability"] - half["probability"])))
            dc = half["ce"].mean(1) - fp["ce"].mean(1)
            ces.extend(dc.tolist())
            rows.append(dict(bank=name, max_probability_difference=dp,
                             mean_ce_difference=float(dc.mean())))
            c.save(out / f"s{seed}_{arm}" / (name + ".npz"), fp32=fp["probability"],
                   candidate=half["probability"], target=bank["target"],
                   bank_hash=np.array(d.bank_hash(bank)))
        result = dict(seed=seed, arm=arm, source_sha256=c.sha(path), rows=rows,
            max_probability_difference=max(r["max_probability_difference"] for r in rows),
            absolute_mean_ce_difference=abs(float(np.mean(ces))))
        result["passed"] = result["max_probability_difference"] <= .005 and result["absolute_mean_ce_difference"] <= .001
        results.append(result); c.write(out / f"s{seed}_{arm}/result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
        del model
        if not result["passed"]:
            c.write(out / "complete.json", dict(status="failed", results=results,
                started=STARTED, completed=time.time(), formal_generation=False))
            return
    timings = []
    model = tr.load_ema(c.base_path(c.SEEDS[0]), c.SEEDS[0])
    with fp16_mode(model):
        for width in (48, 96):
            for sampler in ("monotone-256", "reveal192-repair64"):
                check(); torch.cuda.synchronize(); start = time.perf_counter()
                z = sam.sample(model, c.SEEDS[0], width, np.arange(2000300, 2000316),
                               sampler=sampler, amp=False, check=check)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                sam.audit_repair(z)
                c.save(out / f"technical_w{width}_{sampler}.npz", **z)
                row = dict(width=width, sampler=sampler, seconds=elapsed, images=16,
                           network_calls=int(z["network_calls"]))
                timings.append(row); print(json.dumps(dict(stage="timing", **row)), flush=True)
    c.write(out / "complete.json", dict(status="passed", selected="fp16_fp32_head",
        results=results, timings=timings, started=STARTED, completed=time.time(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), formal_generation=False))


if __name__ == "__main__":
    STARTED = time.time()
    try:
        run()
    except BaseException as error:
        c.write(EXPORT / "fp16_revision_failure_v1.json", dict(time=time.time(),
            error=repr(error), traceback=traceback.format_exc()))
        raise
