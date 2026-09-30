"""Simulate an interruption after a saved step; compare final weights and EMA."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from ism_diffusion import geometry_study as study
from ism_diffusion.scale_data import load_parent_split


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    # Isolate checkpoint/RNG restoration from nondeterministic BF16 attention
    # backward kernels. The same production trainer runs a small math-SDPA
    # fixture; this is NOT a claim of bitwise reproducibility for production.
    study.MODEL = dict(study.MODEL, d_model=32, n_heads=2, n_blocks=1)
    result_root = ROOT / "artifacts/geometry_alignment_20260921/resume_test"
    out = result_root / "deterministic"
    out.mkdir(parents=True, exist_ok=True)
    parent = load_parent_split(ROOT / "data/level1/parents_l1024.npz", "train")
    val = load_parent_split(ROOT / "data/level1/parents_l1024.npz", "val")
    bench = json.loads((result_root.parent / "preflight/benchmark.json").read_text())
    config = dict(steps=8, checkpoint_interval=4, microbatch=bench["microbatch"], protocol_hash="resume-test-v2-deterministic")
    deadline = time.time()+300
    study.train_one(parent, val, out / "uninterrupted", "B", 877201, config, deadline)
    original = study.train_update
    def interrupt_at_five(*args, **kwargs):
        if args[6] == 5:
            raise RuntimeError("simulated_interruption")
        return original(*args, **kwargs)
    if not (out / "resumed/training_complete.json").exists():
        try:
            with patch.object(study, "train_update", side_effect=interrupt_at_five):
                study.train_one(parent, val, out / "resumed", "B", 877201, config, deadline)
        except RuntimeError as error:
            if str(error) != "simulated_interruption":
                raise
        study.train_one(parent, val, out / "resumed", "B", 877201, config, deadline)
    a = torch.load(out / "uninterrupted/final.pt", map_location="cpu", weights_only=False)
    b = torch.load(out / "resumed/final.pt", map_location="cpu", weights_only=False)
    delta = {group: max(float((a[group][name]-b[group][name]).abs().max()) for name in a[group]) for group in ("model", "ema")}
    assert max(delta.values()) < 2e-6, delta
    assert a["counters"] == b["counters"]
    study.atomic_json(result_root / "complete.json", dict(status="pass", maximum_weight_difference=delta,
        counters_equal=True, resumed_from_step=4, final_step=8,
        scope="deterministic small-model fixture of production trainer and named data streams",
        production_caveat="BF16 SDPA backward can be numerically nondeterministic; full-model pre-interruption reruns already differ. Production resume is not promised bitwise."))
    print(json.dumps(delta), flush=True)


if __name__ == "__main__":
    with sdpa_kernel(SDPBackend.MATH):
        main()
