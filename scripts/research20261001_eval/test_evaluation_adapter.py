"""Small targeted CPU adapter/driver checks, not a scientific or training rerun."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research20261001")]
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
import endpoint_sampling as sam
import run_evaluation as run
from fp16_adapter import fp16_mode


def main():
    torch.set_num_threads(2); torch.manual_seed(8123)
    model = CoordinateDenseDenoiser(CoordinateDenoiserConfig(d_model=32, n_blocks=1)).eval()
    torch.nn.init.normal_(model.output.weight, std=.03)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for method in run.METHODS:
        kwargs = dict(model=model, seed=92601, width=6, ids=np.array([2000500, 2000501]),
                      sampler=method, amp=False, toy=True)
        left = sam.sample(**kwargs)
        with fp16_mode(model):
            right = sam.sample(**kwargs)
        assert left.keys() == right.keys()
        assert all(np.array_equal(left[k], right[k]) for k in left), method
        sam.audit_repair(right)
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert not any(p.name == "run.lock" for p in run.OUT.glob("*"))
    rates = dict(timings=[dict(width=w, sampler=s, seconds=1.) for w in [48,96] for s in run.METHODS])
    f = run.forecast(rates)
    assert sum(f["shard_counts"].values()) == 456
    assert f["generation_measured_projection_seconds"] == 456
    assert run.DEADLINE == 1790863200. and run.COMPUTE_DEADLINE == 1790861400.
    print("Passed: CPU adapter identity, restoration, RNG and both repair paths; 456-shard count; deadline arithmetic")


if __name__ == "__main__":
    main()
