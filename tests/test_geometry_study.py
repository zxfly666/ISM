"""Contracts for the new experiment; no old result is overwritten."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from ism_diffusion.geometry_study import (
    WIDTHS, TOKENS_PER_UPDATE, array_hash, coordinate_arrays, corrupt_batch,
    make_batch, rng, schedule, balanced_risk,
)
from ism_diffusion.scale_data import ParentSplit


def test_equal_token_schedule():
    for cycle in range(8):
        cells = [schedule(53, cycle*8+i+1) for i in range(8)]
        assert len(set(cells)) == 8
        assert all(TOKENS_PER_UPDATE % w**2 == 0 for w in WIDTHS)


def test_coordinate_sampling_and_stream_independence():
    random = np.random.default_rng(19)
    spins = (random.integers(0, 2, (3, 1024, 1024), dtype=np.int8)*2-1)
    parents = ParentSplit(spins, np.arange(3), 1024, {})
    x = make_batch(parents, 6, 32, "gap", (1, 2, 4, 8), 611, 59)
    # Consuming arbitrary independent-geometry numbers changes no public stream.
    rng(611, 59, "independent_geometry").random(999)
    y = make_batch(parents, 6, 32, "gap", (1, 2, 4, 8), 611, 59)
    assert array_hash(x["clean"]) == array_hash(y["clean"])
    assert not np.array_equal(x["coords"]["A"], x["coords"]["C"])
    for r in range(6):
        ix = (x["origin"][r, 0]+x["axes"][0][r]) % 1024
        iy = (x["origin"][r, 1]+x["axes"][1][r]) % 1024
        expected = spins[x["parent"][r]][np.ix_(ix, iy)] > 0
        assert np.array_equal(expected, x["clean"][r])
    assert np.array_equal(corrupt_batch(x["clean"], 611, 59)[1], corrupt_batch(y["clean"], 611, 59)[1])


def test_d4_preserves_distances_and_pairing():
    original, _ = coordinate_arrays(8, 16, "gap", (1, 2, 4, 8), 19, 23)
    augmented, _ = coordinate_arrays(8, 16, "gap", (1, 2, 4, 8), 19, 23, True)
    for arm in ("A", "B", "C"):
        a, b = original[arm].reshape(8, -1, 2), augmented[arm].reshape(8, -1, 2)
        assert np.allclose(((a[:, :20]-a[:, 20:40])**2).sum(-1), ((b[:, :20]-b[:, 20:40])**2).sum(-1))
    continuous, _ = coordinate_arrays(8, 16, "continuous", (1,), 19, 23, True)
    assert np.array_equal(continuous["A"], continuous["B"])
    assert np.array_equal(continuous["A"], continuous["C"])


def test_reference_weighting_not_crop_weighting():
    records = [dict(geometry="x", t=.5, chain=0, parent=0, ce=0.)]*100
    records += [dict(geometry="x", t=.5, chain=1, parent=1, ce=1.)]
    assert balanced_risk(records)["mean_ce"] == .5


def test_physical_correlation_ones():
    path = Path(__file__).resolve().parents[1] / "scripts/research20260921/evaluate_study.py"
    spec = importlib.util.spec_from_file_location("evaluate_study_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spins = np.ones((2, 4, 4), dtype=np.int8)
    axes = [np.tile(np.array([0, 2, 5, 9]), (2, 1)), np.tile(np.array([0, 1, 4, 8]), (2, 1))]
    result = module.physical_axis_statistics(spins, axes)
    assert np.array_equal(result["pair_sum"], result["pair_count"])
    assert result["pair_count"][:, 0].tolist() == [16, 16]
    assert result["pair_count"][:, 1:].sum() == 2*2*6*4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the experiment GPU")
def test_masked_ce_microbatch_sum_gradient():
    from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
    model = CoordinateDenseDenoiser(CoordinateDenoiserConfig(d_model=32, n_heads=2, n_blocks=1)).cuda()
    clean = torch.randint(0, 2, (6, 4, 4), device="cuda")
    t = torch.full((6,), .5, device="cuda")
    coord = torch.zeros((6, 4, 4, 2), device="cuda")
    mask = torch.rand_like(clean.float()) < .5
    noisy = clean.masked_fill(mask, 2)
    def loss(sl):
        ce = torch.nn.functional.cross_entropy(model(noisy[sl], t[sl], coord[sl]), clean[sl], reduction="none")
        return (ce*mask[sl]/t[sl, None, None]).sum()/clean.numel()
    loss(slice(None)).backward()
    first = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for start in (0, 2, 4):
        loss(slice(start, start+2)).backward()
    for p, expected in zip(model.parameters(), first):
        if expected is not None:
            assert torch.allclose(p.grad, expected, atol=3e-6, rtol=3e-5)
