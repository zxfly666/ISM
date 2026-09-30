"""CPU-only preregistered design. No training, GPU use, or old-file mutation.

The companion protocol is CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md.
This is not the campaign runner and does not certify its future GPU pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("G00", "G10", "G01", "G11", "G11S")
SEEDS = tuple(range(92501, 92507))
WIDTHS = (16, 24, 32, 48)
STEPS = 12000
TOKENS = 18432
AUX_SLOTS = 512
MC_SEED = 2026092511
STUDY_SEED = 2026092521
GEOMETRIES = (
    ("id_continuous48", 48, "continuous"),
    ("id_gap48", 48, "train_gap"),
    ("held_gap48_span174", 48, "held_gap"),
    ("held_context96", 96, "continuous"),
    ("held_context128", 128, "continuous"),
    ("held_gap96_span354", 96, "held_gap"),
)
PRIMARY = (
    ("H1_distance", "G11", "G10", "held_gap48_span174"),
    ("H2_context", "G11", "G01", "held_context96"),
    ("H3_fine_geometry", "G11", "G11S", "held_gap48_span174"),
)


def rng(seed: int, index: int, role: str) -> np.random.Generator:
    tag = list(np.frombuffer(hashlib.sha256(role.encode()).digest()[:16], dtype="<u4"))
    return np.random.default_rng(np.random.SeedSequence([STUDY_SEED, int(seed), int(index), *map(int, tag)]))


def increments(width: int, kind: str) -> np.ndarray:
    if width % 4 or width < 8:
        raise ValueError("supported widths must be multiples of four, >= 8")
    q = width // 4 - 1
    if kind == "continuous":
        return np.ones(width - 1, dtype=np.int64)
    if kind == "train_gap":
        return np.repeat([1, 2, 4, 8], [q + 1, q, q + 2, q]).astype(np.int64)
    if kind == "held_gap":
        return np.repeat([3, 6], [3 * q + 3, q]).astype(np.int64)
    raise ValueError(kind)


def axes(width: int, kind: str, seed: int, index: int, batch: int) -> np.ndarray:
    inc = increments(width, kind)
    random = rng(seed, index, "physical_axes")
    out = np.empty((batch, 2, width), dtype=np.int64)
    out[..., 0] = 0
    for row in range(batch):
        for axis in range(2):
            out[row, axis, 1:] = random.permutation(inc).cumsum()
    if out.max() >= 512:
        raise ValueError("axis extent must stay below half of the L1024 parent")
    return out


def supplied_coordinates(physical_axes: np.ndarray, arm: str, d4=None) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(arm)
    a = np.asarray(physical_axes, dtype=np.float32).copy()
    batch, _, width = a.shape
    if arm == "G11S":
        # Same endpoint spans, not rank/unit coordinates.
        spans = a[..., -1:].copy()
        a = (spans / (width - 1)) * np.arange(width, dtype=np.float32)
        a[..., -1:] = spans
    c = np.empty((batch, width, width, 2), dtype=np.float32)
    c[..., 0] = a[:, 0, :, None]
    c[..., 1] = a[:, 1, None, :]
    if d4 is not None:
        for row, code in enumerate(d4):
            if int(code) & 1:
                c[row] = c[row][..., ::-1].copy()
            if int(code) & 2:
                c[row, ..., 0] *= -1
            if int(code) & 4:
                c[row, ..., 1] *= -1
    return c


def schedule(seed: int, step: int, arm: str) -> dict:
    if arm not in ARMS or not 1 <= step <= STEPS:
        raise ValueError((arm, step))
    cycle, offset = divmod(step - 1, 32)
    latent = int(rng(seed, cycle, "balanced_schedule").permutation(32)[offset])
    cell, repeat = divmod(latent, 4)
    latent_width = WIDTHS[cell % 4]
    latent_kind = "continuous" if cell < 4 else "train_gap"
    width = latent_width if arm in ("G10", "G11", "G11S") else 48
    kind = latent_kind if arm in ("G01", "G11", "G11S") else "continuous"
    return dict(width=width, kind=kind, sparse=repeat == 3,
                batch=TOKENS // (width * width), latent_cell=cell)


def observations(clean: np.ndarray, seed: int, index: int, k: int, queries=64) -> dict:
    """Shared true data; query sites excluded from evidence, nested in k."""
    flat = np.asarray(clean).reshape(len(clean), -1)
    batch, sites = flat.shape
    if queries > sites or not 0 <= k <= sites - queries:
        raise ValueError((sites, queries, k))
    noisy = np.full(flat.shape, 2, dtype=np.int64)
    qq = np.empty((batch, queries), dtype=np.int64)
    evidence = np.empty((batch, k), dtype=np.int64)
    random = rng(seed, index, "observation_order")
    for row in range(batch):
        order = random.permutation(sites)
        qq[row] = order[:queries]
        evidence[row] = order[queries:queries + k]
        noisy[row, evidence[row]] = flat[row, evidence[row]]
    return dict(noisy=noisy, queries=qq, evidence=evidence,
                labels=flat[np.arange(batch)[:, None], qq],
                t=np.full(batch, 1 - k / sites, dtype=np.float32))


def evaluation_ks(width: int) -> tuple[int, ...]:
    n = width * width
    return (2, 32, 512, n // 2, int(np.floor(.05 * n)))


def auxiliary_slot_counts(batch: int, seed: int, index: int) -> np.ndarray:
    """Equal TOTAL supervision slots; don't confound window size with labels."""
    count, remainder = divmod(AUX_SLOTS, batch)
    result = np.full(batch, count, dtype=np.int64)
    result[rng(seed, index, "aux_slot_allocation").permutation(batch)[:remainder]] += 1
    return result


def plan() -> dict:
    return dict(study="core_geometry_factorial_20260925", design_version=1,
                status="design_only_pending_runner_and_gpu_preflight",
                arms=ARMS, seeds=SEEDS, widths=WIDTHS, steps=STEPS,
                tokens_per_update=TOKENS, total_updates=len(ARMS)*len(SEEDS)*STEPS,
                auxiliary_query_slots_per_sparse_update=AUX_SLOTS,
                mc_seed=MC_SEED, mc_chains=8, mc_parents_per_chain=128,
                conditioning_geometries=GEOMETRIES, conditions_per_geometry=5,
                conditioning_banks=30, prediction_units=900, parents_per_bank=128,
                queries_per_parent=64, primary_comparisons=PRIMARY,
                primary_ci=1-.05/3, practical_ce_margin=.002,
                generation_width=96, generation_per_model=64,
                generation_images=1920, generation_shards=120,
                hard_seconds=36000, launch_forecast_max_seconds=34200,
                old_weights_allowed=False, automatic_budget_reduction_allowed=False)


def self_test() -> dict:
    for width in (*WIDTHS, 96):
        train, held = increments(width, "train_gap"), increments(width, "held_gap")
        assert len(train) == len(held) == width - 1
        assert train.sum() == held.sum() < 512
        assert set(train) == {1, 2, 4, 8} and set(held) == {3, 6}
    assert increments(48, "train_gap").sum() == 174
    assert increments(96, "held_gap").sum() == 354
    for seed in SEEDS:
        for arm in ARMS:
            rows = [schedule(seed, step, arm) for step in range(1, 33)]
            assert all(x["batch"] * x["width"]**2 == TOKENS for x in rows)
            for cell in range(8):
                rr = [x for x in rows if x["latent_cell"] == cell]
                assert len(rr) == 4 and sum(x["sparse"] for x in rr) == 1
            if arm in ("G00", "G01"):
                assert {x["width"] for x in rows} == {48}
            if arm in ("G00", "G10"):
                assert {x["kind"] for x in rows} == {"continuous"}
        for step in range(1, 65):
            assert schedule(seed, step, "G11") == schedule(seed, step, "G11S")
    a = axes(48, "train_gap", SEEDS[0], 1, 8)
    assert np.array_equal(a, axes(48, "train_gap", SEEDS[0], 1, 8))
    assert not np.array_equal(a, axes(48, "train_gap", SEEDS[0], 2, 8))
    codes = np.arange(8)
    c = supplied_coordinates(a, "G11", codes)
    s = supplied_coordinates(a, "G11S", codes)
    assert not np.array_equal(c, s)
    assert np.array_equal(c[:, 0, 0], s[:, 0, 0])
    assert np.array_equal(c[:, -1, -1], s[:, -1, -1])
    continuous = axes(48, "continuous", SEEDS[0], 1, 8)
    sc = supplied_coordinates(continuous, "G11S", codes)
    cc = supplied_coordinates(continuous, "G11", codes)
    assert np.array_equal(cc, sc)
    for width in WIDTHS:
        slots = auxiliary_slot_counts(TOKENS // width**2, 1, 1)
        assert slots.sum() == AUX_SLOTS and slots.max() - slots.min() <= 1
        assert slots.min() > 0 and slots.max() <= 64
    clean = rng(1, 2, "fixture").integers(0, 2, (4, 48, 48))
    lo, hi = observations(clean, 1, 1, 2), observations(clean, 1, 1, 32)
    assert np.array_equal(lo["queries"], hi["queries"])
    assert np.array_equal(lo["evidence"], hi["evidence"][:, :2])
    for value in (lo, hi):
        assert (value["noisy"][np.arange(4)[:, None], value["queries"]] == 2).all()
        assert np.array_equal(value["labels"], clean.reshape(4, -1)[np.arange(4)[:, None], value["queries"]])
    for _, width, kind in GEOMETRIES:
        for k in evaluation_ks(width):
            assert 0 <= k <= width**2 - 64
        axes(width, kind, 1, 1, 1)
    # Checks the sign/order convention independently of statistical code.
    fixture = dict(G00=0., G10=2., G01=3., G11=10.)
    assert fixture["G11"] - fixture["G10"] - fixture["G01"] + fixture["G00"] == 5
    return dict(status="passed_cpu_design_only", plan=plan(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                checks=["same_span_different_gap_sets", "orthogonal_32_step_schedule",
                        "equal_tokens_and_total_auxiliary_slots", "G11_G11S_shared_schedule", "deterministic_axes",
                        "native_span_encoding", "D4_endpoints", "nested_hidden_queries",
                        "labels_unchanged", "valid_parent_spans", "factorial_signs"],
                not_tested=["training_runner", "GPU_update", "checkpoint_resume", "MC_pipeline",
                            "bootstrap", "generation", "throughput", "full_pipeline"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = self_test()
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
