import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("diagnostic_math", Path(__file__).resolve().parents[1] / "scripts/research20260921/diagnostic_math.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_fft_equals_direct_and_avoids_periodic_wrap():
    x = np.random.default_rng(813).choice([-1, 1], (7, 11, 11))
    g, count = m.uniform_axis_fft(x)
    np.testing.assert_allclose(g, m.direct_axis(x), atol=1e-14)
    assert count[-1] == 22
    assert np.allclose(g[:, 0], 1)


def test_single_site_full_boundary():
    x = np.ones((5, 5))
    b, p, _ = m.cavity_distribution(x, [(2, 2)], .44)
    expected = 1/(1+np.exp(-2*.44*4))
    np.testing.assert_allclose(p[1], expected)


def test_sequential_oracle_exact_parallel_biased_and_kl_identity():
    x = np.random.default_rng(9).choice([-1, 1], (8, 8))
    sites = [(3, 3), (3, 4), (4, 3), (4, 4)]
    bits, p, _ = m.cavity_distribution(x, sites, .4406867935)
    for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
        stages = m.conditional_stages(bits, p, [[i] for i in order])
        q = m.joint_from_conditionals(bits, stages, [s["probability"] for s in stages])
        np.testing.assert_allclose(p, q, atol=1e-12)
    stages = m.conditional_stages(bits, p, [[0, 1], [2, 3]])
    oracle = m.joint_from_conditionals(bits, stages, [s["probability"] for s in stages])
    assert m.divergence(p, oracle)["kl"] > 0
    learned = [np.clip(.9*s["probability"]+.04, .001, .999).astype(np.float32) for s in stages]
    _, parts = m.kl_decomposition(bits, p, stages, learned, oracle)
    assert abs(parts["identity_residual"]) < 1e-10


def test_spin_flip_and_zero_beta():
    x = np.random.default_rng(5).choice([-1, 1], (7, 7))
    sites = [(2, 2), (2, 3), (3, 2), (3, 3)]
    _, p, _ = m.cavity_distribution(x, sites, .44)
    bits, other, _ = m.cavity_distribution(-x, sites, .44)
    np.testing.assert_allclose(p, other[::-1])
    bits, p0, _ = m.cavity_distribution(x, sites, 0.)
    stages = m.conditional_stages(bits, p0, [list(range(4))])
    q = m.joint_from_conditionals(bits, stages, [s["probability"] for s in stages])
    np.testing.assert_allclose(q, p0)


def test_cavity_matches_full_field_energy_enumeration():
    full = np.random.default_rng(55).choice([-1, 1], (7, 7))
    sites = [(2+i, 2+j) for i in range(3) for j in range(3)]
    bits, p, _ = m.cavity_distribution(full, sites, .4406867935)
    states = np.repeat(full[None], len(bits), axis=0)
    for i, (x, y) in enumerate(sites):
        states[:, x, y] = 2*bits[:, i]-1
    score = ((states[:, 1:, :]*states[:, :-1, :]).sum((1, 2))
             +(states[:, :, 1:]*states[:, :, :-1]).sum((1, 2)))
    weights = np.exp(.4406867935*(score-score.max()))
    np.testing.assert_allclose(p, weights/weights.sum(), atol=1e-14)
