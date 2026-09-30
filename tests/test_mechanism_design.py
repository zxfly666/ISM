import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1]/"scripts/research20260921/mechanism_design.py"
spec = importlib.util.spec_from_file_location("mechanism_design", path)
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


def test_nested_subsets_preserve_information_and_span():
    for visible in (1, 19*48, 38, 47*48):
        for policy in ("uniform", "near", "far"):
            previous = set()
            for count in (256, 576, 1024, 2304):
                ids = design.ordered_subset(48, visible, count, policy, 2)
                assert len(ids) == len(set(ids)) == count
                assert list(ids[:2]) == [0, visible]
                assert previous.issubset(set(ids))
                assert {0, 47, 2256, 2303}.issubset(set(ids))
                previous = set(ids)


def test_masks_nested_and_reproducible():
    order = design.nested_visibility(8, 2304, 0, 0)
    assert np.array_equal(order, design.nested_visibility(8, 2304, 0, 0))
    previous = np.zeros_like(order, bool)
    for count in (1, 2, 8, 32):
        mask = design.exact_visible_mask(order, count)
        assert np.all(mask.sum(1) == count)
        assert np.all(~previous | mask)
        previous = mask


def test_frequency_period_and_kl():
    freq = design.axis_frequencies()
    assert len(freq) == 8
    assert np.isclose(2*np.pi/freq[3], 198.69176531592203)
    assert np.allclose(design.bernoulli_kl([.3, .8], [.3, .8]), 0)


def test_scores_only_use_masked_targets():
    y = np.array([[0, 1, 1]])
    m = np.array([[True, True, False]])
    first = design.per_image_scores([[.2, .8, .01]], y, m)
    second = design.per_image_scores([[.2, .8, .99]], y, m)
    assert np.isclose(first["ce"][0], -np.log(.8))
    assert np.isclose(first["brier"][0], .04)
    assert np.array_equal(first["ce"], second["ce"])
