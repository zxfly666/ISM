"""Independent small enumerations of paired inference and low-K pooling."""
import unittest
import numpy as np
import intervention_common as c
import intervention_statistics as s
from test_intervention_cpu import archived_blueprint


class StatisticsTests(unittest.TestCase):
    def test_circular_block_weights(self):
        chain = np.repeat(np.arange(16), 32)
        for block in (1, 2, 4, 8, 16):
            w = s.block_weights(chain, c.rng(99, block, "software_weights"), block)
            self.assertAlmostEqual(w.sum(), 1.)
            self.assertTrue(np.all(w >= 0))

    def test_shared_parent_and_independent_seed_weights(self):
        chain = np.repeat(np.arange(16), 16)
        x = np.broadcast_to(np.arange(6)[:, None, None], (6, 1, 256)).copy()
        a, b = s.resample([x, x], chain, 64, 8, "software_groups")
        self.assertFalse(np.array_equal(a, b))
        y = np.broadcast_to(np.arange(256)[None, None, :], (6, 1, 256)).copy()
        a, b = s.resample([y, y], chain, 64, 8, "software_shared_MC")
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        z = np.concatenate([y, y+.037], axis=1)
        draw = s.resample([z], chain, 64, 8, "software_paired")[0]
        np.testing.assert_allclose(draw[:, 1]-draw[:, 0], .037, atol=1e-12)
        np.testing.assert_array_equal(draw, s.resample([z], chain, 64, 8, "software_paired")[0])

    def test_low_k_full_enumeration(self):
        layout, _ = archived_blueprint()
        grouping, keys = s.geometry_groups(layout)
        counts = np.random.default_rng(1798).integers(0, 30, (80, 2, 8))
        ref = s.low_distribution(counts, grouping)
        truth, mass = ref["pair_probability"], ref["pattern_mass"]
        patterns = [(1,), (0,), (1, 1), (1, 0), (0, 1), (0, 0)]
        num, den = np.zeros_like(mass), np.zeros_like(mass)
        for pair in range(80):
            for side in range(2):
                sym = counts[pair, side]+counts[pair, side, ::-1]
                for cond, values in enumerate(patterns):
                    for y in (0, 1):
                        for v1 in (0, 1):
                            for v2 in (0, 1):
                                if v1 == values[0] and (len(values) == 1 or v2 == values[1]):
                                    weight = sym[y*4+v1*2+v2]/sym.sum()
                                    den[pair, side, cond] += weight
                                    num[pair, side, cond] += y*weight
        np.testing.assert_allclose(mass, den)
        np.testing.assert_allclose(truth, num/den)
        for g in range(len(keys)):
            selected = grouping == g
            np.testing.assert_allclose(ref["pooled_probability"][selected], num[selected].sum()/den[selected].sum())
        risk, references, _ = s.low_values(counts, grouping, np.full((5, 80, 2, 6), .5))
        self.assertEqual(risk.shape, (5, 80, 2, 2))
        self.assertEqual(references.shape, (80, 2, 2, 2))
        np.testing.assert_allclose(risk[..., 0], np.log(2))
        np.testing.assert_allclose(risk[..., 1], .25)

    def test_undefined_events_are_explicit(self):
        layout, _ = archived_blueprint()
        grouping, _ = s.geometry_groups(layout)
        counts = np.zeros((80, 2, 8)); counts[..., 0] = 256
        ref = s.low_distribution(counts, grouping)
        self.assertTrue(np.all(ref["undefined"][..., [3, 4]]))
        self.assertTrue(np.isnan(ref["pair_probability"][..., [3, 4]]).all())
        risk, _, _ = s.low_values(counts, grouping, np.full((5, 80, 2, 6), .5))
        self.assertTrue(np.isfinite(risk).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
