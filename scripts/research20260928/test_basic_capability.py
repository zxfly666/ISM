"""CPU regression checks. Full inference/learning is in the separate runner."""
import unittest

import numpy as np

import capability_common as c
import exact_ising as e


class ExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = e.enumerate_ising()
        cls.bank, cls.split = e.make_bank(cls.truth)

    def test_partition_and_conditionals(self):
        self.assertEqual(e.truth_checks(self.truth, self.bank)["status"], "passed")

    def test_independent_zero_beta_truth(self):
        truth = e.enumerate_ising(side=2, beta=0)
        p, mass = e.conditional_table(truth, 0, [1, 2])
        np.testing.assert_array_equal(p, np.full(4, .5))
        np.testing.assert_array_equal(mass, np.full(4, .25))
        self.assertAlmostEqual(float(truth["log_partition"]), 4*np.log(2), places=12)

    def test_no_query_leakage(self):
        with self.assertRaises(ValueError):
            e.conditional_table(self.truth, 5, [5])
        self.assertTrue(e.validate_bank(self.bank))

    def test_geometry_split_not_symbol_split(self):
        for orbit in np.unique(self.bank["orbit"]):
            self.assertEqual(len(np.unique(self.bank["train"][self.bank["orbit"] == orbit])), 1)

    def test_D4_canonical_identity(self):
        q, observed = 5, [0, 11]
        key = e.geometry_key(q, observed)
        for g in range(8):
            self.assertEqual(e.geometry_key(e.transform_index(q, 4, g),
                [e.transform_index(x, 4, g) for x in observed]), key)

    def test_exact_metrics(self):
        y = np.array([.1, .5, .9])
        m = c.metrics(y, y)
        np.testing.assert_allclose(m["kl"], 0, atol=1e-14)
        np.testing.assert_allclose(m["brier"], y*(1-y), atol=1e-14)
        np.testing.assert_array_equal(m["excess_brier"], np.zeros(3))

    def test_local_gibbs_extension(self):
        a, b = e.k4_extension(self.bank), e.k4_extension(self.bank, True)
        np.testing.assert_array_equal(a["tokens"], b["tokens"])
        np.testing.assert_array_equal(a["target"], b["target"])
        self.assertTrue((a["t"] == .75).all())
        self.assertTrue((b["t"] == .9375).all())
        for row, q in enumerate(a["query"]):
            self.assertEqual(a["tokens"][row, 0, q], 2)
            self.assertEqual(set(np.flatnonzero(a["tokens"][row, 0] < 2)), set([q-8, q-1, q+1, q+8]))

    def test_stream_role_and_repeat(self):
        a = c.stream(92811, 1, "data").integers(2**30, size=8)
        np.testing.assert_array_equal(a, c.stream(92811, 1, "data").integers(2**30, size=8))
        self.assertFalse(np.array_equal(a, c.stream(92811, 1, "model").integers(2**30, size=8)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
