"""CPU data/software tests, no held-size model forward."""
import unittest
import numpy as np
import ms_common as c
import ms_data as d


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = c.load(c.PARENT/"bank.npz")
        cls.views = {s: d.extend_k4(cls.bank, s) for s in c.CFG["evaluation_sides"]}

    def test_exact_bank_shape_split(self):
        b = self.bank
        self.assertEqual(len(b["query"]), 1864)
        self.assertEqual(int(b["train"].sum()), 1296)
        for k, n, nt in ((1,120,80),(2,1680,1152),(4,64,64)):
            self.assertEqual(int((b["k"] == k).sum()), n)
            self.assertEqual(int(((b["k"] == k) & b["train"]).sum()), nt)
        self.assertTrue(np.all(b["tokens"][np.arange(1864), 0, b["query"]] == 2))

    def test_all_local_truth_and_geometry(self):
        for s, v in self.views.items():
            self.assertTrue(d.validate_local(v))
            self.assertTrue(np.all(v["t"] == .75))
            self.assertEqual(v["tokens"].shape, (64,1,s*s))

    def test_natural_clock(self):
        for s in (6,8,12):
            b = d.extend_k4(self.bank, s, True)
            np.testing.assert_allclose(b["t"], 1-4/s**2, atol=4e-8, rtol=0)
            np.testing.assert_array_equal(b["target"], self.views[s]["target"])

    def test_physical_and_native_pairing(self):
        for seed in c.CFG["seed_labels"]:
            for step in (1,2,511,2048):
                ids = d.indices(self.bank, seed, step)
                self.assertTrue(self.bank["train"][ids].all())
                a = d.logical_and_native(self.bank, self.views, ids, 4)
                b = d.logical_and_native(self.bank, self.views, ids, d.view_side("B46", step))
                self.assertEqual(a[2], b[2])
                self.assertEqual(a[3] == b[3], step % 2 == 0)

    def test_schedule_no_held_size(self):
        a = [d.view_side("A4", s) for s in range(1,2049)]
        b = [d.view_side("B46", s) for s in range(1,2049)]
        self.assertEqual(a.count(4), 2048)
        self.assertEqual(b.count(4), 1024)
        self.assertEqual(b.count(6), 1024)
        self.assertTrue(set(a+b) == {4,6})

    def test_diagnostic_geometry(self):
        for order in ("near","far"):
            for scale in (1,2):
                previous = set()
                for n in (16,32,64):
                    b = d.diagnostic_cell(n,scale,.75,order)
                    points = set(map(tuple,b["coordinates"][0,0]))
                    self.assertTrue(previous <= points)
                    self.assertEqual(len(points), n)
                    previous = points

    def test_diagnostic_and_pad(self):
        banks, meta = d.diagnostic_banks()
        self.assertEqual(len(banks),26)
        for key in ("pad64_from16_near","pad64_from16_far"):
            self.assertTrue(d.validate_local(banks[key]))
            self.assertTrue((banks[key]["valid"].sum((1,2)) == 16).all())

    def test_metrics_exact(self):
        p = np.array([.01,.25,.5,.98])
        m = c.metrics(p,p)
        np.testing.assert_allclose(m["kl"],0,atol=1e-14)
        np.testing.assert_allclose(m["brier"],p*(1-p),atol=1e-14)
        self.assertTrue(c.summary(p,p)["ability_passed"])

    def test_rng_addressing(self):
        np.testing.assert_array_equal(c.rng(1,2,"x").integers(1000,size=20),c.rng(1,2,"x").integers(1000,size=20))
        self.assertFalse(np.array_equal(c.rng(1,2,"x").integers(1000,size=20),c.rng(1,2,"y").integers(1000,size=20)))


if __name__ == "__main__":
    unittest.main()
