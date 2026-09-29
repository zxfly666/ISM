"""Read-only CPU data/statistic unit tests; no formal model training."""
import unittest
import numpy as np
import sg_common as c
import sg_data as d


class DataTests(unittest.TestCase):
    def test_exact_enumeration(self):
        self.assertEqual(d.exact_truth_checks()['status'], 'passed')

    def test_split(self):
        b = d.cavity_bank()
        self.assertEqual(len(set(b['orbit_key'])), 72)
        self.assertEqual(b['split'][240], 'test')
        self.assertEqual(b['split'][85], 'test')

    def test_geometry_witness(self):
        b = d.cavity_bank()
        self.assertAlmostEqual(b['target'][85]-b['target'][240], .628539361054709)
        self.assertEqual(int((b['tokens'][85] == 1).sum()), 4)
        self.assertEqual(int((b['tokens'][240] == 1).sum()), 4)

    def test_pairing_and_shapes(self):
        data = d.TrainingData()
        for step in (1, 2, 3, 4, 24000):
            records = [data.batch(93001, step, arm) for arm in 'ABCD']
            self.assertEqual(len({meta['paired_data_digest'] for _, meta in records}), 1)
            self.assertEqual(len({meta['allocated_tokens'] for _, meta in records}), 1)
            for parts, meta in records:
                for b in parts:
                    d.validate_view(b)
            self.assertTrue(np.all(records[1][0][0]['t'] == .75))
        self.assertNotEqual(data.batch(93001, 3, 'A')[1]['actual_input_digest'], data.batch(93001, 3, 'C')[1]['actual_input_digest'])

    def test_no_k1_expansion(self):
        b = d.parent_bank()
        with self.assertRaises(AssertionError):
            d.view(d.subset(b, np.flatnonzero(b['k'] == 1)), 8, 'A')

    def test_metric_oracle_and_saturation(self):
        y = np.array([.2, .5, .9])
        logits = np.stack([np.log1p(-y), np.log(y)], 1)
        self.assertLess(c.summary(logits, y)['max_kl'], 2e-12)
        s = c.summary(np.array([[1000., -1000.]]), np.array([.5]))
        self.assertGreater(s['mean_kl'], 900)
        self.assertTrue(np.isfinite(s['mean_kl']))

    def test_bank_manifest_and_query(self):
        specs = list(d.evaluation_specs())
        self.assertEqual(len(specs), 64)
        self.assertEqual(len({s['key'] for s in specs}), 64)
        for spec in specs:
            for arm in 'ABCD':
                b = d.bank_for_spec(spec, arm)
                d.validate_view(b)
        primary = d.bank_for_spec(next(s for s in specs if s['key'] == 'G8_test_N20_center'), 'D')
        self.assertEqual(len(primary['target']), 168)
        self.assertEqual(primary['tokens'].shape, (168, 1, 400))


if __name__ == '__main__':
    unittest.main()
