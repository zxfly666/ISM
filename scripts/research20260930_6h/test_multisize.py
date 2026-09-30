"""Read-only CPU data/statistic unit tests; no formal model training."""
import unittest
import numpy as np
import sm6_common as c
import sm6_data as d


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
        for step in (1, 2, 3, 4, 8000):
            records = [data.batch(93061, step, arm) for arm in 'NW']
            self.assertEqual(len({meta['paired_data_digest'] for _, meta in records}), 1)
            self.assertEqual(len({meta['allocated_tokens'] for _, meta in records}), 1)
            for parts, meta in records:
                for b in parts:
                    d.validate_view(b)
            self.assertTrue(np.all(records[1][0][0]['t'] == .75))
        self.assertNotEqual(data.batch(93061, 3, 'N')[1]['actual_input_digest'], data.batch(93061, 3, 'W')[1]['actual_input_digest'])

    def test_no_k1_expansion(self):
        b = d.parent_bank()
        with self.assertRaises(AssertionError):
            d.view(d.subset(b, np.flatnonzero(b['k'] == 1)), 8, 'N')

    def test_metric_oracle_and_saturation(self):
        y = np.array([.2, .5, .9])
        logits = np.stack([np.log1p(-y), np.log(y)], 1)
        self.assertLess(c.summary(logits, y)['max_kl'], 2e-12)
        s = c.summary(np.array([[1000., -1000.]]), np.array([.5]))
        self.assertGreater(s['mean_kl'], 900)
        self.assertTrue(np.isfinite(s['mean_kl']))

    def test_bank_manifest_and_query(self):
        specs = list(d.evaluation_specs())
        self.assertEqual(len(specs), 34)
        self.assertEqual(len({s['key'] for s in specs}), 34)
        for spec in specs:
            for arm in 'NW':
                b = d.bank_for_spec(spec, arm)
                d.validate_view(b)
        primary = d.bank_for_spec(next(s for s in specs if s['key'] == 'G8_test_N20_center'), 'W')
        self.assertEqual(len(primary['target']), 168)
        self.assertEqual(primary['tokens'].shape, (168, 1, 400))

    def test_manifest_exact_plan(self):
        import sm6_evaluation as ev
        jobs=list(ev.jobs())
        self.assertEqual(len(jobs),600)
        self.assertEqual(len({ev.job_key(j) for j in jobs}),600)
        self.assertEqual(sum(len(ev.input_bank(j)['query']) for j in jobs),129600)
        planned=c.read(c.ROOT/'artifacts/dense_multisize_canonical_6h_20260930_design/evaluation_manifest.json')
        def ident(j):
            spec=j.get('spec',j)
            return (j['seed'],j['arm'],j['step'],j['weight'],j['kind'],spec['family'],spec['split'],spec['side'],spec['placement'])
        self.assertEqual({ident(j) for j in jobs},{ident(j) for j in planned['jobs']})
        controls=[(meta,b,a) for seed in c.CFG['seed_labels'] for arm in 'NW' for meta,b,a in ev.control_banks(seed,arm)]
        self.assertEqual(len(controls),120);self.assertEqual(sum(len(b['query']) for _,b,_ in controls),13920)
        self.assertEqual({(x['seed'],x['arm'],x['family'],x['side'],x['kind']) for x,_,_ in controls},
                         {(x['seed'],x['arm'],x['family'],x['side'],x['kind']) for x in planned['controls']})

    def test_schedule_and_margins(self):
        import sm6_model as m
        import sm6_statistics as st
        self.assertAlmostEqual(m.lr(512),3e-4)
        self.assertAlmostEqual(m.lr(8000),3e-5)
        with self.assertRaises(AssertionError):m.lr(8001)
        result=st.paired([-.01]*6)
        self.assertAlmostEqual(result['upper'],-.01)
        self.assertEqual(c.CFG['training']['snapshot_steps'],[4000,6000])
        self.assertEqual(list(c.CFG['arms']),['N','W'])
        self.assertEqual(c.CFG['budget']['hard_seconds'],21600)


if __name__ == '__main__':
    unittest.main()
