"""End-to-end software fixtures, explicitly not scientific model results.

CPU mode checks all output identities and statistical/plot plumbing with a
deterministic synthetic predictor. GPU mode exercises the actual predictor on
every native input family, caching only within this labelled software fixture.
Neither creates a formal checkpoint or uses the new confirmation MC seed.
"""
from pathlib import Path
import copy
import numpy as np
import intervention_common as c
import intervention_data as d


def sliced(bank, count):
    n = len(bank['t'])
    return {k: v[:count].copy() if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v
            for k, v in bank.items()}


def synthetic_prediction(b):
    # Deliberately non-scientific; constant output respects all negative controls.
    p = np.full(b['labels'].shape, .5)
    out = dict(probability=p, common_data_hash=np.array(d.common_hash(b)),
        actual_input_hash=np.array(c.array_hash(b['noisy'], b['input_coordinates'], b['t'], b['queries'])),
        parent=b['parent'], chain=b['chain'], target_type=b['target_type'], software_scratch=np.array(True))
    if str(b['target_type']) != 'reference_only':
        y = b['labels'].astype(float)
        out.update(labels=y, ce=np.full_like(y, np.log(2)), brier=np.full_like(y, .25))
        if str(b['target_type']) == 'soft':
            out.update(kl=np.log(2)+y*np.log(y)+(1-y)*np.log1p(-y), probability_error=p-y)
    return out


def run(root, check=lambda: None, *, models=None):
    import intervention_pipeline as pipe
    import intervention_training as tr
    import intervention_analysis as analysis
    from ism_diffusion.scale_data import load_parent_split, ParentSplit
    root = Path(root)
    if 'scratch' not in root.name:
        raise ValueError('Software outputs must be isolated and labelled')
    root.mkdir(exist_ok=False)
    c.write(root/'software_only.json', dict(scientific_results=False,
        predictor='synthetic_constant' if models is None else 'two_shared_fixture_models',
        independent_trained_models=0, confirmation_parents=0))
    original = load_parent_split(c.DATA, 'train')
    parent = ParentSplit(original.spins[:4].copy(), np.array([0, 0, 1, 1]), 1024, {})
    del original
    pipe.copy_blueprint(root)
    pipe.make_banks(parent, root, check, scratch=True)
    low = d.low_reference(parent, c.load(root/'design/low_layouts.npz'), check)
    c.save(root/'reference/low_joint_counts.npz', **low)
    val = d.validation_banks(load_parent_split(c.DATA, 'val'))
    val = {name: sliced(b, 4) for name, b in val.items()}
    for name, b in val.items():
        c.save(root/'validation_banks'/(name+'.npz'), **b)
    cache = {}
    def prediction(mode, category, name, bank):
        key = mode, category, name
        if key not in cache:
            cache[key] = (synthetic_prediction(bank) if models is None else
                          tr.predict(models[mode], bank, check=check))
        return {**cache[key], 'software_scratch': np.array(True)}
    for ident in c.cells():
        for step in c.COHORTS[ident['cohort']]['validation_steps']:
            for name, bank in val.items():
                check()
                pred = prediction(ident['attention_mode'], 'validation', name, bank)
                c.save(root/'training'/ident['cell']/'validation'/f'step_{step}'/(name+'.npz'), **pred)
    for ident in c.evaluation_identities():
        categories = [('core', root/'banks')]
        if ident['kind'] != 'intermediate':
            categories += [(x, root/'diagnostic_banks'/x) for x in ('mechanism', 'padding', 'oracle', 'low')]
        for category, folder in categories:
            for path in sorted(folder.glob('*.npz')):
                if path.stem.endswith('_cpu_baselines'):
                    continue
                check()
                pred = prediction(ident['attention_mode'], category, path.stem, c.load(path))
                c.save(root/'evaluation'/ident['cell']/category/path.name, **pred,
                       model_cell=np.array(ident['cell']), snapshot=np.array(ident['snapshot']))
    counts = dict(validation=len(list((root/'training').glob('*/validation/*/*.npz'))),
                  evaluation=len(list((root/'evaluation').glob('*/*/*.npz'))))
    if counts != dict(validation=768, evaluation=888):
        raise RuntimeError(('Software output counts', counts))
    analysis.run(root, check, software_scratch=True)
    c.write(root/'complete.json', dict(status='passed_software_only', counts=counts,
        actual_distinct_prediction_calls=len(cache) if models is not None else 0,
        actual_inputs=(sum(len(c.load(p)['t']) for folder in [root/'banks', *list((root/'diagnostic_banks').iterdir()), root/'validation_banks']
                          for p in folder.glob('*.npz') if not p.stem.endswith('_cpu_baselines'))*2 if models is not None else 0)))
    return root


if __name__ == '__main__':
    import argparse, sys
    sys.path.insert(0, str(c.ROOT))
    sys.path.insert(0, str(c.ROOT/'scripts/research20260921'))
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    run(p.parse_args().root)
