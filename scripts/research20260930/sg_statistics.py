"""Six paired training seeds; exact test rows are not independent replicates."""
import itertools
import numpy as np
from scipy.stats import t as student_t
import sg_common as c


def paired(values, confidence=.95, one_sided=False):
    x = np.asarray(values, dtype=np.float64)
    assert x.shape == (6,) and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd/np.sqrt(6)
    quantile = confidence if one_sided else (1+confidence)/2
    half = float(student_t.ppf(quantile, 5)*se)
    return dict(estimate=mean, per_seed=x.tolist(), seed_sd=sd, mcse=float(se),
                confidence=confidence, one_sided=one_sided, ci=[mean-half, mean+half],
                upper=mean+half, n=6, independent_unit='training_seed')


def summarize(rows, controls):
    def select(seed, arm, key, step=24000, weight='raw', kind='final'):
        found = [r for r in rows if r['seed'] == seed and r['arm'] == arm and r['spec']['key'] == key
                 and r['step'] == step and r['weight'] == weight and r['kind'] == kind]
        assert len(found) == 1, (seed, arm, key, step, weight, kind)
        return found[0]
    seeds = c.CFG['seed_labels']
    primary_key = 'G8_test_N20_center'
    loss = {a: np.array([select(s, a, primary_key)['mean_kl'] for s in seeds]) for a in 'ABCD'}
    diff = loss['D']-loss['A']
    primary = paired(diff)
    r = np.random.default_rng(np.random.SeedSequence([c.CFG['bootstrap_seed'], 1]))
    draws = diff[r.integers(6, size=(20000, 6))].mean(1)
    primary['seed_bootstrap_ci95'] = np.quantile(draws, [.025, .975]).tolist()
    primary['bootstrap_endpoint_groups'] = [np.quantile(v, [.025, .975]).tolist() for v in np.array_split(draws, 10)]
    loo = []
    for i in range(6):
        v = np.delete(diff, i)
        mean = float(v.mean()); half = float(student_t.ppf(.975, 4)*v.std(ddof=1)/np.sqrt(5))
        loo.append(dict(omitted_seed=seeds[i], estimate=mean, ci95=[mean-half, mean+half]))
    primary['leave_one_out'] = loo
    flips = (np.array(list(itertools.product((-1., 1.), repeat=6)))*diff).mean(1)
    primary['sign_flip_two_sided_p'] = float(np.mean(np.abs(flips) >= abs(diff.mean())-1e-15))
    primary['practical_passed'] = primary['upper'] < -.005
    primary['directional_passed'] = primary['upper'] < 0
    primary['precision_target_met'] = primary['mcse'] <= .002
    primary['sensitivity_direction_stable'] = bool(all(v['estimate'] < 0 for v in loo) and primary['seed_bootstrap_ci95'][1] < 0)
    mechanisms = {name: paired(v, .99) for name, v in {
        'C-A': loss['C']-loss['A'], 'D-B': loss['D']-loss['B'], 'B-A': loss['B']-loss['A'],
        'D-C': loss['D']-loss['C'], '(D-C)-(B-A)': (loss['D']-loss['C'])-(loss['B']-loss['A'])}.items()}
    retention = {}
    for key in ('K1_holdout', 'K2_holdout', 'K4_N4_center'):
        delta = [select(s, 'D', key)['mean_kl']-select(s, 'A', key)['mean_kl'] for s in seeds]
        val = paired(delta, 1-.05/3, True)
        val['comparison_passed'] = val['upper'] <= .002
        val['all_D_absolute_passed'] = all(select(s, 'D', key)['ability_passed'] for s in seeds)
        retention[key] = val
    learning = []
    for seed in seeds:
        for arm in 'ABCD':
            sides = sorted(set(c.CFG['arms'][arm]['valid_side_cycle']))
            earlier = np.mean([select(seed, arm, f'G8_validation_N{side}_center', 16000, kind='learning')['mean_kl'] for side in sides])
            final = np.mean([select(seed, arm, f'G8_validation_N{side}_center')['mean_kl'] for side in sides])
            learning.append(dict(seed=seed, arm=arm, trained_sides=sides, kl_16k=float(earlier), kl_24k=float(final),
                improvement=float(earlier-final), learning_incomplete=bool(earlier-final > .002),
                train_absolute=all(select(seed, arm, f'G8_train_N{side}_center')['ability_passed'] for side in sides),
                validation_absolute=all(select(seed, arm, f'G8_validation_N{side}_center')['ability_passed'] for side in sides)))
    gates = dict(primary_practical=primary['practical_passed'], primary_absolute_D=all(select(s, 'D', primary_key)['ability_passed'] for s in seeds),
                 retention=all(v['comparison_passed'] and v['all_D_absolute_passed'] for v in retention.values()),
                 structural_controls=all(v['passed'] for v in controls),
                 D_trained_patterns=all(v['train_absolute'] for v in learning if v['arm'] == 'D'),
                 D_validation_patterns=all(v['validation_absolute'] for v in learning if v['arm'] == 'D'))
    gates['headline_joint_passed'] = gates['primary_practical'] and gates['primary_absolute_D'] and gates['retention'] and gates['structural_controls']
    return dict(primary=primary, mechanisms=mechanisms, retention=retention, learning_flags=learning, gates=gates,
                primary_arm_loss={a: v.tolist() for a, v in loss.items()}, rows=rows, controls=controls,
                no_generation=True, local_conditional_not_full_joint_or_teacher_idea=True)
