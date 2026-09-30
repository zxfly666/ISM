"""Pure CPU plan arithmetic/toy checks. NOT a scientific runner or GPU preflight.

Reads only the draft JSON. Does not read scientific data, load models, write files,
sample Ising fields, start network connections, or import torch.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'docs/research_reboot_20260921/GEOMETRY_IDENTIFICATION_CONFIG_20260926.json'


def increments(w, kind):
    q = w // 4 - 1
    if kind == 'train':
        return np.repeat([1, 2, 4, 8], [q+1, q, q+2, q])
    return np.repeat([3, 6], [3*q+3, q])


def axis(gaps):
    return np.r_[0, np.cumsum(gaps)]


def bernoulli_entropy(p):
    return -p*math.log(p)-(1-p)*math.log1p(-p)


def main():
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    nseed = len(c['training_seeds'])
    nmodel = nseed * len(c['arms'])
    counts = dict(models=nmodel, training_updates=nmodel*c['steps'],
                  windows_per_model=int(c['steps']*np.mean([c['tokens_per_update']//w**2 for w in c['widths']])),
                  mc_parents=c['mc']['chains']*c['mc']['parents_per_chain'])
    assert c['steps'] % 32 == 0
    assert sum(c['mc']['initial_states'].values()) == c['mc']['chains']
    assert len(set(c['training_seeds'] + list(c['role_seeds'].values()) + [c['mc']['seed']])) == 11
    by_width = {48: 0, 96: 0}
    parent_forwards = 0
    cost = 0.
    banks = 0
    for g in c['conditional_geometries']:
        w = g['width']
        assert len(g['ks']) == len(set(g['ks']))
        assert max(g['ks']) + c['queries_per_parent'] <= w*w
        copies = 1 if g['kind'] == 'continuous' else 4
        nfiles = len(g['ks']) * nseed * (2 + copies)
        by_width[w] += nfiles
        banks += len(g['ks'])
        parent_forwards += nfiles*g['parents']
        cost += nfiles*g['parents']*c['historical_timing']['prediction_seconds_per_parent'][str(w)]
    counts.update(banks=banks, model_bank_cells=banks*nmodel,
                  prediction_files=sum(by_width.values()), prediction_files_w48=by_width[48],
                  prediction_files_w96=by_width[96], conditional_parent_forwards=parent_forwards)
    npairs = sum(c['low_k']['layout_pairs'].values())
    counts.update(low_k_layout_pairs=npairs,
                  low_k_input_forwards=npairs*2*6*nseed*(2+4),
                  padding_cells=nmodel*len(c['padding']['ks'])*len(c['padding']['views']),
                  generation_images=nmodel*c['generation']['samples_per_model'],
                  generation_shards=nmodel*c['generation']['samples_per_model']//c['generation']['shard_size'])
    assert counts == c['expected_counts'], (counts, c['expected_counts'])
    # Deterministic layout checks; no Monte Carlo physics or model predictions.
    random = np.random.default_rng(273151)
    for kind, near, far in [('train', 1, 8), ('held', 3, 6)]:
        gaps = increments(48, kind).tolist()
        gaps.remove(near); gaps.remove(far)
        g = np.empty(47, dtype=int)
        rest = [i for i in range(47) if i not in (22, 23)]
        g[rest] = random.permutation(gaps)
        g[22], g[23] = near, far
        changed = g.copy(); changed[[22, 23]] = changed[[23, 22]]
        a, b = axis(g), axis(changed)
        assert a[-1] == b[-1] == 174
        assert a[22] == b[22] and a[25] == b[25] and a[23] != b[23]
        negative_base = np.empty(47, dtype=int)
        negative_base[[2, 3]] = [near, far]
        negative_base[[i for i in range(47) if i not in (2, 3)]] = random.permutation(gaps)
        negative = negative_base.copy(); negative[[2, 3]] = negative[[3, 2]]
        assert not np.array_equal(axis(negative), axis(negative_base))
        assert np.array_equal(axis(negative)[[22, 23, 25]], axis(negative_base)[[22, 23, 25]])
    for w in c['widths'] + [96]:
        assert len(increments(w, 'train')) == w-1
        assert increments(w, 'train').sum() == increments(w, 'held').sum() < 512
    # K2 unequal-geometry-weight fixture, from a valid sign-symmetric joint law.
    # Joint p(y,v1,v2)=1/8*(1+y*v1*Cq1+y*v2*Cq2+v1*v2*C12).
    corr = [(0.4, 0.2, 0.3), (0.1, 0.1, 0.05)]
    states = [-1, 1]
    totals = []
    for cq1, cq2, c12 in corr:
        row = {(y, v1, v2):(1+y*v1*cq1+y*v2*cq2+v1*v2*c12)/8
               for y in states for v1 in states for v2 in states}
        assert min(row.values()) >= 0 and abs(sum(row.values())-1) < 1e-14
        totals.append(row)
    posteriors_differ = False
    for v1 in states:
        for v2 in states:
            pv = np.array([sum(row[y, v1, v2] for y in states) for row in totals])
            p = np.array([row[1, v1, v2]/z for row, z in zip(totals, pv)])
            weight = pv/pv.sum()
            mixture = sum(row[1, v1, v2] for row in totals)/pv.sum()
            assert abs(mixture-float(weight@p)) < 1e-14
            posteriors_differ |= abs(mixture-p.mean()) > 1e-5
            assert bernoulli_entropy(mixture)-sum(weight[j]*bernoulli_entropy(p[j]) for j in range(2)) >= -1e-14
    assert posteriors_differ
    hist = c['historical_timing']
    training = 2*hist['training_six_fine_models_seconds'] + hist['training_six_span_models_seconds']
    padding = c['padding']['parents']*nmodel*3*(hist['prediction_seconds_per_parent']['48']+2*hist['prediction_seconds_per_parent']['96'])
    lowk = counts['low_k_input_forwards']*hist['prediction_seconds_per_parent']['48']
    generation = counts['generation_shards']*hist['generation_shard_seconds']
    raw_components = dict(training=training, conditional=cost, padding=padding,
                          low_k_forward=lowk, generation=generation,
                          validation_io_cpu_allowance=1800.)
    raw = sum(raw_components.values())
    planning = .5*3600 + 1.2*raw + c['budget']['statistics_reserve_seconds']
    print(json.dumps(dict(status='cpu_design_arithmetic_and_toy_checks_passed_only',
                         config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                         counts=counts, budget_raw_components_seconds=raw_components,
                         historical_extrapolation_hours=dict(training=training/3600,
                         conditional=cost/3600, padding=padding/3600, low_k_forward=lowk/3600,
                         generation=generation/3600, total_with_reserves_but_no_mc_tail=planning/3600),
                         gpu_preflight_passed=False, experimental_runner_implemented=False,
                         new_training_started=False, new_mc_started=False), indent=2))


if __name__ == '__main__':
    main()
