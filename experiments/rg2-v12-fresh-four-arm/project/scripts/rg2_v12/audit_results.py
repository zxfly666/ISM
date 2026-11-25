"""Independent CPU checks of saved probabilities, spins, and physical statistics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, atol=2e-12):
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def physical(fields):
    """Direct integer point-pair sums; no FFT or experiment-code imports."""
    n, w, h = fields.shape
    assert h == w and np.isin(fields, [-1, 1]).all()
    total = np.empty((n, w // 2 + 1), np.int64)
    count = 2 * w * (w - np.arange(w // 2 + 1))
    total[:, 0] = count[0]
    for r in range(1, w // 2 + 1):
        total[:, r] = ((fields[:, r:, :] * fields[:, :-r, :]).sum((1, 2), dtype=np.int64)
                       + (fields[:, :, r:] * fields[:, :, :-r]).sum((1, 2), dtype=np.int64))
    return total / count, fields.mean((1, 2)), -total[:, 1] / count[1]


def coarse(x):
    n, w, _ = x.shape
    return np.sign(x.reshape(n, w // 3, 3, w // 3, 3).sum((2, 4))).astype(np.int8)


def exact(project, root, entries):
    count = 0
    for e in entries:
        for w in (48, 96):
            for m in (1, 8, 32):
                name = f'fine_W{w}_M{m}.npz'
                bankpath = project / 'inputs/banks' / name
                bank = load(bankpath)
                d = load(root / 'A0_exact' / e['key'] / name)
                assert str(d['source_sha']) == e['source_sha256']
                assert str(d['bank_sha']) == sha(bankpath)
                for key, bk in [('parent', 'parent'), ('chain', 'chain_ids'), ('origin', 'origin')]:
                    np.testing.assert_array_equal(d[key], bank[bk])
                p, target, y = d['p'], bank['target'], bank['labels']
                assert p.shape == (256, m) and ((p > 0) & (p < 1)).all()
                close(d['KL'], (target * np.log(target / p) + (1-target) * np.log((1-target)/(1-p))).mean(1))
                close(d['CE'], -(y * np.log(p) + (1-y) * np.log1p(-p)).mean(1))
                count += 1
    return count


def retention(runs, entries):
    root = runs / 'A_supp'
    inputs = json.loads((root / 'inputs.json').read_text())
    banks = sorted(k for k in inputs['files'] if k.startswith('banks/'))
    assert len(banks) == 35
    baselines, count = {}, 0
    for e in entries:
        for bank in banks:
            d = load(root / 'A0_retention' / e['key'] / Path(bank).name)
            assert str(d['ema_sha']) == e['sha256'] and str(d['bank_sha']) == inputs['files'][bank]
            p, y = d['p'], d['labels']
            assert p.shape == y.shape and len(p) == 256
            assert ((p > 0) & (p < 1)).all() and np.isin(y, [0, 1]).all()
            close(d['CE'], -(y * np.log(p) + (1-y) * np.log1p(-p)).mean(1))
            identity = (d['parent'], d['chain'], y)
            if bank in baselines:
                for left, right in zip(identity, baselines[bank]):
                    np.testing.assert_array_equal(left, right)
            else:
                baselines[bank] = identity
            count += 1
    return count


def blocks(runs, entries):
    root = runs / 'A_supp'
    ex = load(root / 'a2_exteriors.npz')
    assert len(ex['labels']) == 33
    count, worst = 0, 0.
    beta = np.log1p(np.sqrt(2)) / 2
    for e in entries:
        for label, exterior in zip(ex['labels'], ex['fields']):
            d = load(root / 'A2' / e['key'] / (str(label) + '.npz'))
            assert str(d['ema_sha']) == e['sha256']
            np.testing.assert_array_equal(exterior, d['exterior'])
            s = (2*d['bits'].astype(np.int64)-1).reshape(512, 3, 3)
            bond = (s[:, 1:] * s[:, :-1]).sum((1, 2)) + (s[:, :, 1:] * s[:, :, :-1]).sum((1, 2))
            bond += (s[:, 0, :] * exterior[5, 6:9]).sum(1) + (s[:, -1, :] * exterior[9, 6:9]).sum(1)
            bond += (s[:, :, 0] * exterior[6:9, 5]).sum(1) + (s[:, :, -1] * exterior[6:9, 9]).sum(1)
            p = np.exp(beta * (bond - bond.max())); p /= p.sum()
            close(d['truth'], p)
            close(d['serial_oracle'], p)
            close(d['control_P'], d['control_oracle'])
            for plan, first in [('serial', 1), ('adjacent', 3), ('checker', 3), ('nine', 9)]:
                q, oracle = d[plan+'_Q'], d[plan+'_oracle']
                assert (q > 0).all() and (oracle > 0).all()
                close(q.sum(), 1., 1e-8); close(oracle.sum(), 1.)
                r = json.loads(str(d[plan+'_result']))
                kl = np.sum(p * np.log(p/q)); factor = np.sum(p * np.log(p/oracle))
                close(r['learned_kl'], kl, 1e-8)
                close(r['parallel_factorization_kl'], factor, 1e-8)
                close(r['single_conditional_kl'] + factor, kl, 1e-8)
                close(r['TV'], .5 * abs(p-q).sum())
                close(r['q_majority'], q[d['bits'].sum(1) >= 5].sum())
                expected_clock = [max(9/256, .01)] + [max(m/256, .01) for m in range(9-first, 0, -1)]
                close(r['clock'], expected_clock)
                worst = max(worst, abs(r['single_conditional_kl'] + factor - kl))
            count += 1
    return dict(cases=count, largest_KL_identity_residual=worst,
                independent_Hamiltonian_truth=True, normalized=True, state_clock_verified=True)


def generation(project, root, entries):
    count, metrics_count, largest_error = 0, 0, 0.
    data = load(project / 'inputs/mc_W144_centered.npz')
    ids = np.concatenate([np.flatnonzero(data['chain'] == c)[:32] for c in range(1, 16, 2)])
    mc = 2*np.unpackbits(data['fine_bits'][ids], axis=-1, bitorder='little').astype(np.int8)-1
    mc = mc[:, 24:120, 24:120]
    references = {}
    for w, x in [(96, mc), (32, coarse(mc))]:
        saved = load(root / f'MC_reference_W{w}.npz')
        np.testing.assert_array_equal(saved['parent_ids'], ids)
        values = physical(x)
        for key, value in zip(('g', 'm', 'e'), values):
            close(saved[key], value)
        references[w] = values
    summary = json.loads((root / 'A1.json').read_text())
    summaries = {(r['model'], r['steps'], r['path']): r for r in summary['arms']}
    for e in entries:
        for w in ((96, 32) if e['arm'] == 'fine_plus_rg' else (96,)):
            for k in (64, 256):
                samples = []
                for i in range(32):
                    d = load(root / 'samples' / e['key'] / f'W{w}_K{k}' / f'{i:05d}.npz')
                    for key, expected in [('sample_id', i), ('width', w), ('steps', k),
                                          ('training_seed', e['seed']), ('level', int(w == 32)), ('rng_root', 2026112501)]:
                        assert int(d[key]) == expected, (e['key'], key)
                    assert str(d['ema_sha']) == e['sha256'] and d['spins'].shape == (w, w)
                    samples.append(d['spins']); count += 1
                x = np.stack(samples)
                paths = [('fine', x), ('B', coarse(x))] if w == 96 else [('C', x)]
                for name, fields in paths:
                    saved = load(root / 'metrics' / e['key'] / f'{name}_K{k}.npz')
                    values = physical(fields)
                    for key, value in zip(('g', 'm', 'e'), values):
                        largest_error = max(largest_error, float(abs(saved[key] - value).max()))
                        close(saved[key], value)
                    width = fields.shape[-1]; lo = width // 4 + 1
                    reference = references[width][0].mean(0)[lo:]
                    error = np.linalg.norm(values[0].mean(0)[lo:] - reference) / np.linalg.norm(reference)
                    close(summaries[e['key'], k, name]['G_error'], error)
                    metrics_count += 1
    return dict(samples=count, metric_cells=metrics_count, largest_direct_pair_sum_difference=largest_error,
                per_sample_identity_verified=True, same_image_majority_verified=True,
                MC_reference='odd historical chains, first 32 each; 256 fields', inference='descriptive')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--stage', choices=['A', 'B'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    if args.stage == 'A':
        allentries = json.loads((args.project/'inputs/manifest.json').read_text())['models']
        entries = [e for e in allentries if e['stage'] != 'midpoint']
        results = dict(exact_cells=exact(args.project, args.runs/'A_v2', allentries),
                       retention_cells=retention(args.runs, allentries), A2=blocks(args.runs, entries),
                       generation=generation(args.project, args.runs/'A_v2', entries))
        assert (results['exact_cells'], results['retention_cells'], results['A2']['cases'], results['generation']['samples']) == (126, 735, 396, 960)
    else:
        entries = json.loads((args.runs/'B_eval/identity.json').read_text())['models']
        results = dict(exact_cells=exact(args.project, args.runs/'B_eval', entries),
                       generation=generation(args.project, args.runs/'B_eval', entries),
                       training_audit=json.loads((args.runs/'B_eval/training_audit.json').read_text()))
        assert results['exact_cells'] == 48 and results['generation']['samples'] == 512
        assert results['training_audit']['status'] == 'passed'
    results.update(stage=args.stage, status='passed', seconds=time.monotonic()-start,
                   audit_source_sha256=sha(Path(__file__)), independent_implementation='NumPy integer axis-pair sums and direct 3x3 Hamiltonian')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
