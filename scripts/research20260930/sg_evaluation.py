"""Manifest-driven final/early evaluation; primary data remain sealed until lock."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import sg_common as c
import sg_data as d
import sg_model as m


def jobs():
    specs = list(d.evaluation_specs())
    for seed in c.CFG['seed_labels']:
        for arm in 'ABCD':
            for weight in ('raw', 'ema'):
                for spec in specs:
                    if weight == 'ema' and spec.get('raw_only'):
                        continue
                    yield dict(seed=seed, arm=arm, step=24000, weight=weight, spec=spec, kind='final')
            for step in (8000, 16000):
                for weight in ('raw', 'ema'):
                    sides = set(c.CFG['arms'][arm]['valid_side_cycle'])
                    for spec in specs:
                        eligible = (spec['family'] in ('K1', 'K2') and spec['split'] == 'holdout') or (
                            spec['placement'] == 'center' and spec['side'] in sides and
                            (spec['family'] == 'K4' or (spec['family'] == 'G8' and spec['split'] == 'validation')))
                        if eligible:
                            yield dict(seed=seed, arm=arm, step=step, weight=weight, spec=spec, kind='learning')
            if arm in 'AC':
                for spec in specs:
                    if spec['placement'] == 'center' and (spec['family'] == 'K4' or
                         (spec['family'] == 'G8' and spec['split'] == 'test')):
                        yield dict(seed=seed, arm=arm, step=24000, weight='raw', spec=spec, kind='clock_fixed_diagnostic')


def job_key(job):
    return f"s{job['seed']}_{job['arm']}/{job['step']}_{job['weight']}_{job['kind']}_{job['spec']['key']}"


def input_bank(job):
    arm = 'B' if job['kind'] == 'clock_fixed_diagnostic' else job['arm']
    bank = d.bank_for_spec(job['spec'], arm)
    return bank


def save_bank(root, key, bank):
    path = root / 'banks' / (key+'.npz')
    native_hash = d.native_digest([bank])
    if path.exists():
        assert d.native_digest([c.load(path)]) == native_hash
    else:
        c.save(path, **bank)
    return path.relative_to(root).as_posix(), native_hash


def emit_prediction(root, job, bank, logits, source_sha):
    key = job_key(job)
    policy = 'canonical' if job['arm'] in 'BD' or job['kind'] == 'clock_fixed_diagnostic' else 'native'
    bank_path, input_hash = save_bank(root, job['spec']['key']+'_'+policy, bank)
    metrics = c.metrics(logits, bank['target'])
    relative = 'evaluation/'+key+'.npz'
    c.save(root / relative, logits=logits, **metrics, target=bank['target'], row_id=bank['row_id'],
           identity=np.array(key), common_input_hash=np.array(c.digest(bank['row_id'], bank['tokens'], bank['coordinates'], bank['valid'], bank['query'], bank['target'])),
           native_input_hash=np.array(input_hash), source_sha256=np.array(source_sha), bank_path=np.array(bank_path))
    row = dict(**job, key=key, prediction=relative, bank=bank_path, native_hash=input_hash,
               source_sha256=source_sha, **c.summary(logits, bank['target']))
    c.write(root / ('evaluation/'+key+'.json'), row)
    return row


def negative_controls(model, seed, arm, root):
    result = []
    for family, split in (('K4', 'train'), ('G8', 'test')):
        for side, allocated in ((4, 144), (6, 144), (12, 400), (20, 576)):
            spec = dict(family=family, split=split, side=side, placement='center', key=f'{family}_{split}_N{side}_center')
            base = d.bank_for_spec(spec, arm)
            base_logits = m.predict(model, base)
            n = len(base['query'])
            extra = allocated-side*side
            padded = {k: v.copy() for k, v in base.items()}
            padded['tokens'] = np.concatenate([base['tokens'], np.full((n, 1, extra), 3, np.int64)], 2)
            padded['valid'] = np.concatenate([base['valid'], np.zeros((n, 1, extra), bool)], 2)
            padded['coordinates'] = np.concatenate([base['coordinates'], np.zeros((n, 1, extra, 2), np.float32)], 2)
            candidates = [('PAD', padded)]
            if side == 20:
                translated = {k: v.copy() for k, v in base.items()}
                translated['coordinates'] += np.array([17, -11], dtype=np.float32)
                candidates.append(('translation', translated))
                perm = c.rng(seed, 20, 'joint_permutation_'+family).permutation(side*side)
                inv = np.argsort(perm)
                reordered = {k: v.copy() for k, v in base.items()}
                for field in ('tokens', 'coordinates', 'valid'):
                    reordered[field] = base[field][:, :, perm].copy()
                reordered['query'] = inv[base['query']]
                candidates.append(('joint_permutation', reordered))
            for kind, altered in candidates:
                logits = m.predict(model, altered)
                p = c.metrics(logits, altered['target'])['probability']
                bp = c.metrics(base_logits, base['target'])['probability']
                delta = float(np.max(np.abs(p-bp)))
                key = f's{seed}_{arm}_{family}_N{side}_{kind}'
                bank_path, input_hash = save_bank(root, 'control_'+key, altered)
                c.save(root / 'controls' / (key+'.npz'), logits=logits, base_logits=base_logits,
                       target=base['target'], row_id=base['row_id'], bank_path=np.array(bank_path), native_hash=np.array(input_hash))
                result.append(dict(seed=seed, arm=arm, family=family, side=side, kind=kind,
                                   max_probability_change=delta, passed=delta <= 2e-5))
    return result


def run(device='cuda'):
    lock = c.read(c.OUT / 'finals_locked.json')
    assert len(lock['finals']) == 24
    for rel, expected in lock['finals'].items():
        assert c.sha(c.OUT / rel) == expected
    plan = list(jobs())
    c.write(c.OUT / 'evaluation_manifest.json', dict(jobs=plan, jobs_count=len(plan), primary='G8_test_N20_center', unlocked=time.time()))
    rows, controls = [], []
    for seed in c.CFG['seed_labels']:
        for arm in 'ABCD':
            folder = c.OUT / 'training' / f's{seed}_{arm}'
            model = m.new_model(seed, device)
            model_jobs = [j for j in plan if j['seed'] == seed and j['arm'] == arm]
            loaded = None
            for j in model_jobs:
                c.deadline()
                path = folder / ('final.pt' if j['step'] == 24000 else f"step_{j['step']}.pt")
                identity = (j['step'], j['weight'])
                if identity != loaded:
                    payload = torch.load(path, map_location=device, weights_only=False)
                    assert payload['step'] == j['step'] and payload['seed'] == seed and payload['arm'] == arm
                    model.load_state_dict(payload[j['weight']])
                    source_sha = c.sha(path)
                    loaded = identity
                bank = input_bank(j)
                logits = m.predict(model, bank)
                rows.append(emit_prediction(c.OUT, j, bank, logits, source_sha))
                if len(rows) % 25 == 0:
                    c.status('evaluation', predictions=len(rows), expected=len(plan), current=job_key(j))
            payload = torch.load(folder / 'final.pt', map_location=device, weights_only=False)
            model.load_state_dict(payload['raw'])
            controls.extend(negative_controls(model, seed, arm, c.OUT))
            c.write(c.OUT / 'evaluation' / f's{seed}_{arm}' / 'complete.json',
                    dict(time=time.time(), rows=len(model_jobs), final_sha=c.sha(folder/'final.pt')))
            del model, payload
    c.write(c.OUT / 'evaluation_complete.json', dict(time=time.time(), rows=rows, controls=controls, count=len(rows)))
    assert len(rows) == len(plan)
    return rows, controls
