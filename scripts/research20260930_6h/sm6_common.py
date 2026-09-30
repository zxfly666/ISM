"""Isolated immutable protocol and addressed RNG for the size/clock study."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/research_reboot_20260921'
PREFIX = 'DENSE_MULTISIZE_CANONICAL_6H'
CONFIG_PATH = DOCS / (PREFIX + '_CONFIG_20260930.json')
PLAN = DOCS / (PREFIX + '_PLAN_20260930_ZH.md')
AUTH = DOCS / (PREFIX + '_AUTHORIZATION_20260930_ZH.md')
CFG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
OUT = ROOT / 'artifacts' / CFG['study']
EXPORT = ROOT / 'artifacts/dense_multisize_canonical_6h_exports_20260930'
PARENT = ROOT / CFG['data']['K1_K2_parent_bank']
BETA = CFG['data']['beta']
INHERITED_CONFIG = ROOT / CFG['data']['inherit_exact_tasks_from']
EXACT_G8 = json.loads(INHERITED_CONFIG.read_text(encoding='utf-8'))['data']['G8']
assert hashlib.sha256(INHERITED_CONFIG.read_bytes()).hexdigest() == CFG['data']['inherited_config_sha256']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        if a.dtype.hasobject:
            raise TypeError('Object arrays forbidden')
        h.update(str((a.shape, a.dtype.str)).encode('ascii'))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    if path.exists() and not replace:
        raise FileExistsError(path)
    os.replace(temp, path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise TypeError('Object arrays forbidden')
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rng(seed, step, role):
    tag = np.frombuffer(hashlib.sha256(role.encode('utf-8')).digest()[:16], dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence([CFG['root_seed'], int(seed), int(step), *map(int, tag)]))


def init_seed(label):
    return int(rng(label, 0, 'initialization').integers(0, 2**31))


def deadline(required=True):
    path = OUT / 'budget.json'
    if not path.exists() and not required:
        return None
    b = read(path)
    if time.time() >= b['deadline']:
        raise TimeoutError('Original six-hour deadline exhausted; no reset or extension')
    return b['deadline']


def source_snapshot():
    paths = [CONFIG_PATH, PLAN, AUTH, PARENT, INHERITED_CONFIG, Path(__file__).parent / 'README.md']
    paths += sorted(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'artifacts/dense_multisize_canonical_6h_20260930_design/evaluation_manifest.json']
    paths += [ROOT / 'scripts/research20260927/intervention_model.py']
    paths += [ROOT / ('ism_diffusion/' + x) for x in ('__init__.py', 'model.py', 'scale_model.py', 'ising.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}


def check_sources(snapshot):
    bad = [p for p, h in snapshot.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    if bad:
        raise RuntimeError('Frozen source/data mismatch: ' + repr(bad))


def metrics(logits, target):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    assert logits.shape == (len(y), 2) and np.isfinite(logits).all()
    assert np.isfinite(y).all() and ((y > 0) & (y < 1)).all()
    shifted = logits - logits.max(1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    p = np.exp(lp[:, 1])
    entropy = -(y * np.log(y) + (1-y) * np.log1p(-y))
    ce = -(y * lp[:, 1] + (1-y) * lp[:, 0])
    kl = ce - entropy
    assert kl.min() >= -2e-12
    return dict(probability=p, ce=ce, truth_entropy=entropy, kl=np.maximum(kl, 0),
                error=p-y, brier=y*(1-p)**2+(1-y)*p**2, excess_brier=(p-y)**2)


def summary(logits, target):
    m = metrics(logits, target)
    result = dict(n=len(target), mean_kl=float(m['kl'].mean()), max_kl=float(m['kl'].max()),
                  mean_ce=float(m['ce'].mean()), mean_entropy=float(m['truth_entropy'].mean()),
                  max_probability_error=float(np.abs(m['error']).max()),
                  probability_mae=float(np.abs(m['error']).mean()),
                  expected_brier=float(m['brier'].mean()), excess_brier=float(m['excess_brier'].mean()))
    result['ability_passed'] = result['max_kl'] <= 0.01 and result['max_probability_error'] <= 0.05
    return result


def status(stage, **values):
    write(OUT / 'status.json', dict(time=time.time(), stage=stage, **values), replace=True)


def resources():
    import subprocess
    r = dict(time=time.time(), disk_free=shutil.disk_usage(ROOT).free)
    if os.name != 'nt':
        for name in ('cpu.max', 'memory.max', 'memory.current'):
            p = Path('/sys/fs/cgroup') / name
            r[name] = p.read_text().strip() if p.exists() else None
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                             '--format=csv,noheader'], capture_output=True, text=True)
    r['gpu'] = result.stdout.strip()
    return r
