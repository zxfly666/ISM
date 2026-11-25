"""New RG pilot paths and immutable evidence; old scientific outputs are read-only."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/research20261001'))
import endpoint_common as old
import endpoint_data as data
from endpoint_management import archive, verify, union

CONFIG_PATH = ROOT/'docs/research_reboot_20260921/RG_DATA_TRAINING_PILOT_CONFIG_20261001.json'
PLAN_PATH = ROOT/'docs/research_reboot_20260921/RG_DATA_TRAINING_PILOT_PLAN_20261001_ZH.md'
CFG = old.read(CONFIG_PATH)
OUT = ROOT/'artifacts'/CFG['study']
EXPORT = ROOT/'artifacts'/(CFG['study']+'_exports')
TRAIN_DATA = ROOT/'data/level1/parents_l1024.npz'
TEST_DATA = ROOT/'artifacts/endpoint_mask_sampler_evaluation_20261001/reference/fresh_l1024.npz'
SEEDS, ARMS = CFG['seeds'], CFG['arms']
REQUEST_STOP = False
sha, read, write, save, digest = old.sha, old.read, old.write, old.save, old.digest

def rng(seed, index, role, root=None):
    words = np.frombuffer(hashlib.sha256(role.encode()).digest()[:16], dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence([root or CFG['master_seed'], seed, index, *words]))

def base(seed):
    return ROOT/f'artifacts/geometry_identification_20260926/training/s{seed}_I-F/final.pt'

def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)+'\n')
        f.flush()

def log(event, **kw):
    row=dict(time=time.time(), event=event, **kw)
    append(OUT/'run.jsonl', row)
    write(OUT/'status.json', row, exclusive=False)
    print(json.dumps(row, ensure_ascii=False), flush=True)

def deadline(reserve=0):
    if REQUEST_STOP or time.time()+reserve >= CFG['compute_deadline']:
        raise TimeoutError('New RG pilot compute deadline; no automatic continuation')

def source_files():
    return sorted(set([CONFIG_PATH, PLAN_PATH, TRAIN_DATA, TEST_DATA, *[base(s) for s in SEEDS],
        *list((ROOT/'scripts/research20261001_rgpilot').glob('*.py')),
        *list((ROOT/'ism_diffusion').glob('*.py')),
        *[ROOT/'scripts/research20261001'/x for x in ['endpoint_common.py','endpoint_data.py','endpoint_management.py']],
        ROOT/'docs/research_reboot_20260921/ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_CONFIG_20261001.json']))

def records(paths):
    return [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(set(paths))]

def check_sources(proto):
    for row in proto['sources']:
        p=ROOT/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], row['path']

def chain_select(path, split, count):
    with np.load(path, allow_pickle=False) as z:
        md=json.loads(str(z['metadata']))
        chains=z[split+'_chain_id'].astype(np.int16)
        unique=np.unique(chains)
        assert count % len(unique)==0
        ids=np.concatenate([np.flatnonzero(chains==k)[:count//len(unique)] for k in unique])
        assert len(ids)==count
        packed=z[split+'_packed'][ids]
        bits=np.unpackbits(packed, axis=-1, count=int(md['lattice_size']), bitorder='little')
        parent=data.Parent(2*bits.astype(np.int8)-1, chains[ids], int(md['lattice_size']), md)
    return parent, ids

def environment():
    import platform
    import subprocess
    import torch
    import scipy
    return dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
        scipy=scipy.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        deterministic=torch.are_deterministic_algorithms_enabled(), tf32=torch.backends.cuda.matmul.allow_tf32,
        torch_threads=torch.get_num_threads(), env={k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        gpu=torch.cuda.get_device_name(0), packages=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout.splitlines())
