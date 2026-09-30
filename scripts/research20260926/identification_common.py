"""Shared immutable identities and CPU utilities for the authorized I study."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONFIG_PATH=ROOT/'docs/research_reboot_20260921/GEOMETRY_IDENTIFICATION_CONFIG_20260926.json'
PLAN_PATH=ROOT/'docs/research_reboot_20260921/GEOMETRY_IDENTIFICATION_PLAN_20260926_ZH.md'
CONFIG=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
OUT=ROOT/'artifacts/geometry_identification_20260926'
DATA=ROOT/'data/level1/parents_l1024.npz'
ARMS=tuple(CONFIG['arms']);SEEDS=tuple(CONFIG['training_seeds'])
WIDTHS=tuple(CONFIG['widths']);STEPS=CONFIG['steps'];TOKENS=CONFIG['tokens_per_update']
DEADLINE=float('inf')

def check():
    if time.time()>=DEADLINE:raise TimeoutError('Immutable twelve-hour study deadline')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def ah(*arrays):
    h=hashlib.sha256()
    for v in arrays:
        a=np.asarray(v);h.update(str((a.shape,a.dtype.str)).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def rng(seed,index,role):
    tag=np.frombuffer(hashlib.sha256(role.encode()).digest()[:16],dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence([2026092621,int(seed),int(index),*map(int,tag)]))

def stream(seed,index,role):return int(rng(seed,index,role).integers(0,2**63-1))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def log(stage,**kw):
    row=dict(stage=stage,time=time.time(),**kw);write(OUT/'status.json',row)
    print(json.dumps(row),flush=True)

def increment_digest(old,new):return hashlib.sha256((old+new).encode()).hexdigest()

def source_files():
    p=[v for folder in ['ism_diffusion','scripts/research20260921','scripts/research20260926'] for v in (ROOT/folder).glob('*.py')]
    return sorted(p+[PLAN_PATH,CONFIG_PATH,DATA,ROOT/'docs/research_reboot_20260921/GEOMETRY_IDENTIFICATION_AUTHORIZATION_20260926_ZH.md'])

def verify_sources(protocol):
    for p,h in protocol['files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Frozen input changed: '+p)

def manifests(folder):
    return {p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.name not in ['last.pt','status.json','queue.log','manifest.json'] and '.tmp' not in p.name
            and not any(part.startswith('preflight') for part in p.relative_to(folder).parts)}
