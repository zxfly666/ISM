"""RG-2 fresh F/P/D/M development, with paired streams and gated execution."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
import phase_a as a
from pilot_net import ScaleDenoiser

ROOT=a.ROOT
TRAIN=Path('/root/shared-nvme/ISM_research_20260921/data/level1/parents_l1024.npz')
TRAIN_SHA='1f7de1ec81e82ebcfcbc4134d5670ee35711230f05d00cd037c7b9e575ef8934'
CONFIG=dict(protocol='RG-2 v1.2',stage='B_development',seeds=[112501,112502],arms=['F','P','D','M'],
            updates=12000,tokens=18432,widths=[16,24,32,48],warmup=750,lr_peak=.0003,lr_min=.00003,
            ema=.999,clip=1.,weight_decay=.05,betas=[.9,.95],rng_root=2026112503,
            validation_seed=2026112504,gradient_interval=1000,checkpoint_interval=500,
            generation_n=32,generation_steps=[64,256],precision='BF16 autocast, FP32 parameters and loss',
            width_schedule='cycle through 16/24/32/48; common then auxiliary within every cycle',
            task_embedding='0 common, 1 auxiliary for all four arms',
            learning_rate='linear 0 to peak over 750 optimizer updates, then cosine to lr_min at 12000',
            training_data_sha256=TRAIN_SHA,train_chains=[0,1,2,3,4,5],validation_chains=[6,7])
STOP=False

def digest(*arrays):
    h=hashlib.sha256()
    for v in arrays:
        x=np.asarray(v);h.update(str((x.shape,x.dtype.str)).encode());h.update(x.tobytes())
    return h.hexdigest()

def rand(seed,cycle,role):
    words=np.frombuffer(hashlib.sha256(role.encode()).digest()[:16],dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence([CONFIG['rng_root'],seed,cycle,*map(int,words)]))

def parent(split):
    if a.file_sha(TRAIN)!=TRAIN_SHA:raise ValueError('Training MC changed')
    with np.load(TRAIN,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));chain=z[split+'_chain_id']
        x=np.unpackbits(z[split+'_packed'],axis=-1,bitorder='little').astype(np.int8)
    assert meta['lattice_size']==1024 and np.isclose(meta['beta'],a.BETA)
    assert np.array_equal(np.unique(chain),CONFIG['train_chains' if split=='train' else 'validation_chains'])
    return x,chain,meta

def batch(fields,seed,cycle,role,arm):
    if role not in ('common','aux'):raise ValueError(role)
    w=CONFIG['widths'][(cycle-1)%4];b=CONFIG['tokens']//w**2
    def rng(s):return rand(seed,cycle,role+'_'+s)
    ids=rng('parent').integers(len(fields),size=b);origin=rng('origin').integers(1024,size=(b,2))
    flip=rng('flip').random(b)<.5;d4=rng('d4').integers(8,size=b)
    span=w if role=='common' else 3*w;axis=np.arange(span)
    raw=fields[ids[:,None,None],(origin[:,0,None,None]+axis[None,:,None])%1024,
               (origin[:,1,None,None]+axis[None,None,:])%1024]
    if role=='common':clean=raw.copy()
    elif arm in ('F','P'):clean=raw[:,w:2*w,w:2*w].copy()
    elif arm=='D':clean=raw[:,1::3,1::3].copy()
    elif arm=='M':clean=(raw.reshape(b,w,3,w,3).sum((2,4))>=5).astype(np.int8)
    else:raise ValueError(arm)
    clean[flip]=1-clean[flip]
    scale=role=='aux' and arm!='F'
    c=np.broadcast_to(a.coords(w,int(scale),'cpu').numpy(),(b,w,w,2)).copy()
    for i,code in enumerate(d4):
        if code&1:c[i]=c[i][...,::-1].copy()
        if code&2:c[i,...,0]*=-1
        if code&4:c[i,...,1]*=-1
    t=rng('time').uniform(.002,1,b).astype(np.float32)
    endpoint=rng('endpoint').random(b)<.02;t[endpoint]=1.
    mask=rng('mask').random(clean.shape)<t[:,None,None];noisy=np.where(mask,2,clean).astype(np.int8)
    assert clean.shape==(b,w,w) and clean.size==18432 and mask[endpoint].all()
    return dict(clean=clean,noisy=noisy,coords=c,t=t,mask=mask,level=int(role=='aux'),width=w,
                parent=ids,origin=origin,flip=flip,d4=d4,endpoint=endpoint,
                selection_hash=digest(ids,origin,flip,d4),noise_hash=digest(t,mask,endpoint),
                spin_hash=digest(clean),coordinate_hash=digest(c),input_hash=digest(noisy,c,t,np.array(int(role=='aux'))))

def initialize(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    m=ScaleDenoiser().cuda();assert sum(p.numel() for p in m.parameters())==1976962
    ema=copy.deepcopy(m).eval().requires_grad_(False)
    opt=torch.optim.AdamW(m.parameters(),lr=CONFIG['lr_peak'],betas=tuple(CONFIG['betas']),weight_decay=.05,fused=False)
    return dict(model=m,ema=ema,opt=opt,step=0,seed=seed)

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x

def payload(s):
    return dict(model=cpu(s['model'].state_dict()),ema=cpu(s['ema'].state_dict()),optimizer=cpu(s['opt'].state_dict()),
                step=s['step'],seed=s['seed'],arm=s.get('arm'),config=CONFIG,source_sha256=a.file_sha(Path(__file__)),torch_rng=torch.get_rng_state(),
                cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()],python_rng=random.getstate(),numpy_rng=np.random.get_state())

def restore(p):
    assert p['config']==CONFIG and p['source_sha256']==a.file_sha(Path(__file__))
    s=initialize(p['seed']);s['model'].load_state_dict(p['model'],strict=True);s['ema'].load_state_dict(p['ema'],strict=True)
    s['opt'].load_state_dict(p['optimizer']);s['step']=p['step'];s['arm']=p['arm']
    torch.set_rng_state(p['torch_rng']);torch.cuda.set_rng_state_all(p['cuda_rng'])
    random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng'])
    return s

def loss(model,b):
    device=next(model.parameters()).device
    x=torch.as_tensor(b['noisy'],device=device,dtype=torch.long);y=torch.as_tensor(b['clean'],device=device,dtype=torch.long)
    t=torch.as_tensor(b['t'],device=device);xy=torch.as_tensor(b['coords'],device=device);mask=torch.as_tensor(b['mask'],device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,t,xy,b['level'])
    ce=F.cross_entropy(p.float(),y,reduction='none')
    return (ce*mask/t[:,None,None]).sum()/18432

def lr(step):
    if step<=750:return .0003*step/750
    return .00003+.5*(.0003-.00003)*(1+math.cos(math.pi*(step-750)/(12000-750)))

def update(s,b):
    s['model'].train();s['opt'].zero_grad(set_to_none=True)
    objective=loss(s['model'],b)
    if not bool(torch.isfinite(objective)):raise FloatingPointError('Nonfinite loss')
    objective.backward();norm=torch.nn.utils.clip_grad_norm_(s['model'].parameters(),1.,error_if_nonfinite=True)
    rate=lr(s['step']+1)
    for g in s['opt'].param_groups:g['lr']=rate
    s['opt'].step()
    with torch.no_grad():
        for em,raw in zip(s['ema'].parameters(),s['model'].parameters()):em.mul_(.999).add_(raw,alpha=.001)
    s['step']+=1
    return dict(step=s['step'],loss=float(objective.detach()),unclipped_grad_norm=float(norm),clipped=bool(norm>1),lr=rate,
                width=b['width'],level=b['level'],endpoint_samples=int(b['endpoint'].sum()),
                **{key:b[key] for key in ('selection_hash','noise_hash','spin_hash','coordinate_hash','input_hash')})

def gradients(s,fields,arm):
    m=s['model'];before=digest(*[p.detach().cpu().numpy() for p in m.parameters()])
    vectors=[];norms=[]
    for role in ('common','aux'):
        m.zero_grad(set_to_none=True)
        b=batch(fields,CONFIG['validation_seed'],4,role,arm)
        loss(m,b).backward()
        v=torch.cat([p.grad.flatten() for name,p in m.named_parameters() if name!='scale_embedding.weight' and p.grad is not None])
        vectors.append(v.detach().clone());norms.append(float(torch.linalg.vector_norm(v)))
    cosine=float(F.cosine_similarity(vectors[0],vectors[1],dim=0)) if min(norms)>0 else None
    m.zero_grad(set_to_none=True)
    assert before==digest(*[p.detach().cpu().numpy() for p in m.parameters()])
    return dict(step=s['step'],cosine=cosine,common_norm=norms[0],auxiliary_norm=norms[1],
                aux_to_common_norm=norms[1]/norms[0] if norms[0]>0 else None,excluded='scale_embedding.weight',
                precision='BF16 forward, FP32 gradients; inverse-t loss over 18432 tokens',diagnostic_cycle=4)

@torch.no_grad()
def validate(model,fields):
    model.eval();rows=[]
    for cycle in range(1,5):
        b=batch(fields,CONFIG['validation_seed'],cycle,'common','F')
        rows.append(dict(width=b['width'],loss=float(loss(model,b)),input_hash=b['input_hash']))
    return rows

def compare(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left.cpu(),right.cpu()),'Resume tensor mismatch'
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:compare(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        for x,y in zip(left,right):compare(x,y)
    else:assert left==right

def preflight(out):
    fields,chain,meta=parent('train');started=time.monotonic();checks=0
    for cycle in range(1,9):
        for role in ('common','aux'):
            arms={arm:batch(fields,112501,cycle,role,arm) for arm in CONFIG['arms']}
            for key in ('selection_hash','noise_hash'):
                assert len({b[key] for b in arms.values()})==1
            assert arms['F']['spin_hash']==arms['P']['spin_hash']
            assert len({arms[x]['coordinate_hash'] for x in ('P','D','M')})==1
            if role=='common':assert len({b['input_hash'] for b in arms.values()})==1
            checks+=1
    # Exact resume of raw, EMA and Adam moments after serializing at a common/
    # auxiliary boundary; all four widths occur before the endpoint.
    resume=[]
    for arm in CONFIG['arms']:
        s=initialize(112501);s['arm']=arm
        for step in range(1,9):
            cycle=(step+1)//2;role='common' if step%2 else 'aux'
            update(s,batch(fields,112501,cycle,role,arm))
            if step==4:a.atomic(out/f'resume_{arm}.pt',lambda f:torch.save(payload(s),f))
        expected=payload(s);del s
        s=restore(torch.load(out/f'resume_{arm}.pt',map_location='cpu',weights_only=False))
        for step in range(5,9):update(s,batch(fields,112501,(step+1)//2,'common' if step%2 else 'aux',arm))
        got=payload(s)
        for key in ('model','ema','optimizer'):compare(expected[key],got[key])
        resume.append(dict(arm=arm,status='bitwise_equal',updates=8));del s,expected,got
    # Fresh initialization identical across arms, no weights loaded from A.
    s=initialize(112501);h1=digest(*[p.cpu().numpy() for p in s['model'].state_dict().values()]);del s
    s=initialize(112501);h2=digest(*[p.cpu().numpy() for p in s['model'].state_dict().values()]);assert h1==h2
    timings=[]
    for step in range(1,25):
        b=batch(fields,112501,(step+1)//2,'common' if step%2 else 'aux','M')
        torch.cuda.synchronize();begin=time.monotonic();update(s,b);torch.cuda.synchronize()
        if step>8:timings.append(dict(width=b['width'],seconds=time.monotonic()-begin))
    per_width={str(w):float(np.mean([r['seconds'] for r in timings if r['width']==w])) for w in CONFIG['widths']}
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    a.write_json(out/'preflight.json',dict(status='passed',paired_stream_cases=checks,resume=resume,
                seconds=elapsed,initial_state_hash=h1,training_data_sha256=TRAIN_SHA,training_chain_ids=np.unique(chain).tolist(),
                update_seconds_by_width=per_width,
                eight_model_training_forecast_hours=1.25*8*12000*np.mean(list(per_width.values()))/3600,
                forecast_note='25% reserve for validation, gradients, inputs, checkpoint I/O; generation separate',
                torch=str(torch.__version__),gpu=torch.cuda.get_device_name()))
    print('B PREFLIGHT PASSED',elapsed,flush=True)

def append(path,row):
    with path.open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()

def train(out,gate):
    accept=json.loads(gate.read_text());assert accept['A_to_B']=='accepted'
    assert json.loads((out/'preflight.json').read_text())['status']=='passed'
    fields,_,_=parent('train');validation,_,_=parent('val')
    for seed in CONFIG['seeds']:
        for arm in CONFIG['arms']:
            folder=out/f's{seed}_{arm}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():continue
            s=restore(torch.load(folder/'last.pt',map_location='cpu',weights_only=False)) if (folder/'last.pt').exists() else initialize(seed)
            assert s['seed']==seed and (s['step']==0 or s['arm']==arm)
            # Replay interrupted updates, discarding only this run's log tail.
            for name in ('updates.jsonl','validation.jsonl','gradients.jsonl'):
                path=folder/name
                if path.exists():
                    rows=[json.loads(line) for line in path.read_text().splitlines()]
                    a.atomic(path,lambda f:f.write((''.join(json.dumps(r)+'\n' for r in rows if r['step']<=s['step'])).encode()))
            if s['step']==0:
                initial=payload(s)
                initial_hash=digest(*[v.numpy() for v in initial['model'].values()])
                ema_hash=digest(*[v.numpy() for v in initial['ema'].values()]);assert initial_hash==ema_hash
                initial_path=out/'initial'/f's{seed}.pt'
                if initial_path.exists():
                    saved=torch.load(initial_path,map_location='cpu',weights_only=False)
                    compare(saved['model'],initial['model']);compare(saved['ema'],initial['ema'])
                else:a.atomic(initial_path,lambda f:torch.save(initial,f))
                a.write_json(folder/'initialization.json',dict(seed=seed,arm=arm,step=0,model_hash=initial_hash,
                             ema_hash=ema_hash,optimizer_state_entries=len(initial['optimizer']['state']),
                             shared_initial_checkpoint=str(initial_path.relative_to(out)),training_data_sha256=TRAIN_SHA))
                append(folder/'validation.jsonl',dict(step=0,values=validate(s['ema'],validation)))
            s['arm']=arm
            started=time.monotonic()
            while s['step']<CONFIG['updates']:
                step=s['step']+1;cycle=(step+1)//2;role='common' if step%2 else 'aux'
                row=update(s,batch(fields,seed,cycle,role,arm));append(folder/'updates.jsonl',row)
                if step%1000==0:
                    append(folder/'validation.jsonl',dict(step=step,values=validate(s['ema'],validation)))
                    append(folder/'gradients.jsonl',gradients(s,validation,arm))
                if step%500==0 or STOP:
                    p=payload(s);a.atomic(folder/'last.pt',lambda f:torch.save(p,f))
                    if step in (4000,8000,12000):a.atomic(folder/f'step{step:05d}.pt',lambda f:torch.save(p,f))
                    a.write_json(out/'status.json',dict(stage='B',seed=seed,arm=arm,step=step,seconds_this_session=time.monotonic()-started,pid=os.getpid()))
                    print('B',seed,arm,step,flush=True)
                if STOP:raise InterruptedError('Stopped after saving a resumable checkpoint')
            final=payload(s);a.atomic(folder/'final_ema.pt',lambda f:torch.save(final['ema'],f))
            a.write_json(folder/'complete.json',dict(seed=seed,arm=arm,updates=s['step'],common_updates=6000,auxiliary_updates=6000,
                         tokens=12000*18432,ema_sha=a.file_sha(folder/'final_ema.pt'),seconds_this_session=time.monotonic()-started))
            del s

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gate',type=Path);args=p.parse_args();out=args.out.absolute();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    def stop(sig,frame):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    identity=dict(config=CONFIG,source_sha256=a.file_sha(Path(__file__)),model_source=a.file_sha(ROOT/'scripts/rg2_v12/pilot_net.py'),
                  torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name())
    with a.run_lock(out/'run.lock'):
        target=out/'identity.json'
        if target.exists():assert json.loads(target.read_text())==identity
        a.write_json(target,identity)
        if args.action=='preflight':preflight(out)
        else:
            if args.gate is None:p.error('--gate required to train')
            train(out,args.gate)

if __name__=='__main__':main()
