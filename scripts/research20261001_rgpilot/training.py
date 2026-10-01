"""Fixed-cycle paired training and recoverable raw/EMA/AdamW/RNG checkpoints."""
from __future__ import annotations
import copy
import hashlib
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import common as c
import model_data as m
from ism_diffusion.geometry_study import ema_update, model_hash

def learning_rate(cycle):
    cfg=c.CFG['optimizer'];warm=cfg['warmup_cycles']
    if cycle<=warm:return cfg['lr_start']+(cfg['lr_peak']-cfg['lr_start'])*cycle/warm
    return cfg['lr_end']+.5*(cfg['lr_peak']-cfg['lr_end'])*(1+math.cos(math.pi*(cycle-warm)/(c.CFG['cycles']-warm)))

def update(model,ema,opt,batch,cycle):
    c.deadline();model.train();opt.zero_grad(set_to_none=True)
    device=next(model.parameters()).device
    x=torch.as_tensor(batch['noisy'],device=device,dtype=torch.long)
    xy=torch.as_tensor(batch['coords'],device=device)
    t=torch.as_tensor(batch['t'],device=device)
    y=torch.as_tensor(batch['clean'],device=device,dtype=torch.long)
    mask=torch.as_tensor(batch['mask'],device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x,t,xy,batch['level'])
    ce=F.cross_entropy(logits.float(),y,reduction='none')
    loss=(ce*mask/t[:,None,None]).sum()/c.CFG['tokens_per_update']
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite_training_loss')
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(grad)):raise FloatingPointError('nonfinite_training_gradient')
    lr=learning_rate(cycle)
    for group in opt.param_groups:group['lr']=lr
    opt.step();ema_update(ema,model)
    return dict(loss=float(loss.detach()),grad=float(grad),lr=lr,masked=int(mask.sum()),
        level=batch['level'],width=batch['width'],tokens=c.CFG['tokens_per_update'],
        selection_hash=batch['selection_hash'],mask_time_hash=batch['mask_time_hash'],native_hash=batch['native_hash'])

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

def save_state(path,state,protocol_hash,exclusive=True,ema_only=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():raise FileExistsError(path)
    payload=dict(study=c.CFG['study'],seed=state['seed'],arm=state['arm'],cycle=state['cycle'],
        updates=state['updates'],protocol_hash=protocol_hash,metadata=copy.deepcopy(state['metadata']),
        ema=cpu(state['ema'].state_dict()))
    if not ema_only:
        payload.update(model=cpu(state['model'].state_dict()),optimizer=cpu(state['opt'].state_dict()),
            torch_rng=torch.get_rng_state(),cuda_rng=[s.cpu() for s in torch.cuda.get_rng_state_all()],
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            stateless_input_state=dict(next_input=copy.deepcopy(state['next_input']),master_seed=c.CFG['master_seed']))
    tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp);tmp.replace(path)
    return c.sha(path)

def make_state(seed,arm):
    model,ema,opt=m.initialize(seed)
    return dict(seed=seed,arm=arm,model=model,ema=ema,opt=opt,cycle=0,updates=0,next_input=dict(cycle=1,role='core'),
        metadata=dict(initial_hash=model_hash(model),initial_ema_hash=model_hash(ema),
            base_sha256=c.sha(c.base(seed)),native_digest='',common_digest='',aux_selection_digest='',
            seconds=0.,optimizer_reset=True))

def fold(previous,value):return hashlib.sha256((previous+value).encode()).hexdigest()

def train(states,parent,proto):
    n=c.CFG['cycles'];block=c.CFG['block_cycles'];ph=proto['protocol_hash']
    for start in range(1,n+1,block):
        for state in states:
            c.deadline();begun=time.time();seed,arm=state['seed'],state['arm']
            folder=c.OUT/f'training/s{seed}_{arm}'
            for cycle in range(start,min(start+block,n+1)):
                for role in (['core'] if arm=='fine_only' else ['core','aux']):
                    batch=m.train_batch(parent,seed,cycle,role,arm=='fine_plus_rg' and role=='aux')
                    result=update(state['model'],state['ema'],state['opt'],batch,cycle)
                    state['updates']+=1
                    state['next_input']=dict(cycle=cycle,role='aux') if role=='core' and arm!='fine_only' else dict(cycle=cycle+1,role='core')
                    meta=state['metadata'];meta['native_digest']=fold(meta['native_digest'],batch['native_hash'])
                    if role=='core':meta['common_digest']=fold(meta['common_digest'],batch['native_hash'])
                    else:meta['aux_selection_digest']=fold(meta['aux_selection_digest'],batch['selection_hash']+batch['mask_time_hash'])
                    c.append(folder/'log.jsonl',dict(cycle=cycle,update=state['updates'],role=role,**result))
                state['cycle']=cycle
            torch.cuda.synchronize();state['metadata']['seconds']+=time.time()-begun
            if state['cycle']==n:
                final_sha=save_state(folder/'final.pt',state,ph)
                c.write(folder/'complete.json',dict(seed=seed,arm=arm,cycle=n,updates=state['updates'],
                    final_sha256=final_sha,metadata=state['metadata'],time=time.time()))
            else:
                save_state(folder/'last.pt',state,ph,exclusive=False)
                if state['cycle']==n//2:
                    save_state(folder/'midpoint_ema.pt',state,ph,ema_only=True)
            c.log('training_block_complete',seed=seed,arm=arm,cycles=state['cycle'],
                cumulative_updates=sum(s['updates'] for s in states),total_updates=c.CFG['total_updates'])

def audit_training(parent,proto):
    rows=[];common={};aux={};initial={};checked=0
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=c.OUT/f'training/s{seed}_{arm}'
            complete=c.read(folder/'complete.json')
            assert c.sha(folder/'final.pt')==complete['final_sha256']
            payload=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
            assert payload['protocol_hash']==proto['protocol_hash'] and payload['cycle']==c.CFG['cycles']
            expected_updates=c.CFG['cycles']*(1 if arm=='fine_only' else 2)
            assert payload['updates']==expected_updates
            for key in ['model','ema']:
                assert all(bool(torch.isfinite(t).all()) for t in payload[key].values())
            for state in payload['optimizer']['state'].values():
                assert int(state['step'])==expected_updates
                assert bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all())
            for key in ['torch_rng','cuda_rng','numpy_rng','python_rng','stateless_input_state']:assert key in payload
            restored=m.ScaleDenoiser();restored.load_state_dict(payload['model'],strict=True)
            opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.05,fused=False)
            opt.load_state_dict(payload['optimizer'])
            for group in opt.param_groups:group['fused']=False
            # Full input reconstruction, not merely checking recorded hashes against each other.
            logs=[__import__('json').loads(line) for line in (folder/'log.jsonl').read_text().splitlines()]
            assert len(logs)==expected_updates
            nd=cd=ad='';index=0
            for cycle in range(1,c.CFG['cycles']+1):
                for role in (['core'] if arm=='fine_only' else ['core','aux']):
                    row=logs[index];index+=1
                    assert (row['cycle'],row['update'],row['role'])==(cycle,index,role)
                    assert np.isfinite([row['loss'],row['grad']]).all()
                    key=(seed,cycle)
                    if role=='core' and key in common:
                        assert row['native_hash']==common[key]
                    else:
                        b=m.train_batch(parent,seed,cycle,role,arm=='fine_plus_rg' and role=='aux')
                        for field in ['native_hash','selection_hash','mask_time_hash']:assert row[field]==b[field]
                        if role=='core':common[key]=row['native_hash']
                    if role=='aux':
                        pairing=row['selection_hash']+row['mask_time_hash']
                        if key in aux:assert aux[key]==pairing
                        aux[key]=pairing;ad=fold(ad,pairing)
                    else:cd=fold(cd,row['native_hash'])
                    nd=fold(nd,row['native_hash']);checked+=1
            meta=payload['metadata']
            assert (nd,cd,ad)==(meta['native_digest'],meta['common_digest'],meta['aux_selection_digest'])
            if seed in initial:assert initial[seed]==meta['initial_hash']
            initial[seed]=meta['initial_hash']
            rows.append(dict(seed=seed,arm=arm,updates=expected_updates,final_sha256=complete['final_sha256']))
    assert checked==c.CFG['total_updates']
    result=dict(status='passed',checkpoints=9,midpoints=9,updates=checked,all_logs_finite=True,
        full_unique_inputs_rebuilt=True,common_inputs_paired=True,aux_random_choices_paired=True,
        all_initializations_paired=True,cpu_model_and_optimizer_restore=True,rows=rows,time=time.time())
    c.write(c.OUT/'audit/training.json',result)
    return result
