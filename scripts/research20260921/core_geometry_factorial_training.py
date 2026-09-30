"""Fresh-model update, prediction and recoverable state for geometry factorial.

No old-checkpoint fallback. The runner must enforce campaign deadlines and
scientific completion gates. GPU execution still requires independent preflight.
"""
from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ism_diffusion import geometry_study as gs
import core_geometry_factorial_design as design
from core_geometry_factorial_data import array_hash


def fresh_state(seed, device='cuda'):
    if seed not in design.SEEDS:
        raise ValueError('Unexpected initialization seed')
    model = gs.new_model(seed,device=device)
    initial_hash = gs.model_hash(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),
                                 weight_decay=.05,fused=str(device).startswith('cuda'))
    return model,ema,optimizer,dict(step=0,initial_hash=initial_hash,elapsed_seconds=0.,
                                    paired_data_digest='',actual_input_digest='')


def learning_rate(step):
    if not 1 <= step <= design.STEPS:
        raise ValueError(step)
    if step <= 750:
        return 3e-4*step/750
    return 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-750)/(design.STEPS-750)))


def update(model,ema,optimizer,data,step,microbatch=None,amp=True):
    device=next(model.parameters()).device
    b=len(data['clean']); microbatch=b if microbatch is None else microbatch
    if data['clean'].size != design.TOKENS or microbatch < 1:
        raise ValueError('Invalid token/microbatch count')
    model.train(); optimizer.zero_grad(set_to_none=True)
    total=0.
    for start in range(0,b,microbatch):
        sl=slice(start,min(start+microbatch,b))
        x=torch.as_tensor(data['noisy'][sl],device=device)
        c=torch.as_tensor(data['coords'][sl],device=device)
        t=torch.as_tensor(data['t'][sl],device=device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                            enabled=amp and device.type=='cuda'):
            logits=model(x,t,c)
        if data['sparse']:
            queries=torch.as_tensor(data['queries'][sl],device=device)
            labels=torch.as_tensor(data['labels'][sl],device=device)
            valid=torch.as_tensor(data['query_valid'][sl],device=device)
            lp=logits.float().flatten(2).transpose(1,2).log_softmax(-1)
            selected=lp.gather(1,queries[:,:,None].expand(-1,-1,2))
            nll=-selected.gather(2,labels[:,:,None]).squeeze(-1)
            objective=nll.masked_select(valid).sum()/design.AUX_SLOTS
        else:
            labels=torch.as_tensor(data['clean'][sl],device=device)
            mask=torch.as_tensor(data['mask'][sl],device=device)
            nll=F.cross_entropy(logits.float(),labels,reduction='none')
            objective=(nll*mask/t[:,None,None]).sum()/design.TOKENS
        if not bool(torch.isfinite(objective)):
            raise RuntimeError('Nonfinite objective before optimizer mutation')
        objective.backward(); total+=float(objective.detach())
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError('Nonfinite gradient before optimizer mutation')
    lr=learning_rate(step)
    for group in optimizer.param_groups: group['lr']=lr
    optimizer.step(); gs.ema_update(ema,model)
    return dict(loss=total,grad_norm=float(grad),lr=lr,tokens=design.TOKENS,
                sparse=bool(data['sparse']),k_mean=float(data['k'].mean()),
                t_mean=float(data['t'].mean()),
                supervised_slots=int(data['query_valid'].sum()) if data['sparse'] else int(data['mask'].sum()),
                paired_data_hash=data['paired_data_hash'],actual_input_hash=data['actual_input_hash'])


@torch.inference_mode()
def predictions(model,bank,arm,batch_size=2,check=lambda:None):
    """FP32 conditional inference; use the trained arm's native coordinates."""
    if arm not in design.ARMS:
        raise ValueError(arm)
    device=next(model.parameters()).device; model.eval()
    coords=bank['summary_coords'] if arm=='G11S' else bank['true_coords']
    parts=[]
    for start in range(0,len(bank['t']),batch_size):
        check(); sl=slice(start,min(start+batch_size,len(bank['t'])))
        logits=model(torch.as_tensor(bank['noisy'][sl],device=device),
                     torch.as_tensor(bank['t'][sl],device=device),
                     torch.as_tensor(coords[sl],device=device))
        p=logits.float().softmax(1)[:,1].flatten(1).gather(
            1,torch.as_tensor(bank['queries'][sl],device=device)).cpu().numpy()
        if not np.isfinite(p).all(): raise RuntimeError('Nonfinite conditional prediction')
        parts.append(p)
    probability=np.concatenate(parts).astype(np.float64)
    y=bank['labels']; clipped=np.clip(probability,1e-12,1-1e-12)
    return dict(probability=probability,ce=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)),
                brier=(probability-y)**2,
                common_data_hash=array_hash(bank['noisy'],bank['true_coords'],bank['t'],bank['queries'],y),
                actual_input_hash=array_hash(bank['noisy'],coords,bank['t'],bank['queries']),
                parent=bank['parent'],chain=bank['chain'])


def save_state(path,model,ema,optimizer,metadata,seed,arm,protocol_hash,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists(): raise FileExistsError(path)
    if seed not in design.SEEDS or arm not in design.ARMS:
        raise ValueError('Invalid cell')
    value=dict(study='core_geometry_factorial_20260925',seed=seed,arm=arm,
        protocol_hash=protocol_hash,config=dict(model=gs.MODEL,variant='A',seed=seed,
            coordinate_mode='span_only' if arm=='G11S' else 'physical'),
        model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        metadata=metadata,step=int(metadata['step']),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def resume_state(path,seed,arm,protocol_hash,device='cuda'):
    value=torch.load(Path(path),map_location='cpu',weights_only=False)
    if (value['study']!='core_geometry_factorial_20260925' or value['seed']!=seed or
        value['arm']!=arm or value['protocol_hash']!=protocol_hash or value['step']>design.STEPS):
        raise RuntimeError('Refuse foreign, old, or incompatible checkpoint')
    model,ema,optimizer,initial=fresh_state(seed,device)
    if initial['initial_hash']!=value['metadata']['initial_hash']:
        raise RuntimeError('Initial-model identity mismatch')
    model.load_state_dict(value['model']);ema.load_state_dict(value['ema'])
    optimizer.load_state_dict(value['optimizer'])
    torch.set_rng_state(value['torch_rng'])
    if str(device).startswith('cuda'): torch.cuda.set_rng_state_all(value['cuda_rng'])
    np.random.set_state(value['numpy_rng']);random.setstate(value['python_rng'])
    return model,ema,optimizer,value['metadata']
