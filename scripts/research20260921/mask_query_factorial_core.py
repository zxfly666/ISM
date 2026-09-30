"""Same dense network and common CE normalization; only two training factors."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from ism_diffusion import geometry_study as gs
from sparse_conditioning_core import batch
from mask_query_factorial_design import auxiliary_views, ARMS


def views(parent, seed, step):
    stream_seed=seed+10000
    main=batch(parent,stream_seed,step,0);aux=batch(parent,stream_seed,step,1)
    for data in (main,aux):
        data['t'],data['mask'],data['noisy']=gs.corrupt_batch(data['clean'],stream_seed,data['index'])
    choices,retries=auxiliary_views(aux,stream_seed,aux['index'])
    hashes=[]
    for a in ARMS:
        d=choices[a]
        hashes.extend([d['noisy'],d['coords'],d['t'],d['queries'],d['labels']])
    return dict(main=main,aux=aux,choices=choices,mask_retries=retries,
        main_hash=gs.array_hash(main['clean'],main['coords']['A'],main['noisy'],main['t']),
        aux_clean_hash=gs.array_hash(aux['clean'],aux['coords']['A']),all_views_hash=gs.array_hash(*hashes))


def update(model,ema,opt,packed,step,steps,arm):
    main,aux=packed['main'],packed['aux'];sp=packed['choices'][arm]
    b=len(main['clean']);w=main['width'];micro={24:8,48:4,96:1}[w]
    opt.zero_grad(set_to_none=True);loss=0.
    for role,weight in (('main',.75),('aux',.25)):
        for start in range(0,b,micro):
            sl=slice(start,min(start+micro,b))
            if role=='aux':
                x=torch.as_tensor(sp['noisy'][sl,None],device='cuda')
                c=torch.as_tensor(sp['coords'][sl,None],device='cuda')
                t=torch.as_tensor(sp['t'][sl],device='cuda')
            else:
                x=torch.as_tensor(main['noisy'][sl],device='cuda')
                c=torch.as_tensor(main['coords']['A'][sl],device='cuda')
                t=torch.as_tensor(main['t'][sl],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x,t,c)
            if role=='aux':
                q=torch.as_tensor(sp['queries'][sl],device='cuda')
                lp=logits[:,:,0].transpose(1,2).gather(1,q[:,:,None].expand(-1,-1,2)).float().log_softmax(-1)
                y=torch.as_tensor(sp['labels'][sl],device='cuda',dtype=torch.long)
                objective=-lp.gather(-1,y[:,:,None]).sum()/(b*64)
            else:
                y=torch.as_tensor(main['clean'][sl],device='cuda')
                mask=torch.as_tensor(main['mask'][sl],device='cuda')
                ce=F.cross_entropy(logits.float(),y,reduction='none')
                objective=(ce*mask/t[:,None,None]).sum()/main['clean'].size
            value=weight*objective
            if not bool(torch.isfinite(value)):raise RuntimeError('Nonfinite loss')
            value.backward();loss+=float(value.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite gradient')
    lr=3e-5+7e-5*step/400 if step<=400 else 3e-5+3.5e-5*(1+math.cos(math.pi*(step-400)/max(steps-400,1)))
    for g in opt.param_groups:g['lr']=lr
    opt.step();gs.ema_update(ema,model)
    return dict(loss=loss,grad_norm=float(norm),lr=lr,tokens=18432,aux_slots=b*64,
        aux_unique_queries=int(sp['unique_queries'].sum()),aux_duplicate_slots=int(b*64-sp['unique_queries'].sum()),
        k_min=int(sp['k'].min()),k_max=int(sp['k'].max()),k_mean=float(sp['k'].mean()),
        t_mean=float(sp['t'].mean()),mask_retries=packed['mask_retries'],
        main_hash=packed['main_hash'],aux_clean_hash=packed['aux_clean_hash'],all_views_hash=packed['all_views_hash'],
        actual_input_hash=gs.array_hash(sp['noisy'],sp['coords'],sp['t']),
        actual_query_hash=gs.array_hash(sp['queries'],sp['labels']))
