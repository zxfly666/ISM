"""Matched-token dense continuation. This module never reads evaluation data."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from ism_diffusion import geometry_study as gs
from sparse_conditioning_design import sparse_view,probability_target,schedule


def batch(parent,seed,step,stream):
    w,kind=schedule(seed,step); b=9216//(w*w)
    for retry in range(100):
        index=step*10000+stream*100+retry
        try:
            value=gs.make_batch(parent,b,w,kind,gs.GAPS,seed,index,True)
            value.update(index=index,retries=retry,width=w,kind=kind)
            return value
        except ValueError as exc:
            if "physical axis span" not in str(exc): raise
    raise RuntimeError("geometry rejection limit")


def views(parent,oracle,seed,step):
    main=batch(parent,seed,step,0); aux=batch(parent,seed,step,1)
    main["t"],main["mask"],main["noisy"]=gs.corrupt_batch(main["clean"],seed,main["index"])
    aux["t"],aux["mask"],aux["noisy"]=gs.corrupt_batch(aux["clean"],seed,aux["index"])
    sparse=sparse_view(aux["clean"],aux["coords"]["A"],seed,aux["index"])
    soft=sparse["labels"].copy()
    flat=aux["clean"].reshape(len(soft),-1)
    for i,e in enumerate(sparse["evidence"]):
        if len(e)<=2:
            soft[i]=probability_target(oracle[int(aux["chain"][i])],sparse["coords"][i,sparse["queries"][i]],
                                      sparse["coords"][i,e],flat[i,e])
    sparse["soft"]=soft
    return main,aux,sparse


def update(model,ema,opt,packed,step,steps,arm,microbatch=None):
    main,aux,sp=packed; b=len(main["clean"]); w=main["width"]
    microbatch=microbatch or {24:8,48:4,96:1}[w]
    opt.zero_grad(set_to_none=True); loss_values=[]
    for role,data,weight in (("main",main,.75),("aux",aux,.25)):
        sparse=role=="aux" and arm!="T0"
        for start in range(0,b,microbatch):
            sl=slice(start,min(b,start+microbatch))
            if sparse:
                x=torch.as_tensor(sp["noisy"][sl,None],device="cuda")
                c=torch.as_tensor(sp["coords"][sl,None],device="cuda")
                t=torch.as_tensor(sp["t"][sl],device="cuda")
            else:
                x=torch.as_tensor(data["noisy"][sl],device="cuda")
                c=torch.as_tensor(data["coords"]["A"][sl],device="cuda")
                t=torch.as_tensor(data["t"][sl],device="cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=model(x,t,c)
            if sparse:
                q=torch.as_tensor(sp["queries"][sl],device="cuda")
                lp=logits[:,:,0].transpose(1,2).gather(1,q[:,:,None].expand(-1,-1,2)).float().log_softmax(-1)
                p=torch.as_tensor(sp["soft" if arm=="T2" else "labels"][sl],device="cuda",dtype=torch.float32)
                objective=(-(1-p)*lp[:,:,0]-p*lp[:,:,1]).sum()/(b*64)
            else:
                y=torch.as_tensor(data["clean"][sl],device="cuda")
                mask=torch.as_tensor(data["mask"][sl],device="cuda")
                ce=F.cross_entropy(logits.float(),y,reduction="none")
                objective=(ce*mask/t[:,None,None]).sum()/data["clean"].size
            value=weight*objective
            if not bool(torch.isfinite(value)):raise RuntimeError("nonfinite loss before optimizer mutation")
            value.backward();loss_values.append(float(value.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(norm)):raise RuntimeError("nonfinite gradient before optimizer mutation")
    lr=3e-5+7e-5*step/400 if step<=400 else 3e-5+3.5e-5*(1+math.cos(math.pi*(step-400)/max(steps-400,1)))
    for group in opt.param_groups:group["lr"]=lr
    opt.step();gs.ema_update(ema,model)
    return dict(loss=sum(loss_values),grad_norm=float(norm),lr=lr,tokens=18432,
        main_targets=int(main["mask"].sum()),aux_targets=int(aux["mask"].sum()) if arm=="T0" else b*64,
        main_hash=gs.array_hash(main["clean"],main["coords"]["A"],main["noisy"],main["t"]),
        aux_clean_hash=gs.array_hash(aux["clean"],aux["coords"]["A"]),
        sparse_input_hash=gs.array_hash(sp["noisy"],sp["coords"],sp["t"],sp["queries"]),
        k_counts={str(k):int((sp["k"]==k).sum()) for k in (1,2,4,8,16,32)},
        soft_target_windows=int((sp["k"]<=2).sum()) if arm=="T2" else 0,
        geometry_rejections=main["retries"]+aux["retries"])
