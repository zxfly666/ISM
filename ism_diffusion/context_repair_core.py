"""Full/subset paired denoising objectives for the authorized 2026-09-22 study.

Retains all observations/queries, original physical coordinates and original t.
No production model changes and no reference/test inputs are used for training.
"""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from . import geometry_study as gs
from .scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig


def construct(config, seed, device="cuda"):
    torch.manual_seed(seed)
    return CoordinateDenseDenoiser(CoordinateDenoiserConfig(**config)).to(device)


def pack_views(batch, t, masked, noisy, seed, step, qmax=64, rho=None, policy="uniform"):
    clean=batch["clean"].reshape(len(t),-1)
    coords=batch["coords"]["A"].reshape(len(t),-1,2)
    xx=noisy.reshape(len(t),-1); mm=masked.reshape(len(t),-1)
    B,N=clean.shape
    qr=gs.rng(seed,step,"repair_queries_v1")
    ur=gs.rng(seed,step,"repair_subsets_v1")
    fractions=ur.choice([.125,.25,.5,.75,1.],B) if rho is None else np.full(B,rho)
    queries=[];selected=[]
    for b in range(B):
        m=np.flatnonzero(mm[b]);v=np.flatnonzero(~mm[b])
        q=np.sort(qr.choice(m,min(qmax,len(m)),replace=False))
        other=np.setdiff1d(m,q,assume_unique=True)
        count=int(np.floor(fractions[b]*len(other)))
        if policy=="uniform": u=ur.choice(other,count,replace=False)
        elif policy=="far":
            if len(q):
                distance=((coords[b,other,None,:]-coords[b,q][None,:,:])**2).sum(-1).min(1)
                order=np.lexsort((other,-distance))
                u=other[order[:count]]
            else: u=other[:count]
        else: raise ValueError(policy)
        s=np.sort(np.concatenate([v,q,u])).astype(np.int64)
        if not len(s): s=np.array([0],dtype=np.int64)
        assert np.isin(v,s).all() and np.isin(q,s).all()
        queries.append(q);selected.append(s)
    Q=max(1,max(map(len,queries))); L=max(map(len,selected))
    fullq=np.zeros((B,Q),np.int64);subq=fullq.copy()
    target=np.zeros((B,Q),np.int64); qvalid=np.zeros((B,Q),bool)
    sub=np.full((B,L),3,np.int64);sc=np.zeros((B,L,2),np.float32);valid=np.zeros((B,L),bool)
    for b,(q,s) in enumerate(zip(queries,selected)):
        n=len(q);fullq[b,:n]=q;subq[b,:n]=np.searchsorted(s,q)
        target[b,:n]=clean[b,q];qvalid[b,:n]=True
        sub[b,:len(s)]=xx[b,s];sc[b,:len(s)]=coords[b,s];valid[b,:len(s)]=True
        assert np.array_equal(coords[b,q],sc[b,subq[b,:n]])
    nmask=mm.sum(1)
    weights=np.divide(nmask,t*np.maximum(qvalid.sum(1),1))/(B*N)
    weights[nmask==0]=0
    return dict(full=xx,full_coords=coords,sub=sub,sub_coords=sc,sub_valid=valid,
                fullq=fullq,subq=subq,labels=target,qvalid=qvalid,weights=weights.astype(np.float32),
                t=np.asarray(t,np.float32),N=N,B=B,nmask=nmask,retained=valid.sum(1),rho=fractions,
                original_visible=(~mm).sum(1),
                paired_hash=gs.array_hash(clean,coords,t,masked,fullq,sub,sc,valid))


def tensor_batch(views, device="cuda"):
    return {k:torch.as_tensor(v,device=device) for k,v in views.items() if isinstance(v,np.ndarray)}


def query_logits(model, data, view):
    if view=="full":
        logits=model(data["full"][:,None,:],data["t"],data["full_coords"][:,None,:,:])
    else:
        logits=model(data["sub"][:,None,:],data["t"],data["sub_coords"][:,None,:,:],
                     data["sub_valid"][:,None,:])
    q=data["fullq" if view=="full" else "subq"]
    return logits[:,:,0,:].transpose(1,2).gather(1,q[:,:,None].expand(-1,-1,2)).float()


def objective(model, data, arm, lam, amp=True):
    device=next(model.parameters()).device.type
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp and device=="cuda"):
        lp=query_logits(model,data,"full")
        ls=query_logits(model,data,"sub") if arm!="F0" else lp
    weight=data["weights"][:,None]*data["qvalid"]
    full=(F.cross_entropy(lp.transpose(1,2),data["labels"],reduction="none")*weight).sum()
    sub=(F.cross_entropy(ls.transpose(1,2),data["labels"],reduction="none")*weight).sum()
    p_log=lp.log_softmax(-1);q_log=ls.log_softmax(-1)
    mix_log=torch.logaddexp(p_log,q_log)-math.log(2.)
    js_row=.5*((p_log.exp()*(p_log-mix_log)).sum(-1)+(q_log.exp()*(q_log-mix_log)).sum(-1))
    js=(js_row*weight).sum()
    sup=full if arm=="F0" else .5*(full+sub)
    loss=sup+lam*js if arm=="F2" else sup
    return loss,dict(full_ce=full,sub_ce=sub,js=js,supervised=sup)


def update(model,ema,optimizer,batch,seed,step,total_steps,arm,lam):
    global_step=24000+step
    t,mask,noisy=gs.corrupt_batch(batch["clean"],seed,global_step)
    views=pack_views(batch,t,mask,noisy,seed,global_step)
    data=tensor_batch(views)
    optimizer.zero_grad(set_to_none=True)
    loss,parts=objective(model,data,arm,lam)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite gradients/loss before optimizer mutation")
    lr=3e-5+(1e-4-3e-5)*step/400 if step<=400 else (
        3e-5+.5*(1e-4-3e-5)*(1+math.cos(math.pi*(step-400)/max(total_steps-400,1))))
    for group in optimizer.param_groups: group["lr"]=lr
    optimizer.step();gs.ema_update(ema,model)
    result={k:float(v.detach()) for k,v in parts.items()}
    result.update(loss=float(loss.detach()),grad_norm=float(norm),lr=lr,
        original_tokens=int(views["N"]*views["B"]),retained_tokens=int(views["retained"].sum()),
        queries=int(views["qvalid"].sum()),mean_t=float(t.mean()),
        min_visible=int(views["original_visible"].min()),
        n_windows_k1to8=int(((views["original_visible"]>=1)&(views["original_visible"]<=8)).sum()),
        paired_hash=views["paired_hash"])
    if not all(math.isfinite(v) for v in result.values() if isinstance(v,(float,int))):
        raise RuntimeError("nonfinite repair training state")
    return result


def self_test(device="cpu"):
    torch.set_num_threads(1)
    config=dict(d_model=32,n_heads=1,n_blocks=2,mlp_ratio=4.,dropout=0.,vocab_size=4,output_classes=2,rope_base=10000.)
    model=construct(config,722,device)
    # Nonzero output weights are needed to detect an erroneously zero JS test.
    torch.nn.init.normal_(model.output.weight,std=.1)
    random=np.random.default_rng(882)
    batch=dict(clean=random.integers(0,2,(2,16,16)),coords={"A":gs.coordinate_arrays(2,16,"gap",gs.GAPS,88,1)[0]["A"]})
    t,mask,noisy=gs.corrupt_batch(batch["clean"],88,1,.8)
    views=pack_views(batch,t,mask,noisy,88,1,qmax=8,rho=.25)
    data=tensor_batch(views,device)
    loss,parts=objective(model,data,"F2",1.,False)
    assert torch.isfinite(loss) and parts["js"]>=-1e-7
    loss.backward();assert model.output.weight.grad.abs().sum()>0
    identity=tensor_batch(pack_views(batch,t,mask,noisy,88,1,qmax=8,rho=1),device)
    a=query_logits(model,identity,"full");b=query_logits(model,identity,"sub")
    assert float((a-b).abs().max())<1e-5
    _,identity_parts=objective(model,identity,"F2",1.,False)
    assert abs(float(identity_parts["js"]))<1e-6
    # All visible: no query loss, while autograd remains well defined.
    empty=pack_views(batch,np.ones(2,np.float32)*.01,np.zeros_like(mask),batch["clean"],99,1)
    z,_=objective(model,tensor_batch(empty,device),"F2",1.,False)
    assert abs(float(z))<1e-7
    # Compare the exhaustive-query estimator with the historical full loss.
    allq=tensor_batch(pack_views(batch,t,mask,noisy,88,1,qmax=256,rho=1),device)
    matched,_=objective(model,allq,"F0",0.,False)
    logits=model(torch.tensor(noisy,device=device),torch.tensor(t,device=device),
                 torch.tensor(batch["coords"]["A"],device=device))
    ce=F.cross_entropy(logits.float(),torch.tensor(batch["clean"],device=device),reduction="none")
    original=(ce*torch.tensor(mask,device=device)/torch.tensor(t,device=device)[:,None,None]).sum()/batch["clean"].size
    assert abs(float(matched-original))<1e-5
    return dict(status="passed",identity_logit_max_error=float((a-b).abs().max()),
                loss_identity_error=float(abs(matched-original)),gradient_test=True,empty_query_test=True,
                physical_coordinates_and_visible_preserved=True,device=device)
