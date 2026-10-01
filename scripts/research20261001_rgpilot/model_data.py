"""Physical majority-map inputs, explicitly labelled scales and paired streams."""
from __future__ import annotations
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import common as c
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
from ism_diffusion.model import timestep_embedding
from ism_diffusion.geometry_study import model_hash

MODEL=dict(d_model=128,n_heads=4,n_blocks=7,mlp_ratio=4.,dropout=0.,vocab_size=4,output_classes=2,rope_base=10000.)

class ScaleDenoiser(CoordinateDenseDenoiser):
    def __init__(self):
        super().__init__(CoordinateDenoiserConfig(**MODEL))
        self.scale_embedding=torch.nn.Embedding(2,128)
        torch.nn.init.zeros_(self.scale_embedding.weight)

    def forward(self,tokens,t,coords,level=0):
        b,h,w=tokens.shape
        valid=torch.ones((b,h*w),device=tokens.device,dtype=torch.bool)
        x=self.token_embedding(tokens).reshape(b,h*w,-1)
        levels=torch.full((b,),int(level),device=tokens.device,dtype=torch.long)
        time=self.time_mlp(timestep_embedding(t,self.config.d_model))+self.scale_embedding(levels)
        x=x+time[:,None,:]
        for block in self.blocks:
            x=block(x,time,coords.reshape(b,h*w,2),valid)
        return self.output(F.silu(self.output_norm(x))).reshape(b,h,w,2).permute(0,3,1,2)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def initialize(seed, device='cuda'):
    seed_all(seed)
    payload=torch.load(c.base(seed),map_location='cpu',weights_only=False)
    assert (payload['seed'],payload['arm'],payload['step'])==(seed,'I-F',12000)
    model=ScaleDenoiser()
    result=model.load_state_dict(payload['ema'],strict=False)
    assert result.missing_keys==['scale_embedding.weight'] and not result.unexpected_keys
    model=model.to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=c.CFG['optimizer']['lr_start'],betas=(.9,.95),weight_decay=.05,fused=device=='cuda')
    return model,ema,opt

def axes_coords(w,b,code,scale=1):
    axes=np.broadcast_to(np.arange(w,dtype=np.int64),(b,2,w)).copy()
    coords=c.data.coordinates(axes,code)
    if scale==3:
        # Block-center translation is common to all positions; D4 remains consistent.
        centered=axes*3+1
        coords=c.data.coordinates(centered,code)
    return axes,coords

def field(parent,ids,origin,width):
    ax=np.broadcast_to(np.arange(width,dtype=np.int64),(len(ids),2,width))
    return c.data.sample(parent,ids,ax,origin).astype(np.int8)

def train_batch(parent,seed,cycle,role,coarse=False):
    w=c.CFG['widths'][(cycle-1)%4]; b=c.CFG['tokens_per_update']//w**2
    def rand(name):return c.rng(seed,cycle,role+'_'+name)
    ids=rand('parent').integers(len(parent.spins),size=b)
    origin=rand('origin').integers(parent.lattice_size,size=(b,2))
    flip=rand('flip').random(b)<.5; code=rand('d4').integers(8,size=b)
    if role=='aux':
        if coarse:
            raw=field(parent,ids,origin,3*w)
            clean=(c.data.majority(2*raw-1)>0).astype(np.int8)
        else:
            clean=field(parent,ids,(origin+w)%parent.lattice_size,w)
    else:
        assert not coarse
        clean=field(parent,ids,origin,w)
    clean[flip]=1-clean[flip]
    _,coords=axes_coords(w,b,code,3 if coarse else 1)
    t=rand('time').uniform(.002,1.,b).astype(np.float32)
    mask=rand('mask').random((b,w,w))<t[:,None,None]
    noisy=np.where(mask,2,clean).astype(np.int8)
    selection_hash=c.digest(ids,origin,flip,code)
    mask_time_hash=c.digest(mask,t)
    native_hash=c.digest(clean,noisy,coords,t,mask,np.array(int(coarse)))
    return dict(clean=clean,noisy=noisy,coords=coords,t=t,mask=mask,level=int(coarse),width=w,
        parent_ids=ids,origin=origin,flip=flip,d4=code,selection_hash=selection_hash,
        mask_time_hash=mask_time_hash,native_hash=native_hash)

def empirical_bank(parent,count,w,t,coarse,tag):
    r=c.rng(0,0,tag,root=c.CFG['test_master_seed'])
    ids=np.arange(count);origin=r.integers(parent.lattice_size,size=(count,2))
    clean=field(parent,ids,origin,3*w if coarse else w)
    if coarse:clean=(c.data.majority(2*clean-1)>0).astype(np.int8)
    mask=r.random(clean.shape)<t
    # Queries are sampled from masked positions independently of labels.
    queries=np.stack([r.choice(np.flatnonzero(row),size=32,replace=False) for row in mask.reshape(count,-1)])
    noisy=np.where(mask,2,clean).astype(np.int8)
    _,coords=axes_coords(w,count,np.zeros(count,dtype=np.int64),3 if coarse else 1)
    labels=np.take_along_axis(clean.reshape(count,-1),queries,1)
    return dict(clean=clean,noisy=noisy,coords=coords,t=np.full(count,t,dtype=np.float32),queries=queries,
        labels=labels,level=np.array(int(coarse)),chain_ids=parent.chain_ids[:count],parent_ids=ids,
        origin=origin,target_kind=np.array('empirical_coarse' if coarse else 'empirical_fine'))

def make_banks(parent,count,folder,learning=False):
    paths=[]
    if not learning:
        for w in (48,96):
            for m in (1,8,32):
                tag=f'rgpilot_test_fine_W{w}_M{m}'
                bank=c.data.local_bank(parent,np.arange(count),w,m,tag)
                bank['level']=np.array(0);bank['chain_ids']=parent.chain_ids[:count]
                bank['target_kind']=bank['target_type']
                path=folder/f'fine_W{w}_M{m}.npz';c.save(path,**bank);paths.append(path)
    for coarse,w in [(False,48),(True,32)]:
        for t in (.1,.5,.9):
            name=f'{"coarse" if coarse else "retention"}_W{w}_t{int(100*t)}'
            bank=empirical_bank(parent,count,w,t,coarse,f'rgpilot_{"learning" if learning else "test"}_{name}')
            path=folder/(name+'.npz');c.save(path,**bank);paths.append(path)
    return paths

@torch.no_grad()
def predict(model,bank,level_override=None):
    model.eval();device=next(model.parameters()).device
    n,w,_=bank['noisy'].shape
    batch=2 if w==96 else 8
    probs=[]
    for start in range(0,n,batch):
        c.deadline()
        sl=slice(start,start+batch)
        x=torch.as_tensor(bank['noisy'][sl],device=device,dtype=torch.long)
        xy=torch.as_tensor(bank['coords'][sl],device=device,dtype=torch.float32)
        tt=torch.as_tensor(bank['t'][sl],device=device,dtype=torch.float32)
        q=torch.as_tensor(bank['queries'][sl],device=device,dtype=torch.long)
        logits=model(x,tt,xy,int(bank['level']) if level_override is None else level_override).float()
        p=logits.softmax(1)[:,1].flatten(1).gather(1,q)
        assert bool(torch.isfinite(p).all())
        probs.append(p.cpu().numpy())
    p=np.concatenate(probs).astype(np.float64)
    p=np.clip(p,1e-12,1-1e-12)
    labels=bank['labels'].astype(np.float64)
    ce=-(labels*np.log(p)+(1-labels)*np.log1p(-p)).mean(1)
    result=dict(probabilities=p,empirical_ce=ce)
    kind=str(bank['target_kind'])
    if kind.startswith('exact'):
        q=bank['target'].astype(np.float64)
        target_ce=-(q*np.log(p)+(1-q)*np.log1p(-p)).mean(1)
        ent=-(q*np.log(q)+(1-q)*np.log1p(-q)).mean(1)
        result.update(exact_ce=target_ce,exact_kl=target_ce-ent)
        assert np.all(result['exact_kl']>=-1e-9)
    return result
