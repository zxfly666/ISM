"""Native-coordinate I training/inference and complete recoverable states."""
import copy,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_evaluation import open_energy_density
import identification_common as c
import identification_data as d

def fresh(seed,device='cuda'):
    if seed not in c.SEEDS:raise ValueError(seed)
    m=gs.new_model(seed,device=device);e=copy.deepcopy(m).eval().requires_grad_(False)
    o=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.05,fused=str(device).startswith('cuda'))
    return m,e,o,dict(step=0,initial_hash=gs.model_hash(m),elapsed_seconds=0.,paired_data_digest='',actual_input_digest='')

def lr(step):
    if not 1<=step<=12000:raise ValueError(step)
    return 3e-4*step/750 if step<=750 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-750)/11250))

def update(m,e,o,b,step,amp=True):
    device=next(m.parameters()).device;m.train();o.zero_grad(set_to_none=True)
    x=torch.as_tensor(b['noisy'],device=device,dtype=torch.long);coords=torch.as_tensor(b['coords'],device=device)
    t=torch.as_tensor(b['t'],device=device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
        logits=m(x,t,coords)
    if b['sparse']:
        q=torch.as_tensor(b['queries'],device=device);y=torch.as_tensor(b['labels'],device=device);valid=torch.as_tensor(b['query_valid'],device=device)
        lp=logits.float().flatten(2).transpose(1,2).log_softmax(-1).gather(1,q[:,:,None].expand(-1,-1,2))
        nll=-lp.gather(2,y[:,:,None]).squeeze(-1);loss=nll[valid].sum()/512
    else:
        y=torch.as_tensor(b['clean'],device=device,dtype=torch.long);mask=torch.as_tensor(b['mask'],device=device)
        nll=F.cross_entropy(logits.float(),y,reduction='none');loss=(nll*mask/t[:,None,None]).sum()/c.TOKENS
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite objective')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    if not bool(torch.isfinite(grad)):raise RuntimeError('Nonfinite gradient')
    rate=lr(step)
    for group in o.param_groups:group['lr']=rate
    o.step();gs.ema_update(e,m)
    return dict(loss=float(loss.detach()),grad_norm=float(grad),lr=rate,tokens=c.TOKENS,
                width=b['width'],kind=b['kind'],sparse=bool(b['sparse']),
                supervised_slots=int(b['query_valid'].sum()) if b['sparse'] else int(b['mask'].sum()),
                paired_data_hash=b['paired_data_hash'],actual_input_hash=b['actual_input_hash'])

def save_state(path,m,e,o,meta,seed,arm,protocol_hash,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():raise FileExistsError(path)
    payload=dict(study=c.CONFIG['study'],seed=seed,arm=arm,protocol_hash=protocol_hash,
        config=dict(model=gs.MODEL,coordinate_mode=arm),model=m.state_dict(),ema=e.state_dict(),optimizer=o.state_dict(),
        metadata=meta,step=int(meta['step']),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if next(m.parameters()).is_cuda else [],
        numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def restore(path,seed,arm,protocol_hash,device='cuda'):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if (p['study'],p['seed'],p['arm'],p['protocol_hash'])!=(c.CONFIG['study'],seed,arm,protocol_hash):raise RuntimeError('Foreign checkpoint')
    m,e,o,meta=fresh(seed,device)
    assert meta['initial_hash']==p['metadata']['initial_hash'] and p['step']<=c.STEPS
    m.load_state_dict(p['model']);e.load_state_dict(p['ema']);o.load_state_dict(p['optimizer'])
    torch.set_rng_state(p['torch_rng']);np.random.set_state(p['numpy_rng']);random.setstate(p['python_rng'])
    if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(p['cuda_rng'])
    return m,e,o,p['metadata']

@torch.inference_mode()
def predict(m,b,arm,rep=0,batch_size=2,check=c.check):
    device=next(m.parameters()).device;m.eval();coords=d.native_coordinates(b,arm,rep);parts=[]
    for start in range(0,len(b['t']),batch_size):
        check();sl=slice(start,start+batch_size)
        logits=m(torch.as_tensor(b['noisy'][sl],device=device,dtype=torch.long),torch.as_tensor(b['t'][sl],device=device),torch.as_tensor(coords[sl],device=device))
        q=torch.as_tensor(b['queries'][sl],device=device,dtype=torch.long)
        p=logits.float().softmax(1)[:,1].flatten(1).gather(1,q).cpu().numpy()
        parts.append(p)
    p=np.concatenate(parts).astype(np.float64);y=b['labels'];pc=np.clip(p,1e-12,1-1e-12)
    if not np.isfinite(p).all():raise RuntimeError('Nonfinite predictions')
    return dict(probability=p,ce=-(y*np.log(pc)+(1-y)*np.log1p(-pc)),brier=(p-y)**2,
                common_data_hash=np.array(d.common_hash(b)),actual_input_hash=np.array(c.ah(b['noisy'],coords,b['t'],b['queries'])),
                parent=b['parent'],chain=b['chain'],coordinate_rep=np.array(rep),native_arm=np.array(arm))

def image_seed(seed,image):return c.stream(c.CONFIG['role_seeds']['generation'],int(image),f'generation_image_model_seed_{seed}')

@torch.inference_mode()
def sample_images(m,seeds,width=96,steps=256,amp=True,check=c.check):
    """Same categorical/reveal S0 law, independently keyed uniform arrays per image.

    For two classes categorical sampling is an inverse-CDF Bernoulli draw.
    No change to cosine-square times, t_min=.01, temperature1 or reveal kernel.
    Different numerical RNG plumbing is declared, not a bitwise old-run replay.
    """
    device=next(m.parameters()).device;m.eval();b=len(seeds)
    u=torch.stack([torch.rand((steps+1,2,width,width),device=device,generator=torch.Generator(device=device).manual_seed(int(s))) for s in seeds],1)
    coords=torch.as_tensor(d.coordinates(np.broadcast_to(np.arange(width),(b,2,width)).copy(),'I-F'),device=device)
    valid=torch.ones((b,width,width),device=device,dtype=torch.bool)
    tokens=torch.full((b,width,width),2,device=device,dtype=torch.long)
    times=[math.cos(.5*math.pi*j/steps)**2 for j in range(steps+1)]
    for j in range(steps):
        check();masked=tokens==2
        if not bool(masked.any()):break
        t,s=times[j:j+2];mt=torch.full((b,),max(t,.01),device=device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):logits=m(tokens,mt,coords,valid)
        prob=logits.float().softmax(1)[:,1]
        sampled=(u[j,:,0]<prob).long();reveal=(u[j,:,1]<min(max(1-s/max(t,1e-12),0.),1.))&masked
        tokens=torch.where(reveal,sampled,tokens)
    remaining=tokens==2
    if bool(remaining.any()):
        mt=remaining.float().mean((1,2)).clamp_min(.01)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):logits=m(tokens,mt,coords,valid)
        sampled=(u[steps,:,0]<logits.float().softmax(1)[:,1]).long();tokens=torch.where(remaining,sampled,tokens)
    if not bool(((tokens==0)|(tokens==1)).all()):raise RuntimeError('Unresolved generated token')
    return (2*tokens.cpu().numpy()-1).astype(np.int8)

def physical_stats(spins):
    # Same open-axis definition as old science; no dependence between images.
    from evaluate_study import physical_axis_statistics
    w=spins.shape[-1];axis=np.broadcast_to(np.arange(w),(len(spins),w))
    result=physical_axis_statistics(spins,[axis,axis]);result['energy']=open_energy_density(spins)
    return result

def generate(m,out,seed,samples=128,check=c.check):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);parts=[];seconds=0.
    for start in range(0,samples,16):
        check();ids=np.arange(start,min(start+16,samples));seeds=np.array([image_seed(seed,i) for i in ids],dtype=np.int64)
        if next(m.parameters()).is_cuda:torch.cuda.synchronize()
        began=time.perf_counter();spins=sample_images(m,seeds,check=check)
        if next(m.parameters()).is_cuda:torch.cuda.synchronize()
        elapsed=time.perf_counter()-began;seconds+=elapsed;stat=physical_stats(spins);parts.append(stat)
        c.save(out/f'shard_{start:05d}.npz',spins=spins,image_ids=ids,rng_seeds=seeds,seconds=np.array(elapsed),**stat)
    merged={k:np.concatenate([z[k] for z in parts]) for k in parts[0]};c.save(out/'statistics.npz',**merged)
    c.write(out/'complete.json',dict(status='complete',images=samples,shards=len(parts),sampling_seconds=seconds,
                                   independent_image_rng=True,sampler='S0_cos_squared_256_temperature1_no_MC_correction'))
    return merged
