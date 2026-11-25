"""RG-2 v1.2 phase A. Run inside the portable, hash-verified project."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4
import numpy as np
import torch

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts/research20261001_rgpilot"))
sys.path.insert(0,str(ROOT/"scripts/research20260921"))
from pilot_net import ScaleDenoiser
from diagnostic_math import (uniform_axis_fft,cavity_distribution,conditional_stages,
                             joint_from_conditionals,kl_decomposition,divergence)

BETA=np.log1p(np.sqrt(2))/2
RNG_ROOT=2026112501
STEPS=(64,256)

def file_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()

def atomic(path,writer):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.parent/(path.name+"."+uuid4().hex+".tmp")
    try:
        with tmp.open("wb") as out:
            writer(out)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def write_json(path,obj):
    data=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()
    atomic(path,lambda out:out.write(data))

def npz(path,**arrays):
    atomic(path,lambda out:np.savez_compressed(out,**arrays))

@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as fd:
        if os.name=="nt":
            import msvcrt
            if fd.tell()==0:fd.write(b"0");fd.flush()
            fd.seek(0);msvcrt.locking(fd.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def load_model(entry,device):
    path=ROOT/entry["file"]
    if file_sha(path)!=entry["sha256"]:raise ValueError(f"Bad EMA: {path}")
    model=ScaleDenoiser()
    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),strict=True)
    assert sum(p.numel() for p in model.parameters())==1976962
    return model.to(device).eval().requires_grad_(False)

def coords(w,level,device):
    u=torch.arange(w,device=device,dtype=torch.float32)
    if level==1:u=3*u+1
    y,x=torch.meshgrid(u,u,indexing="ij")
    return torch.stack((y,x),dim=-1)

def predict(model,inputs,xy,t,level,device):
    x=torch.as_tensor(inputs,device=device,dtype=torch.long)
    xy=torch.as_tensor(xy,device=device,dtype=torch.float32)
    t=torch.as_tensor(t,device=device,dtype=torch.float32)
    return model(x,t,xy,level).float().softmax(1)[:,1]

def uniform_fields(seed,w,img):
    shape=(w,w)
    # Two independent counter-keyed fields. Identical for all arms and step counts.
    def one(role):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([RNG_ROOT,seed,w,img,role])))
        return (rng.integers(0,2**32,shape,dtype=np.uint32).astype(np.float64)+.5)/2**32
    return one(0),one(1)

def schedule(k):
    if k not in STEPS:raise ValueError(k)
    ts=np.cos(.5*np.pi*np.arange(257)/256)**2
    ts[0]=1.;ts[-1]=0.
    return ts[::256//k]

def reveal_at(u,k):
    return np.searchsorted(-schedule(k),-u,side="right")-1

def preflight(entries,device,out):
    checks=[]
    for seed in (92601,92602,92603):
        for w in (32,96):
            for i in (0,1,2,3):
                u,v=uniform_fields(seed,w,i)
                u2,v2=uniform_fields(seed,w,i)
                j64,j256=reveal_at(u,64),reveal_at(u,256)
                assert np.array_equal(j64,j256//4) and j256.min()>=0 and j256.max()<256
                assert np.array_equal(u,u2) and np.array_equal(v,v2) and not np.array_equal(u,v)
    for e in entries:
        model=load_model(e,device)
        row={"model":e["key"],"source_sha256":e["source_sha256"],"portable_sha256":e["sha256"]}
        for name,expected in e["checks"].items():
            with np.load(ROOT/"inputs"/("replay_"+name+".npz"),allow_pickle=False) as d:
                p=predict(model,d["noisy"],d["coords"],d["t"],int(d["level"]),device)
                got=np.take_along_axis(p.cpu().numpy().reshape(1,-1),d["queries"],axis=1)[0]
                err=float(np.max(np.abs(got-expected)))
                row[name]=err
                if err>5e-5:raise RuntimeError(f"{e['key']} {name} archived forward mismatch {err:g}")
        checks.append(row)
        write_json(out/"preflight_progress.json",checks)
    write_json(out/"preflight.json",dict(status="passed",paired_rng_cases=24,checks=checks))
    print("PASS strict loading, archived probabilities and 64/256 reveal-grid coupling",len(checks),flush=True)

def generate(model,e,w,k,image_ids,device):
    lev=int(w==32)
    ts=schedule(k)
    draws=[uniform_fields(e["seed"],w,i) for i in image_ids]
    event=torch.as_tensor(np.stack([reveal_at(u,k) for u,v in draws]),device=device)
    v=torch.as_tensor(np.stack([v for u,v in draws]),device=device)
    x=torch.full(event.shape,2,dtype=torch.long,device=device)
    xy=coords(w,lev,device)[None].expand(len(x),-1,-1,-1)
    for j in range(k):
        t=torch.full((len(x),),max(float(ts[j]),.01),dtype=torch.float32,device=device)
        p=model(x,t,xy,lev).float().softmax(1)[:,1]
        if not bool(torch.isfinite(p).all()):raise RuntimeError("Nonfinite spin probabilities")
        x=torch.where(event==j,(v<p).long(),x)
    assert not bool((x==2).any())
    return (x.cpu().numpy()*2-1).astype(np.int8)

def perf(device,model,e,out):
    for w in (32,96):
        start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        spin=generate(model,e,w,64,[10_000_000],device)
        torch.cuda.synchronize(device)
        sec=time.perf_counter()-start
        row=dict(width=w,steps=64,images=1,seconds=sec,peak_alloc_MiB=torch.cuda.max_memory_allocated(device)/2**20,
                 device=torch.cuda.get_device_name(device),torch=torch.__version__,finite=bool(np.isin(spin,[-1,1]).all()))
        write_json(out/f"benchmark_W{w}.json",row);print(json.dumps(row),flush=True)

def generate_matrix(entries,device,out,n,batch):
    for e in entries:
        if e["stage"]=="midpoint":continue
        model=load_model(e,device)
        widths=(96,32) if e["arm"]=="fine_plus_rg" else (96,)
        for w in widths:
            for k in STEPS:
                folder=out/"samples"/e["key"]/f"W{w}_K{k}"
                pending=[]
                for i in range(n):
                    f=folder/f"{i:05d}.npz"
                    if not f.exists():pending.append(i);continue
                    with np.load(f,allow_pickle=False) as d:
                        assert str(d["ema_sha"])==e["sha256"] and int(d["sample_id"])==i
                        assert int(d["width"])==w and int(d["steps"])==k and d["spins"].shape==(w,w)
                        assert np.isin(d["spins"],[-1,1]).all()
                for j in range(0,len(pending),batch):
                    ids=pending[j:j+batch]
                    generated=generate(model,e,w,k,ids,device)
                    for i,x in zip(ids,generated):
                        npz(folder/f"{i:05d}.npz",spins=x,width=w,level=int(w==32),steps=k,sample_id=i,
                            rng_root=RNG_ROOT,training_seed=e["seed"],ema_sha=e["sha256"])
                print("A1",e["key"],w,k,n,flush=True)

def observables(x):
    w=x.shape[-1]
    g=np.concatenate([uniform_axis_fft(x[j:j+16])[0][:,:w//2+1] for j in range(0,len(x),16)])
    return g,x.mean((1,2)),-g[:,1]

def majority(x):
    b,w,_=x.shape
    return np.sign(x.reshape(b,w//3,3,w//3,3).sum((2,4))).astype(np.int8)

def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def summarize_a1(entries,out,n):
    with np.load(ROOT/"inputs/mc_W144_centered.npz",allow_pickle=False) as d:
        ids=np.concatenate([np.flatnonzero(d["chain"]==c)[:32] for c in range(1,16,2)])
        z=2*np.unpackbits(d["fine_bits"][ids,24:120,3:15],axis=-1,bitorder="little").astype(np.int8)-1
    ref={w:observables(x) for w,x in [(96,z),(32,majority(z))]}
    res=[];dirs={}
    for e in entries:
        if e["stage"]=="midpoint":continue
        for kw in ((96,32) if e["arm"]=="fine_plus_rg" else (96,)):
            for k in STEPS:
                p=out/"samples"/e["key"]/f"W{kw}_K{k}"
                fields=[]
                for i in range(n):
                    with np.load(p/f"{i:05d}.npz",allow_pickle=False) as d:fields.append(d["spins"])
                xs=np.stack(fields)
                views=[("fine",xs),("B",majority(xs))] if kw==96 else [("C",xs)]
                for path,x in views:
                    w=x.shape[-1];g,m,energy=observables(x);rg,rm,re=ref[w];lo=w//4+1
                    npz(out/"metrics"/e["key"]/f"{path}_K{k}.npz",g=g,m=m,e=energy,r=np.arange(g.shape[1]))
                    res.append(dict(model=e["key"],steps=k,path=path,n=n,width=w,
                            G_error=relative(g.mean(0)[lo:],rg.mean(0)[lo:]),
                            Gc_error=relative((g.mean(0)-m.mean()**2)[lo:],(rg.mean(0)-rm.mean()**2)[lo:]),
                            mean_m=float(m.mean()),mean_abs_m=float(abs(m).mean()),energy=float(energy.mean())))
                    dirs[e["key"],k,path]=g.mean(0)
    compare=[]
    for (key,k,path),gg in dirs.items():
        if path=="C":
            bg=dirs[key,k,"B"]; denom=ref[32][0].mean(0)[9:]
            compare.append(dict(model=key,steps=k,B_C_distance_MC_denominator=float(np.linalg.norm(bg[9:]-gg[9:])/np.linalg.norm(denom))))
    for w,(g,m,e) in ref.items():npz(out/f"MC_reference_W{w}.npz",parent_ids=ids,g=g,m=m,e=e)
    write_json(out/"A1.json",dict(arms=res,paths_BC=compare,reference="historical MC, odd chains, 32 fields/chain; central W96"))

def midpoint_exact(entries,device,out,batch):
    for e in entries:
        model=load_model(e,device)
        for w in (48,96):
            for m in (1,8,32):
                name=f"fine_W{w}_M{m}"
                path=out/"A0_exact"/e["key"]/(name+".npz")
                if path.exists():
                    with np.load(path,allow_pickle=False) as d:
                        assert str(d["source_sha"])==e["source_sha256"] and d["KL"].shape==(256,)
                    continue
                with np.load(ROOT/"inputs/banks"/(name+".npz"),allow_pickle=False) as f:
                    x,xy,t,q,y,target=[f[key] for key in ("noisy","coords","t","queries","labels","target")]
                    probs=[]
                    for i in range(0,len(x),batch):
                        sl=slice(i,i+batch);p=predict(model,x[sl],xy[sl],t[sl],0,device).cpu().numpy().reshape(len(x[sl]),-1)
                        probs.append(np.take_along_axis(p,q[sl],axis=1))
                    p=np.clip(np.concatenate(probs).astype(float),1e-12,1-1e-12)
                    target=np.clip(target.astype(float),1e-12,1-1e-12)
                    kl=(target*np.log(target/p)+(1-target)*np.log((1-target)/(1-p))).mean(1)
                    ce=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(1)
                    # The archived exact banks call the parent identity "parent";
                    # empirical banks use "parent_ids". Never infer identities.
                    parent=f["parent"] if "parent" in f else f["parent_ids"]
                    npz(path,KL=kl,CE=ce,p=p,chain=f["chain_ids"],parent=parent,
                        origin=f["origin"],source_sha=e["source_sha256"],bank_sha=file_sha(ROOT/"inputs/banks"/(name+".npz")))
                print("A0",e["key"],name,float(kl.mean()),flush=True)

def block_diagnostics(entries,device,out,only_plus,batch):
    sites=[(6+i,6+j) for i in range(3) for j in range(3)]
    rows,cols=map(np.asarray,zip(*sites))
    plans={"serial":[0],"adjacent":[0,1,2],"checker":[0,2,4],"nine":list(range(9))}
    with np.load(ROOT/"inputs/a2_exteriors.npz",allow_pickle=False) as d:
        exteriors=list(zip(d["labels"].tolist(),d["fields"]))
    for e in entries:
        model=load_model(e,device)
        for label,x in exteriors[:1] if only_plus else exteriors:
            output=out/"A2"/e["key"]/(label+".npz")
            if output.exists():
                with np.load(output,allow_pickle=False) as z:
                    assert str(z["ema_sha"])==e["sha256"] and z["truth"].shape==(512,)
                continue
            bits,truth,_=cavity_distribution(x,sites,BETA)
            stored={"truth":truth,"bits":bits,"exterior":x,"ema_sha":e["sha256"]}
            known={}
            for name,first in plans.items():
                plan=[first]+[[j] for j in range(9) if j not in first]
                stages=conditional_stages(bits,truth,plan)
                predictions=[]
                for st in stages:
                    prior=tuple(sorted(st["prior"]))
                    if prior not in known:
                        size=2**len(prior); probs=[]
                        for i in range(0,size,batch):
                            ix=np.arange(i,min(i+batch,size))
                            state=np.broadcast_to((x>0).astype(np.int64),(len(ix),16,16)).copy()
                            state[:,rows,cols]=2
                            if prior:
                                state[:,rows[list(prior)],cols[list(prior)]]=(ix[:,None]>>np.arange(len(prior)))&1
                            timeval=max((9-len(prior))/256,.01)
                            pred=predict(model,state,coords(16,0,device)[None].expand(len(ix),-1,-1,-1),
                                         np.full(len(ix),timeval),0,device).cpu().numpy()
                            probs.append(pred[:,rows,cols])
                        known[prior]=np.concatenate(probs).astype(float)
                    order=[st["prior"].index(i) for i in prior]
                    code=st["assignments"][:,order]@(2**np.arange(len(prior),dtype=np.int64))
                    predictions.append(known[prior][code][:,st["group"]])
                oracle=joint_from_conditionals(bits,stages,[a["probability"] for a in stages])
                q,parts=kl_decomposition(bits,truth,stages,predictions,oracle)
                good=bits.sum(1)>=5
                summary=dict(**parts,TV=divergence(truth,q)["tv"],p_majority=float(truth[good].sum()),
                             q_majority=float(q[good].sum()),clock=[max((9-len(s["prior"]))/256,.01) for s in stages])
                stored[name+"_Q"]=q;stored[name+"_oracle"]=oracle;stored[name+"_result"]=json.dumps(summary,allow_nan=False)
            # Full exterior observed, compare the three nonadjacent checkerboard sites.
            b3,p3,_=cavity_distribution(x,[sites[i] for i in plans["checker"]],BETA)
            g=conditional_stages(b3,p3,[[0,1,2]])
            oracle3=joint_from_conditionals(b3,g,[g[0]["probability"]])
            assert np.max(abs(oracle3-p3))<1e-12
            state=((x>0).astype(np.int64))[None].copy()
            cidx=plans["checker"];state[:,rows[cidx],cols[cidx]]=2
            pp=predict(model,state,coords(16,0,device)[None],np.array([3/256]),0,device).cpu().numpy()[0,rows[cidx],cols[cidx]]
            q3=joint_from_conditionals(b3,g,[pp.astype(float)[None]])
            stored.update(control_P=p3,control_Q=q3,control_oracle=oracle3,control_clock=3/256)
            npz(output,**stored);print("A2",e["key"],label,flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["preflight","benchmark","a0","a1","a2","all"])
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--n",type=int,default=32)
    p.add_argument("--batch",type=int,default=1);p.add_argument("--block-batch",type=int,default=8)
    p.add_argument("--eval-batch",type=int,default=8)
    p.add_argument("--model",action="append",help="Exact model key, only for targeted preflight or A2")
    p.add_argument("--plus-only",action="store_true");p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    if min(args.n,args.batch,args.block_batch,args.eval_batch,args.threads)<1:p.error("Positive counts required")
    if args.model and args.action not in ("preflight","a2"):p.error("Model filtering is limited to diagnostics")
    if args.plus_only and args.action!="a2":p.error("--plus-only belongs to A2")
    if args.device=="cpu" and args.action in ("all","benchmark","a1"):p.error("Use the selected CUDA server for the generation matrix")
    manifest=json.loads((ROOT/"inputs/manifest.json").read_text("utf8"))
    torch.set_num_threads(args.threads)
    if args.device=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device);out=args.out.absolute()
    entries=manifest["models"]
    if args.model:
        if set(args.model)-{e["key"] for e in entries}:p.error("Unknown model")
        entries=[e for e in entries if e["key"] in args.model]
    gpu=torch.cuda.get_device_name(device) if args.device=="cuda" else None
    identity=dict(protocol=manifest["protocol"],input_manifest=file_sha(ROOT/"inputs/manifest.json"),
                  runner=file_sha(Path(__file__)),torch=str(torch.__version__),numpy=np.__version__,
                  device=str(device),gpu=gpu,precision="FP32, TF32 disabled",rng=RNG_ROOT,
                  n=args.n,generation_batch=args.batch,block_batch=args.block_batch,eval_batch=args.eval_batch)
    with run_lock(out/"run.lock"):
        ident=out/"identity.json"
        if ident.exists() and json.loads(ident.read_text("utf8"))!=identity:
            raise ValueError("Output directory belongs to a different run configuration")
        write_json(ident,identity)
        for f,expected in manifest["inputs"].items():
            if file_sha(ROOT/f)!=expected:raise ValueError(f"Changed input {f}")
        print(json.dumps(dict(action=args.action,host=platform.node(),gpu=gpu)),flush=True)
        if args.action in ("preflight","all"):preflight(entries,device,out)
        if args.action in ("benchmark","all"):
            e=next(e for e in entries if e["seed"]==92601 and e["arm"]=="fine_plus_rg" and e["stage"]=="final")
            perf(device,load_model(e,device),e,out)
        if args.action in ("a0","all"):midpoint_exact(entries,device,out,args.eval_batch)
        if args.action in ("a2","all"):
            tests=[e for e in entries if e["stage"]!="midpoint"] if not args.model else entries
            block_diagnostics(tests,device,out,args.plus_only,args.block_batch)
        if args.action in ("a1","all"):
            generate_matrix(entries,device,out,args.n,args.batch);summarize_a1(entries,out,args.n)

if __name__=="__main__":
    main()
