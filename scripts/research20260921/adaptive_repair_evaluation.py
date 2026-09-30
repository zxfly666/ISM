"""Preserved query-level confirmation evidence and seven geometry generations."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.context_repair_core import pack_views,tensor_batch,query_logits
from ism_diffusion.scale_evaluation import load_scale_model,open_energy_density
from ism_diffusion.scale_data import load_parent_split
import evaluate_study as ev

GEN_DEFS=[dict(name="continuous48",width=48,kind="continuous",gaps=[1],samples=256),
          dict(name="continuous64",width=64,kind="continuous",gaps=[1],samples=384),
          dict(name="continuous96",width=96,kind="continuous",gaps=[1],samples=384),
          dict(name="held_gap57_w32",width=32,kind="gap",gaps=[5,7],samples=512),
          dict(name="held_s10_w48",width=48,kind="gap",gaps=[10],samples=512),
          dict(name="new_s9_w48",width=48,kind="gap",gaps=[9],samples=256),
          dict(name="new_irregular_w48",width=48,kind="irregular",gaps=[9,10,11],samples=256)]
_original_batch=gs.make_batch


def prepare_reference_evidence(parent,out,definitions):
    """Freeze support using only MC/geometry, before reading model predictions."""
    out.mkdir(parents=True,exist_ok=True)
    manifest=out / "support.json"
    if manifest.exists(): return json.loads(manifest.read_text())
    supports={}
    for definition in definitions:
        tag=int.from_bytes(hashlib.sha256(definition["name"].encode()).digest()[:3],"little")
        shards=[];parents=[];chains=[]
        for start in range(0,definition["samples"],16):
            b=confirmation_batch(parent,min(16,definition["samples"]-start),definition["width"],
                                 definition["kind"],definition["gaps"],918073,tag+start)
            shards.append(ev.physical_axis_statistics((2*b["clean"]-1).astype(np.int8),b["axes"]))
            parents.extend(b["parent"].tolist());chains.extend(b["chain"].tolist())
        stats=ev.merge_statistics(shards);parents=np.array(parents);chains=np.array(chains)
        counts=stats["pair_count"];r=np.arange(counts.shape[1])
        nparent=np.array([len(np.unique(parents[counts[:,j]>0])) for j in r])
        nchain=np.array([len(np.unique(chains[counts[:,j]>0])) for j in r])
        support=(nparent>=64)&(nchain>=4)
        long=((r>=25)&(r<=definition["width"]//2)) if definition["kind"]=="continuous" else r>=129
        ev.atomic_npz(out / (definition["name"]+".npz"),**stats,parent=parents,chain=chains,
                      r=r,nparent=nparent,nchain=nchain,support=support)
        supports[definition["name"]]=dict(long_r=r[support&long].tolist(),
            definition=definition,rule="MC only: >=64 parents and >=4 chains; no model values used")
    gs.atomic_json(manifest,dict(supports=supports,created=time.time(),status="frozen"))
    return json.loads(manifest.read_text())


def confirmation_batch(parent,batch,width,kind,gaps,seed,index,augment=False):
    if kind!="irregular":
        result=_original_batch(parent,batch,width,kind,gaps,seed,index,False)
    else:
        assert width==48
        random=gs.rng(seed,index,"fixed_span_irregular")
        base=np.array([9]*23+[11]*23+[10])
        axes=[np.stack([np.r_[0,random.permutation(base).cumsum()] for _ in range(batch)]) for _ in range(2)]
        cc=np.empty((batch,width,width,2),np.float32)
        cc[...,0]=axes[0][:,:,None];cc[...,1]=axes[1][:,None,:]
        result=dict(axes=axes,coords={"A":cc},origin=gs.rng(seed,index,"origin").integers(parent.lattice_size,size=(batch,2)))
    # Balanced distinct parent order, not independent crop pseudoreplication.
    chain_values=np.unique(parent.chain_ids)
    k=len(chain_values)
    ordinal=np.arange(batch)+int(index)
    ids=[]
    for j in ordinal:
        ch=chain_values[j%k];pool=np.flatnonzero(parent.chain_ids==ch)
        ids.append(pool[(j//k)%len(pool)])
    ids=np.array(ids,dtype=np.int64)
    origin=result["origin"];axes=result["axes"];L=parent.lattice_size
    xx=(origin[:,0,None]+axes[0])%L;yy=(origin[:,1,None]+axes[1])%L
    clean=(parent.spins[ids[:,None,None],xx[:,:,None],yy[:,None,:]]>0).astype(np.int64)
    result.update(clean=clean,parent=ids,chain=parent.chain_ids[ids])
    return result


@torch.inference_mode()
def conditionals(model,parent,out,seed,definitions,samples=512,deadline=float("inf")):
    rows=[]
    for gi,definition in enumerate(definitions):
        for ti,tv in enumerate((.2,.5,.8,.95)):
            for rep in range(2):
                for start in range(0,samples,8):
                    if time.time()>=deadline: raise TimeoutError("conditional deadline")
                    file=out / f"g{gi}_t{ti}_m{rep}_b{start:04d}.npz"
                    if file.exists():
                        with np.load(file) as z:
                            records=json.loads(str(z["records"].item()))
                        rows.extend(records);continue
                    b=confirmation_batch(parent,min(8,samples-start),definition["width"],definition["kind"],
                                         definition["gaps"],2026092303,gi*100000+start)
                    t,mask,noisy=gs.corrupt_batch(b["clean"],2026092303,gi*100000+ti*10000+rep*1000+start,tv)
                    views=pack_views(b,t,mask,noisy,2026092303,gi*100000+ti*10000+rep*1000+start,rho=1/3)
                    data=tensor_batch(views)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        full=query_logits(model,data,"full").softmax(-1)[...,1].cpu().numpy()
                        sub=query_logits(model,data,"sub").softmax(-1)[...,1].cpu().numpy()
                    records=[];qv=views["qvalid"]
                    for j in range(len(t)):
                        for name,pred in (("full",full[j]),("sub",sub[j])):
                            p=np.clip(pred[qv[j]].astype(float),1e-7,1-1e-7);y=views["labels"][j,qv[j]]
                            if not len(y): continue
                            records.append(dict(geometry=definition["name"],t=tv,mask=rep,view=name,
                                chain=int(b["chain"][j]),parent=int(b["parent"][j]),sample=start+j,
                                ce=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
                                brier=float(((p-y)**2).mean()),
                                sensitivity=float(np.abs(full[j,qv[j]]-sub[j,qv[j]]).mean())))
                    ev.atomic_npz(file,full_p=full,sub_p=sub,labels=views["labels"],qvalid=qv,
                        original_query=views["fullq"],subset_query=views["subq"],
                        original_coordinates=b["coords"]["A"],subset_coordinates=views["sub_coords"],
                        subset_valid=views["sub_valid"],subset_tokens=views["sub"],
                        noisy=noisy,t=t,chain=b["chain"],parent=b["parent"],origin=b["origin"],
                        records=np.array(json.dumps(records)))
                    rows.extend(records)
    return rows


@torch.inference_mode()
def single_visible(model,parent,out,deadline):
    from run_existing_diagnostics import single_visible_reference
    # Independent fresh parent pool; reference stores per-parent G and chain IDs.
    single_visible_reference(parent,out,False)
    with np.load(out / "single_visible_reference.npz") as z:
        moments=z["G_per_parent"]
        reference=moments.mean(0)
    # Reference axis order/shape is validated against the source convention.
    rows=[]
    for axis in (0,1):
        for r in (10,20,50,100,150,180,190,200,210,300,360,380,400,470):
            if time.time()>=deadline: raise TimeoutError("single-visible deadline")
            W=48;idx=(r//10)*W if axis==0 else r//10
            for sign in (-1,1):
                b=confirmation_batch(parent,1,W,"gap",[10],2026092303,0)
                t=np.array([.99],np.float32);mask=np.ones_like(b["clean"],bool)
                mask.reshape(1,-1)[0,idx]=False
                noisy=np.full_like(b["clean"],2);noisy.reshape(1,-1)[0,idx]=(sign+1)//2
                # Force query 0 with deterministic packed arrays; keep visible.
                for rho,policy in ((1.,"uniform"),(1/3,"uniform"),(1/3,"far")):
                    n=W*W;other=np.setdiff1d(np.arange(n),[0,idx])
                    rng=np.random.default_rng(2026092303+axis*1000+r)
                    if policy=="far":
                        cc=b["coords"]["A"].reshape(-1,2)
                        other=other[np.argsort(-((cc[other]-cc[0])**2).sum(1),kind="stable")]
                    else: other=rng.permutation(other)
                    selection=np.r_[0,idx,other[:int(rho*len(other))]]
                    xx=torch.tensor(noisy.reshape(1,-1)[:,selection,None].transpose(0,2,1),device="cuda")
                    cc=torch.tensor(b["coords"]["A"].reshape(1,-1,2)[:,selection][:,None],device="cuda")
                    pred=float(model(xx,torch.tensor(t,device="cuda"),cc).float().softmax(1)[0,1,0,0])
                    # Stored lags are 10,20,...,470 (not direct distance indices).
                    G=float(reference[axis,r//10-1])
                    p=(1+sign*G)/2
                    q=np.clip(pred,1e-7,1-1e-7)
                    kl=p*np.log(max(p,1e-12)/q)+(1-p)*np.log(max(1-p,1e-12)/(1-q))
                    rows.append(dict(axis=axis,r=r,sign=sign,rho=rho,policy=policy,
                                     model_p=pred,reference_p=p,G=G,kl=float(kl),probability_sq_error=(pred-p)**2))
    ev.write_csv(out / "single_visible.csv",rows)
    return dict(mean_kl=float(np.mean([r["kl"] for r in rows])),rows=len(rows))


def evaluate_checkpoint(checkpoint,reference,out,seed,arm,protocol,deadline):
    out.mkdir(parents=True,exist_ok=True)
    done=out / "complete.json"
    if done.exists(): return json.loads(done.read_text())
    model,payload=load_scale_model(checkpoint,torch.device("cuda"))
    parent=load_parent_split(reference,"test_target")
    support_manifest=prepare_reference_evidence(parent,out.parents[1] / "reference_evidence",protocol["generation"])
    (out / "conditional").mkdir(exist_ok=True)
    definitions=[GEN_DEFS[3],GEN_DEFS[4],GEN_DEFS[0],GEN_DEFS[5],GEN_DEFS[6]]
    records=conditionals(model,parent,out / "conditional",seed,definitions,
                         samples=protocol["conditional_samples"],deadline=deadline)
    ev.write_csv(out / "conditional.csv",records)
    groups={}
    for d in definitions:
        for view in ("full","sub"):
            rr=[r for r in records if r["geometry"]==d["name"] and r["view"]==view]
            groups[d["name"]+"/"+view]={k:float(np.mean([r[k] for r in rr])) for k in ("ce","brier","sensitivity")}
    diagnostics=ev.natural_markov(model,parent,"A",out)
    single=single_visible(model,parent,out,deadline)
    generations={}
    previous=ev.make_batch;ev.make_batch=confirmation_batch
    try:
        for definition in protocol["generation"]:
            if time.time()>=deadline:raise TimeoutError("generation deadline")
            generations[definition["name"]]=ev.generate_and_score(model,parent,"A",seed,definition,
                                 out / "generation" / definition["name"],deadline)
    finally: ev.make_batch=previous
    # Keep the parent IDs and per-r support for a predeclared low-support audit.
    long_errors=[];support={}
    for name in ("continuous96","held_gap57_w32","held_s10_w48"):
        folder=out / "generation" / name
        with np.load(folder / "statistics.npz") as z:
            g=z["model_G"];mc=z["mc_G"];counts=z["mc_pair_count"]
        r=np.arange(len(g));valid=np.isin(r,support_manifest["supports"][name]["long_r"])
        if not (np.isfinite(g[valid]).all() and np.isfinite(mc[valid]).all()):
            raise RuntimeError("nonfinite model/reference at frozen support; do not silently drop distances")
        if not valid.any(): raise RuntimeError("insufficient fresh MC support for "+name)
        err=float(np.sqrt(np.mean((g[valid]-mc[valid])**2))/np.sqrt(np.mean(mc[valid]**2)))
        support[name]=dict(r=r[valid].tolist(),nrmse=err)
        long_errors.append(err)
    nested=[]
    for w in (48,64):
        crops=[];mccrops=[]
        for shard in sorted((out / "generation/continuous96").glob("shard_*.npz")):
            with np.load(shard) as z:
                a=(96-w)//2;crops.append(z["spins"][:,a:a+w,a:a+w]);mccrops.append(z["mc"][:,a:a+w,a:a+w])
        xx=np.concatenate(crops);mm=np.concatenate(mccrops)
        axes=[np.broadcast_to(np.arange(w),(len(xx),w))]*2
        s=ev.physical_axis_statistics(xx,axes);m=ev.physical_axis_statistics(mm,axes)
        ev.atomic_npz(out / f"nested96_to{w}.npz",**{"model_"+k:v for k,v in s.items()},**{"mc_"+k:v for k,v in m.items()})
        nested.append(dict(crop_width=w,n=len(xx),model_m2=float(s["m2"].mean()),mc_m2=float(m["m2"].mean())))
    result=dict(status="complete",seed=seed,arm=arm,checkpoint_sha256=gs.file_hash(checkpoint),
                reference_sha256=gs.file_hash(reference),protocol_hash=protocol["protocol_hash"],
                conditional=groups,markov=diagnostics,single_visible=single,generation=generations,primary_support=support,
                J=float(np.mean(long_errors)),nested=nested,
                limitation="final aggregate must distinguish paired training seeds and MC chains; no causal share inferred")
    gs.atomic_json(done,result)
    del model,payload,parent;torch.cuda.empty_cache()
    return result
