"""Frozen held-out banks and unchanged-sampler evaluation for R00/R01/R10/R11."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_evaluation import load_scale_model,open_energy_density
from mask_query_factorial_design import GEN_DEFS,COND_DEFS,sparse_view,make_cases
from fixed_geometry_math import case_inputs,query_specs,metrics,symmetric_distribution
from run_fixed_geometry_joint_20260923 import npz,predict
from adaptive_repair_evaluation import confirmation_batch
import evaluate_study as ev


def budget(deadline):
    if time.time()>=deadline:raise TimeoutError("Mask-query factorial campaign hard deadline; no automatic restart")


def log(out,stage,**kw):
    row=dict(stage=stage,time=time.time(),**kw);gs.atomic_json(out/"status.json",row)
    print(json.dumps(row),flush=True)


def condition_bank(parent,definition):
    w=definition["width"];n=128;tag=COND_DEFS.index(definition)
    b=confirmation_batch(parent,n,w,"continuous",[1],2026092413,tag*10000)
    # One observation per selected parent, spread across each chain's time axis.
    chains=np.unique(parent.chain_ids)
    ids=np.array([np.flatnonzero(parent.chain_ids==chains[i%8])[(i//8)*8] for i in range(n)])
    xx=(b["origin"][:,0,None]+b["axes"][0])%parent.lattice_size
    yy=(b["origin"][:,1,None]+b["axes"][1])%parent.lattice_size
    clean=(parent.spins[ids[:,None,None],xx[:,:,None],yy[:,None,:]]>0).astype(np.int64)
    if "k" in definition:
        sp=sparse_view(clean,b["coords"]["A"],2026092413,tag,fixed_k=definition["k"])
        noisy,t,q,y=sp["noisy"],sp["t"],sp["queries"],sp["labels"]
    else:
        t,mask,noisy=gs.corrupt_batch(clean,2026092413,tag,definition["t"])
        mask=mask.reshape(n,-1);noisy=noisy.reshape(n,-1)
        random=gs.rng(2026092413,tag,"heldout_queries")
        q=np.stack([random.choice(np.flatnonzero(row),64,replace=False) for row in mask])
        y=clean.reshape(n,-1)[np.arange(n)[:,None],q]
    return dict(noisy=noisy[:,None],coords=b["coords"]["A"].reshape(n,1,w*w,2),t=t,
        queries=q,labels=y,parent=ids,chain=parent.chain_ids[ids],origin=b["origin"])


def prepare_reference(parent,out,cases,deadline):
    dest=out/"reference_evidence";dest.mkdir(parents=True,exist_ok=False)
    n=len(parent.spins);assert n==1024 and len(np.unique(parent.chain_ids))==8
    origins=np.random.default_rng(2026092414).integers(parent.lattice_size,size=(n,256,2))
    ids=np.arange(n)[:,None];counts=[];cache={}
    for c in cases:
        budget(deadline);key=tuple(map(tuple,c["physical_sites"]))
        if key not in cache:
            values=[(parent.spins[ids,(origins[...,0]+x)%parent.lattice_size,
                                 (origins[...,1]+y)%parent.lattice_size]>0).astype(np.int8) for x,y in key]
            code=values[0]+2*values[1]+4*values[2]
            cache[key]=np.stack([(code==j).sum(1) for j in range(8)],-1).astype(np.int32)
        counts.append(cache[key])
    counts=np.stack(counts);p=symmetric_distribution(counts.sum(1));assert p.min()>0
    assert np.array_equal(counts[:32],counts[32:64]) and np.array_equal(counts[:32],counts[64:96])
    npz(dest/"triplets.npz",counts=counts,reference_p=p,origins=origins,chain=parent.chain_ids,parent=np.arange(n))
    for definition in COND_DEFS:
        budget(deadline);npz(dest/(definition["name"]+".npz"),**condition_bank(parent,definition))
    for gi,d in enumerate(GEN_DEFS):
        shards=[];parent_ids=[];energies=[];crop_origins=[]
        for start in range(0,n,16):
            budget(deadline)
            b=confirmation_batch(parent,16,d["width"],d["kind"],d["gaps"],2026092415,gi*100000+start)
            spins=(2*b["clean"]-1).astype(np.int8)
            shards.append(ev.physical_axis_statistics(spins,b["axes"]))
            parent_ids.extend(b["parent"]);crop_origins.extend(b["origin"])
            energies.extend(open_energy_density(spins))
        stats=ev.merge_statistics(shards);parent_ids=np.array(parent_ids)
        assert len(np.unique(parent_ids))==n
        # Indexed by canonical source parent, shared across all statistical outputs.
        order=np.argsort(parent_ids)
        npz(dest/(d["name"]+".npz"),**{k:v[order] for k,v in stats.items()},
            energy=np.array(energies)[order],origin=np.array(crop_origins)[order],
            parent=parent_ids[order],chain=parent.chain_ids[parent_ids[order]])
        log(out,"reference_task_prepared",task=d["name"])
    gs.atomic_json(dest/"complete.json",dict(status="complete",cases=len(cases),parents=n,
        files={f.name:gs.file_hash(f) for f in dest.glob("*.npz")}))


@torch.inference_mode()
def evaluate_model(path,parent,out,root,seed,arm,deadline):
    out.mkdir(parents=True,exist_ok=False);before=gs.file_hash(path)
    model,payload=load_scale_model(path,torch.device("cuda"));model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==1976706
    cases=json.loads((root/"cases.json").read_text())
    with np.load(root/"reference_evidence/triplets.npz") as z:p=z["reference_p"]
    torch.set_float32_matmul_precision("highest")
    predictions=[];hashes=[];rows=[]
    for i,c in enumerate(cases):
        budget(deadline);x,coords,t,specs=case_inputs(c,"A")
        clocks=[];hh=[]
        for clock in ("natural","common"):
            times=t if clock=="natural" else np.full_like(t,1-2/2304)
            pp=predict(model,x,coords,times,1 if c["width"]==128 else 2)
            mm=metrics(pp,p[i],specs)
            assert float(mm["swap_permutation_error"])<2e-5
            clocks.append(pp);hh.append(gs.array_hash(x,coords,times))
            rows.append(dict(case=i,clock=clock,**{k:float(v) for k,v in mm.items()}))
        predictions.append(clocks);hashes.append(hh)
        if i%32==0:log(root,"triplet_evaluation",seed=seed,arm=arm,case=i,total=len(cases))
    npz(out/"triplet_predictions.npz",probabilities=np.stack(predictions),input_hash=np.array(hashes))
    gs.atomic_json(out/"triplet_metrics.json",rows)
    for definition in COND_DEFS:
        budget(deadline)
        with np.load(root/"reference_evidence"/(definition["name"]+".npz")) as z:
            data={k:z[k] for k in z.files}
        probabilities=[]
        for start in range(0,len(data["t"]),2):
            budget(deadline);sl=slice(start,start+2)
            logits=model(torch.as_tensor(data["noisy"][sl],device="cuda"),
                torch.as_tensor(data["t"][sl],device="cuda"),torch.as_tensor(data["coords"][sl],device="cuda"))
            q=torch.as_tensor(data["queries"][sl],device="cuda")
            prob=logits.float().softmax(1)[:,1,0].gather(1,q).cpu().numpy()
            assert np.isfinite(prob).all();probabilities.append(prob)
        pp=np.concatenate(probabilities).astype(float);yy=data["labels"]
        ce=-(yy*np.log(np.clip(pp,1e-12,1))+(1-yy)*np.log(np.clip(1-pp,1e-12,1))).mean(1)
        npz(out/(definition["name"]+".npz"),probability=pp,ce=ce,brier=((pp-yy)**2).mean(1),
            parent=data["parent"],chain=data["chain"],input_sha256=np.array(gs.array_hash(data["noisy"],data["coords"],data["t"],data["queries"])))
    torch.set_float32_matmul_precision("high")
    original=ev.make_batch;ev.make_batch=confirmation_batch
    try:
        for d in GEN_DEFS:
            budget(deadline);log(root,"generation",seed=seed,arm=arm,task=d["name"])
            ev.generate_and_score(model,parent,"A",seed,d,out/"generation"/d["name"],deadline,16)
    finally:ev.make_batch=original
    after=gs.file_hash(path);assert before==after
    gs.atomic_json(out/"complete.json",dict(status="complete",seed=seed,arm=arm,cases=len(cases),
        clocks=2,conditional_tasks=6,generation_tasks=len(GEN_DEFS),generation_samples=sum(d["samples"] for d in GEN_DEFS),
        checkpoint_sha256=before,checkpoint_unchanged=True,step=payload["step"],
        warning="generation task complete.json has paired-crop descriptive MC; formal analysis uses all 1024 reference parents"))
    del model,payload;torch.cuda.empty_cache()
