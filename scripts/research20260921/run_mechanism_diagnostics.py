"""Paired interventions on frozen dense-RoPE models. No training or generation.

Context membership, rotary frequency, and visible information are independently
controlled. All changes to rotary encoding live in this diagnostic process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import numpy as np
import torch
import ism_diffusion.scale_model as sm
from ism_diffusion.geometry_study import atomic_json, array_hash, coordinate_arrays, file_hash
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model
from evaluate_study import atomic_npz
from run_existing_diagnostics import checkpoint, read_shards
from mechanism_design import (axis_frequencies, ordered_subset, visible_indices,
                              nested_visibility, exact_visible_mask, bernoulli_kl,
                              per_image_scores)

ORIGINAL_ROPE = sm.apply_2d_rope
GEOMETRIES = [("held_s10_w48", 48, "gap", (10,)),
              ("continuous48", 48, "continuous", (1,)),
              ("held_gap57_w32", 32, "gap", (5, 7))]
CONTEXT_DISTANCES = [10, 20, 50, 100, 150, 180, 190, 200, 210, 300, 360, 380, 400, 470]


def status(out, phase, **data):
    row = dict(phase=phase, updated=time.time(), pid=os.getpid(), **data)
    atomic_json(out/"status.json", row)
    print(json.dumps(row), flush=True)


def budget(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Safety budget reached; completed intervention units saved")


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


class RotaryControl:
    def __init__(self):
        self.band = None
        self.factor = 1.

    def configure(self, band=None, factor=1.):
        self.band, self.factor = band, float(factor)

    def __call__(self, values, coordinates, base=10000.):
        half = values.shape[-1]//2
        pieces=[]
        for axis in (0, 1):
            x = values[..., axis*half:(axis+1)*half]
            freq = torch.exp(-math.log(float(base))*torch.arange(0,half,2,
                device=x.device,dtype=torch.float32)/half)
            if self.band is not None:
                freq = freq.clone(); freq[self.band] *= self.factor
            angle=coordinates[...,axis].float()[:,None,:,None]*freq[None,None,None,:]
            c,s=torch.cos(angle).to(x.dtype),torch.sin(angle).to(x.dtype)
            e,o=x[...,0::2],x[...,1::2]
            pieces.append(torch.stack((e*c-o*s,e*s+o*c),-1).flatten(-2))
        return torch.cat(pieces,-1)

    @contextmanager
    def installed(self):
        previous = sm.apply_2d_rope
        sm.apply_2d_rope = self
        try:
            yield
        finally:
            sm.apply_2d_rope = previous


class QueryAttentionRecorder:
    """Recompute ONLY the query row, not the N by N attention matrix."""
    def __init__(self, model, control):
        self.rows=[]; self.metadata=[]; self.control=control
        self.handles=[model.blocks[layer].attention.register_forward_hook(self.hook(layer))
                      for layer in (0, 3, 6)]

    def hook(self, layer):
        def record(module, inputs, output):
            x,coords,valid=inputs
            b,n,_=x.shape
            q,k,v=module.qkv(x).view(b,n,3,module.heads,module.head_dim).unbind(2)
            q=module.q_norm(q).transpose(1,2)
            k=module.k_norm(k).transpose(1,2)
            qr=self.control(q,coords,module.rope_base)
            kr=self.control(k,coords,module.rope_base)
            score=(qr[:,:,0,None,:]*kr).sum(-1)/math.sqrt(module.head_dim)
            score=score.masked_fill(~valid[:,None,:],-torch.inf)
            probability=score.softmax(-1)
            masked=valid.clone(); masked[:,1]=False
            contribution=(qr[:,:,0,:]*kr[:,:,1,:]).reshape(b,module.heads,2,-1,2).sum(-1)/math.sqrt(module.head_dim)
            arrays={
                "visible_logit":score[:,:,1],
                "visible_attention":probability[:,:,1],
                "masked_logsumexp":score.masked_fill(~masked[:,None,:],-torch.inf).logsumexp(-1),
                "entropy":-(probability*probability.clamp_min(1e-30).log()).sum(-1),
                "query_q_norm":q[:,:,0,:].norm(dim=-1),
                "visible_k_norm":k[:,:,1,:].norm(dim=-1),
                "query_attention_output_norm":output[:,0,:].norm(dim=-1)[:,None].expand(-1,module.heads),
            }
            arrays={k:v.detach().cpu().numpy() for k,v in arrays.items()}
            components=contribution.detach().cpu().numpy()
            for i in range(b):
                for head in range(module.heads):
                    row=dict(self.metadata[i],layer=layer,head=head)
                    row.update({key:float(value[i,head]) for key,value in arrays.items()})
                    row.update({f"{axis}_frequency_{band}_visible_logit":float(components[i,head,j,band])
                                for j,axis in enumerate(("x","y")) for band in range(8)})
                    self.rows.append(row)
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def predict(model, tokens, coords, times, valid=None):
    tokens=torch.as_tensor(tokens,device="cuda",dtype=torch.long)
    coords=torch.as_tensor(coords,device="cuda",dtype=torch.float32)
    times=torch.as_tensor(times,device="cuda",dtype=torch.float32)
    vm=None if valid is None else torch.as_tensor(valid,device="cuda",dtype=torch.bool)
    logits=model(tokens,times,coords,vm)
    probabilities=logits.float().softmax(1)[:,1].reshape(len(tokens),-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise AssertionError("nonfinite model probabilities")
    return probabilities.cpu().numpy()


def make_banks(parent,out,smoke):
    dest=out/"inputs";dest.mkdir(exist_ok=True)
    paths=[dest/(g[0]+".npz") for g in GEOMETRIES]
    if all(p.exists() for p in paths):
        return paths
    chains=np.unique(parent.chain_ids)
    random=np.random.default_rng(9232201)
    per_chain=2 if smoke else 32
    ids=np.concatenate([random.choice(np.flatnonzero(parent.chain_ids==chain),per_chain,replace=False)
                        for chain in chains])
    origins=random.integers(parent.lattice_size,size=(len(ids),2))
    for gid,((name,w,kind,gaps),path) in enumerate(zip(GEOMETRIES,paths)):
        coords,axes=coordinate_arrays(len(ids),w,kind,gaps,9232222,gid)
        assert all(int(a.max()) < parent.lattice_size//2 for a in axes)
        xx=(origins[:,0,None]+axes[0])%parent.lattice_size
        yy=(origins[:,1,None]+axes[1])%parent.lattice_size
        clean=(parent.spins[ids[:,None,None],xx[:,:,None],yy[:,None,:]]>0).astype(np.int8)
        trials=1 if smoke else 2
        orders=np.stack([nested_visibility(len(ids),w*w,trial,gid) for trial in range(trials)])
        uniforms=np.stack([np.random.default_rng(9233100+gid*100+trial).random((len(ids),w*w))
                           for trial in range(trials)])
        atomic_npz(path,clean=clean,coords_A=coords["A"],coords_B=coords["B"],coords_C=coords["C"],
                   axis_x=axes[0],axis_y=axes[1],parent=ids,chain=parent.chain_ids[ids],origin=origins,
                   visible_orders=orders,mask_uniforms=uniforms)
    return paths


def invariant_gate(model,control,dest,batch_size):
    if dest.exists():
        return json.loads(dest.read_text())
    rng=np.random.default_rng(92323)
    n=256
    tokens=rng.integers(0,3,size=(2,1,n))
    coords=np.stack(np.meshgrid(np.arange(16),np.arange(16),indexing="ij"),-1).reshape(1,1,n,2)
    coords=np.repeat(coords,2,axis=0).astype(np.float32)
    tv=np.array([.5,.95],np.float32)
    control.configure()
    sm.apply_2d_rope=ORIGINAL_ROPE
    baseline=predict(model,tokens,coords,tv)
    sm.apply_2d_rope=control
    controlled=predict(model,tokens,coords,tv)
    moved=predict(model,tokens,coords+np.array([13.,-7.],np.float32),tv)
    permutation=rng.permutation(n)
    perm=predict(model,tokens[:,:,permutation],coords[:,:,permutation],tv)[:,np.argsort(permutation)]
    padded_tokens=np.concatenate([tokens,np.full((2,1,128),3)],2)
    padded_coords=np.concatenate([coords,rng.uniform(-1000,1000,(2,1,128,2)).astype(np.float32)],2)
    valid=np.ones_like(padded_tokens,bool);valid[:,:,n:]=False
    padded=predict(model,padded_tokens,padded_coords,tv,valid)[:,:n]
    square=predict(model,tokens.reshape(2,16,16),coords.reshape(2,16,16,2),tv)
    recorder=QueryAttentionRecorder(model,control)
    recorder.metadata=[dict(case=i) for i in range(2)]
    try:
        recorded=predict(model,tokens,coords,tv)
    finally:
        recorder.close()
    discrepancies={key:float(np.max(np.abs(value-baseline))) for key,value in {
        "identity_rotary_patch":controlled,"translation":moved,"permutation":perm,
        "ignored_padding":padded,"storage_reshape":square,"recorder_noninterference":recorded}.items()}
    if max(discrepancies.values()) > 2e-5:
        raise AssertionError(f"Preflight invariance failed: {discrepancies}")
    # Synthetic one-pair phase test independent of model weights.
    vector=torch.randn(1,1,3,32,device="cuda")
    position=torch.tensor([[[0.,0.],[19.,0.],[0.,38.]]],device="cuda")
    control.configure(3,1.1);rot=control(vector,position,10000)
    base=ORIGINAL_ROPE(vector,position,10000)
    untouched=[i for i in range(32) if i not in (6,7,22,23)]
    if not torch.equal(rot[...,untouched],base[...,untouched]):
        raise AssertionError("single-frequency intervention changed other dimensions")
    if not torch.allclose(rot.norm(dim=-1),vector.norm(dim=-1),atol=2e-6,rtol=2e-6):
        raise AssertionError("rotation does not preserve Q/K norm")
    control.configure()
    # Benchmark actual primary input shape in FP32, no optimizer or sampling.
    bt=np.full((batch_size,1,2304),2);bt[:,:,1]=1
    bc=np.stack(np.meshgrid(np.arange(48)*10,np.arange(48)*10,indexing="ij"),-1)
    bc=np.broadcast_to(bc.reshape(1,1,2304,2),(batch_size,1,2304,2)).copy()
    times=np.full(batch_size,1-1/2304)
    predict(model,bt,bc,times)
    torch.cuda.synchronize();began=time.perf_counter()
    for _ in range(3):
        predict(model,bt,bc,times)
    seconds=(time.perf_counter()-began)/3
    result=dict(status="passed",discrepancies=discrepancies,
                batch_size=batch_size,fp32_w48_batch_seconds=seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    atomic_json(dest,result)
    return result


def pair_cases(base_coords,distances,count,policy,repetition,clock,moments):
    for axis in (0,1):
        for distance in distances:
            index=visible_indices(48,axis,distance//10)
            ids=ordered_subset(48,index,count,policy,repetition)
            for sign in (-1,1):
                tokens=np.full((1,count),2,np.int64);tokens[0,1]=(sign+1)//2
                coords=base_coords.reshape(-1,2)[ids][None]
                t=1-1/2304 if clock=="fixed" else 1-1/count
                metadata=dict(axis=axis,physical_r=distance,visible_sign=sign,n_tokens=count,
                    layout=policy,repetition=repetition,clock=clock,t=t,
                    reference_p=float(.5*(1+sign*moments[:,axis,distance//10-1].mean())))
                yield tokens,coords,t,metadata


def eval_pairs(model,cases,batch_size,deadline,recorder=None):
    cases=list(cases);rows=[]
    for start in range(0,len(cases),batch_size):
        budget(deadline)
        chunk=cases[start:start+batch_size]
        if recorder:
            recorder.metadata=[r[3] for r in chunk]
        pp=predict(model,np.stack([r[0] for r in chunk]),np.stack([r[1] for r in chunk]),
                   np.array([r[2] for r in chunk]))[:,0]
        for case,p in zip(chunk,pp):
            row=dict(case[3],model_p=float(p),kl=float(bernoulli_kl(case[3]["reference_p"],p)))
            rows.append(row)
    return rows


def save_unit(path,fn):
    if path.exists():
        return json.loads(path.read_text())
    began=time.perf_counter()
    rows=fn()
    result=dict(rows=rows,seconds=time.perf_counter()-began)
    atomic_json(path,result)
    print(json.dumps(dict(unit_complete=str(path),rows=len(rows),seconds=result["seconds"])),flush=True)
    return result


def context_probe(model,control,base_coords,moments,dest,args,deadline):
    distances=[10,190,380] if args.smoke else CONTEXT_DISTANCES
    count_list=[256] if args.smoke else [256,576,1024]
    policies=["uniform"] if args.smoke else ["uniform","near","far"]
    reps=1 if args.smoke else 3
    control.configure()
    full=save_unit(dest/"full.json",lambda:eval_pairs(model,
        pair_cases(base_coords,distances,2304,"uniform",0,"fixed",moments),args.batch_size,deadline))
    baseline={(r["axis"],r["physical_r"],r["visible_sign"]):r["model_p"] for r in full["rows"]}
    units=1
    for count in count_list:
        for policy in policies:
            for rep in range(reps):
                for clock in ("fixed","natural"):
                    path=dest/f"n{count}_{policy}_rep{rep}_{clock}.json"
                    def run():
                        rows=eval_pairs(model,pair_cases(base_coords,distances,count,policy,rep,clock,moments),args.batch_size,deadline)
                        for row in rows:
                            ref=baseline[row["axis"],row["physical_r"],row["visible_sign"]]
                            row.update(full_context_p=ref,delta_p=row["model_p"]-ref)
                        return rows
                    save_unit(path,run);units+=1
    atomic_json(dest/"complete.json",dict(status="complete",units=units))


def frequency_probe(model,control,base_coords,moments,dest,args,deadline):
    distances=[10,180,190,200,210,360,380,400] if args.smoke else list(range(10,471,10))
    variants=[("base",None,1.)]+[(f"f{band}_{factor}",band,factor)
                for band in ([2,3] if args.smoke else range(8)) for factor in (.9,1.1)]
    for name,band,factor in variants:
        control.configure(band,factor)
        save_unit(dest/(name+".json"),lambda:eval_pairs(model,
            pair_cases(base_coords,distances,2304,"uniform",0,"fixed",moments),args.batch_size,deadline))
    control.configure()
    atomic_json(dest/"complete.json",dict(status="complete",units=len(variants)))


def info_states(smoke):
    if smoke:
        return [("k1_natural",1,"natural"),("k8_fixed",8,"fixed"),("rate0.8",.8,"rate")]
    return ([(f"k{k}_natural",k,"natural") for k in (1,2,8,32)]
            +[(f"k{k}_fixed",k,"fixed") for k in (2,8,32)]
            +[(f"rate{t}",t,"rate") for t in (.5,.8,.95)])


def information_probe(model,control,banks,arm,dest,args,deadline):
    units=0
    for gid,path in enumerate(banks):
        with np.load(path) as z:
            a={k:z[k] for k in z.files}
        clean=a["clean"].reshape(len(a["clean"]),-1)
        coords=a["coords_"+arm].reshape(len(clean),1,-1,2)
        n=clean.shape[1];trials=1 if args.smoke else 2
        orders=a["visible_orders"]
        uniforms=a["mask_uniforms"]
        for state,count,clock in info_states(args.smoke):
            for variant,band,factor in (("base",None,1.),("f3_0.9",3,.9),("f3_1.1",3,1.1)):
                control.configure(band,factor)
                def run():
                    rows=[]
                    for trial in range(trials):
                        visible=uniforms[trial]>=count if clock=="rate" else exact_visible_mask(orders[trial],count)
                        masked=~visible
                        t=float(count) if clock=="rate" else (1-1/n if clock=="fixed" else 1-count/n)
                        noisy=np.where(visible,clean,2)[:,None,:]
                        for start in range(0,len(clean),args.batch_size):
                            budget(deadline);sl=slice(start,start+args.batch_size)
                            pp=predict(model,noisy[sl],coords[sl],np.full(len(noisy[sl]),t))
                            scores=per_image_scores(pp,clean[sl],masked[sl])
                            for j in range(len(pp)):
                                i=start+j
                                row=dict(geometry=path.stem,state=state,variant=variant,clock=clock,t=t,
                                    sample=i,trial=trial,chain=int(a["chain"][i]),parent=int(a["parent"][i]),
                                    visible=int(visible[i].sum()),input_hash=array_hash(noisy[i],coords[i],np.array([t],np.float32)))
                                row.update({key:float(value[j]) for key,value in scores.items()})
                                # Fixed bins allow calibration re-analysis without treating pixels as independent replicates.
                                bins=np.minimum((pp[j]*10).astype(int),9)
                                for bin_id in range(10):
                                    select=(bins==bin_id)&masked[i]
                                    row[f"bin{bin_id}_count"]=int(select.sum())
                                    row[f"bin{bin_id}_psum"]=float(pp[j,select].sum())
                                    row[f"bin{bin_id}_ysum"]=int(clean[i,select].sum())
                                rows.append(row)
                    return rows
                save_unit(dest/path.stem/(state+"_"+variant+".json"),run)
                units+=1
    control.configure()
    atomic_json(dest/"complete.json",dict(status="complete",units=units))


def attention_probe(model,control,base_coords,moments,dest,args,deadline):
    units=0
    for name,band,factor in (("base",None,1.),("f3_0.9",3,.9),("f3_1.1",3,1.1),("f2_0.9",2,.9)):
        control.configure(band,factor)
        for count in (256,2304):
            path=dest/(name+f"_n{count}.json")
            def run():
                recorder=QueryAttentionRecorder(model,control)
                try:
                    rows=eval_pairs(model,pair_cases(base_coords,[10,190,200,380],count,"uniform",0,"fixed",moments),
                                    args.batch_size,deadline,recorder)
                    predicted={(r["axis"],r["physical_r"],r["visible_sign"]):r["model_p"] for r in rows}
                    for r in recorder.rows:
                        r["model_p"]=predicted[r["axis"],r["physical_r"],r["visible_sign"]]
                    return recorder.rows
                finally:
                    recorder.close()
            save_unit(path,run);units+=1
    control.configure()
    atomic_json(dest/"complete.json",dict(status="complete",units=units))


def aggregate(out,models):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=out/"figures";figures.mkdir(exist_ok=True)
    totals={}
    for phase in ("context","frequency","information","attention"):
        records=[]
        for seed,arm in models:
            root=out/phase/f"s{seed}_{arm}"
            for path in sorted(root.glob("**/*.json")):
                if path.name=="complete.json":continue
                payload=json.loads(path.read_text())
                for row in payload["rows"]:
                    records.append(dict(seed=seed,arm=arm,unit=str(path.relative_to(root).with_suffix("")),**row))
        frame=pd.DataFrame(records)
        frame.to_csv(out/(phase+"_records.csv"),index=False)
        totals[phase]=len(list((out/phase).glob("s*/complete.json")))
        if phase=="context":
            df=frame[frame.unit!="full"].copy();df["abs_delta_p"]=df.delta_p.abs()
            summary=df.groupby(["seed","arm","n_tokens","layout","clock"],as_index=False)[["abs_delta_p","delta_p","kl"]].mean()
            summary.to_csv(out/"context_by_seed.csv",index=False)
            fig,axes=plt.subplots(1,3,figsize=(12,3.5))
            for ax,arm in zip(axes,"ABC"):
                for policy in ("uniform","near","far"):
                    line=summary[(summary.arm==arm)&(summary.layout==policy)&(summary.clock=="fixed")].groupby("n_tokens").abs_delta_p.mean()
                    if len(line):ax.plot(line.index,line.values,"o-",label=policy)
                ax.set(title=arm,xlabel="Unobserved-set size (including pair)",ylabel="Mean |p(subset)-p(full)|")
                if ax.lines:ax.legend()
            fig.tight_layout();fig.savefig(figures/"context_invariance.png",dpi=160);plt.close(fig)
        elif phase=="frequency":
            frame.groupby(["seed","arm","unit"],as_index=False).kl.mean().to_csv(out/"frequency_by_seed.csv",index=False)
            fig,axes=plt.subplots(1,3,figsize=(13,3.5))
            for ax,arm in zip(axes,"ABC"):
                selected=frame[(frame.arm==arm)&(frame.visible_sign==1)]
                for name in ("base","f3_0.9","f3_1.1","f2_0.9"):
                    line=selected[selected.unit==name].groupby("physical_r").model_p.mean()
                    if len(line):ax.plot(line.index,line.values,label=name)
                ref=selected.groupby("physical_r").reference_p.mean()
                if len(ref):ax.plot(ref.index,ref.values,"k--",label="MC")
                ax.set(title=arm,xlabel="Physical axial distance",ylabel="P(query + | visible +)")
                if ax.lines:ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(figures/"single_band_response.png",dpi=160);plt.close(fig)
        elif phase=="information":
            frame.groupby(["seed","arm","geometry","state","variant"],as_index=False)[["ce","brier"]].mean().to_csv(out/"information_by_seed.csv",index=False)
            fig,axes=plt.subplots(1,3,figsize=(13,3.5))
            baseline=frame[frame.variant=="base"]
            order=["k1_natural","k2_natural","k8_natural","k32_natural","rate0.95","rate0.8","rate0.5"]
            for ax,geometry in zip(axes,[g[0] for g in GEOMETRIES]):
                for arm in "ABC":
                    values=baseline[(baseline.geometry==geometry)&(baseline.arm==arm)].groupby("state").ce.mean().reindex(order)
                    ax.plot(range(len(order)),values,"o-",label=arm)
                ax.set(title=geometry,ylabel="Masked CE (nats; difficulty varies)",xticks=range(len(order)))
                ax.set_xticklabels(order,rotation=65,ha="right",fontsize=7);ax.legend()
            fig.tight_layout();fig.savefig(figures/"information_profiles.png",dpi=160);plt.close(fig)
    return totals


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--study",type=Path,default=ROOT/"artifacts/geometry_alignment_20260921")
    parser.add_argument("--prior-diagnostics",type=Path,default=ROOT/"artifacts/existing_diagnostics_20260922")
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/mechanism_diagnostics_20260922")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--hours",type=float,default=4.)
    parser.add_argument("--batch-size",type=int,default=16)
    args=parser.parse_args()
    out,study=args.out.resolve(),args.study.resolve()
    if out==study or study in out.parents or out==args.prior_diagnostics.resolve():
        raise ValueError("Use a separate diagnostic directory")
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/"queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    models=[(91001,"A")] if args.smoke else [(seed,arm) for seed in range(91001,91007) for arm in "ABC"]
    source=json.loads((study/"final_summary.json").read_text())
    old_protocol=json.loads((args.prior_diagnostics/"frozen_diagnostic_protocol.json").read_text())
    reference=Path(source["reference"]["path"])
    weights={f"s{s}_{a}":file_hash(checkpoint(study,s,a)) for s,a in models}
    if any(weights[key]!=old_protocol["checkpoint_sha256"][key] for key in weights):
        raise AssertionError("Checkpoint differs from completed diagnostic run")
    source_paths=[Path(__file__),Path(__file__).with_name("mechanism_design.py"),
        Path(__file__).with_name("run_existing_diagnostics.py"),Path(__file__).with_name("diagnostic_math.py"),
        Path(__file__).with_name("evaluate_study.py")]+[ROOT/"ism_diffusion"/(n+".py") for n in
        ("geometry_study","scale_model","scale_data","scale_evaluation","model")]
    config=dict(version="mechanism_diagnostics_v1",smoke=args.smoke,models=models,batch_size=args.batch_size,
        source_protocol=source["protocol_hash"],checkpoint_sha256=weights,reference_sha256=file_hash(reference),
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},
        reused_reference_sha256=file_hash(args.prior_diagnostics/"single_visible_reference.npz"),
        context_distances=CONTEXT_DISTANCES,counts=[256,576,1024,2304],policies=["uniform","near","far"],
        context_layout_repetitions=3,context_clocks=["fixed","natural"],
        frequency_bands=list(range(8)),frequency_factors=[.9,1.,1.1],frequency_axes="both, same band; all layers/heads",
        information_geometries=GEOMETRIES,information_parents_per_chain=2 if args.smoke else 32,
        information_trials=1 if args.smoke else 2,information_states=info_states(args.smoke),
        time_budget_hours=args.hours,precision="FP32 highest; no autocast; TF32 disabled",
        primary="A six-seed fixed-time paired context shifts; all-band controlled response; within-state paired CE/Brier",
        warning="Exploratory follow-up of inspected results. Frozen weights. Interventions are not deployable improvements. No new training, no new MC, no free generation.")
    config["hash"]=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=out/"frozen_protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["hash"]!=config["hash"]:
        raise RuntimeError("Protocol changed; use a new output directory")
    atomic_json(manifest,config)
    for path in source_paths:
        target=out/"source_snapshot"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    began=time.time();deadline=began+args.hours*3600
    try:
        status(out,"input_preparation",protocol_hash=config["hash"])
        parent=load_parent_split(reference,"test_target")
        banks=make_banks(parent,out,args.smoke);del parent
        if config["reference_sha256"]!=old_protocol["reference_sha256"]:
            raise AssertionError("Reference pool differs from previous diagnostics")
        input_hashes={p.name:file_hash(p) for p in banks}
        input_manifest=out/"input_manifest.json"
        if input_manifest.exists() and json.loads(input_manifest.read_text())!=input_hashes:
            raise AssertionError("Frozen input bank changed")
        atomic_json(input_manifest,input_hashes)
        shutil.copy2(args.prior_diagnostics/"single_visible_reference.npz",out/"single_visible_reference.npz")
        with np.load(out/"single_visible_reference.npz") as z:moments=z["G_per_parent"]
        for seed,arm in models:
            budget(deadline)
            key=f"s{seed}_{arm}"
            model,payload=load_scale_model(checkpoint(study,seed,arm),torch.device("cuda"));del payload
            model.eval();control=RotaryControl()
            old=study/"evaluation"/key/"generation/held_s10_w48"
            base_coords=read_shards(old,1)["input_coordinates"][0]
            coord_file=out/"inputs"/(key+"_single_visible_coordinates.npz")
            if coord_file.exists():
                with np.load(coord_file) as saved:
                    if not np.array_equal(saved["coordinates"],base_coords):
                        raise AssertionError("Single-visible coordinate source changed")
            else:
                atomic_npz(coord_file,coordinates=base_coords)
            with control.installed(),torch.inference_mode():
                status(out,"gate",seed=seed,arm=arm)
                gate=invariant_gate(model,control,out/"gates"/(key+".json"),args.batch_size)
                print(json.dumps(dict(gate_model=key,**gate)),flush=True)
                for phase,fn in (("context",context_probe),("frequency",frequency_probe)):
                    status(out,phase,seed=seed,arm=arm)
                    fn(model,control,base_coords,moments,out/phase/key,args,deadline)
                status(out,"attention",seed=seed,arm=arm)
                attention_probe(model,control,base_coords,moments,out/"attention"/key,args,deadline)
                status(out,"information",seed=seed,arm=arm)
                information_probe(model,control,banks,arm,out/"information"/key,args,deadline)
            del model;torch.cuda.empty_cache()
            status(out,"model_complete",seed=seed,arm=arm)
        status(out,"aggregation")
        expected_units=dict(context=3 if args.smoke else 55,
                            frequency=5 if args.smoke else 17,
                            information=27 if args.smoke else 90,attention=8)
        for seed,arm in models:
            gate=json.loads((out/"gates"/f"s{seed}_{arm}.json").read_text())
            if gate["status"]!="passed":raise AssertionError("Missing passing preflight")
            for phase,expected_count in expected_units.items():
                directory=out/phase/f"s{seed}_{arm}"
                marker=json.loads((directory/"complete.json").read_text())
                paths=[p for p in directory.glob("**/*.json") if p.name!="complete.json"]
                if marker["units"]!=expected_count or len(paths)!=expected_count:
                    raise AssertionError(f"Unexpected unit count: {directory}")
                if any(not json.loads(p.read_text())["rows"] for p in paths):
                    raise AssertionError("Empty intervention unit")
        totals=aggregate(out,models)
        after={f"s{s}_{a}":file_hash(checkpoint(study,s,a)) for s,a in models}
        if weights!=after:raise AssertionError("Source checkpoint changed")
        if not all(file_hash(ROOT/p)==h for p,h in config["source_sha256"].items()):
            raise AssertionError("Source code changed during run")
        expected={phase:len(models) for phase in ("context","frequency","information","attention")}
        if totals!=expected:raise AssertionError(f"Incomplete phase counts: {totals}")
        final=dict(status="complete",elapsed_hours=(time.time()-began)/3600,models=len(models),
            phases=totals,units_per_model=expected_units,checkpoint_unchanged=True,protocol_hash=config["hash"],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,full_local_backup_verified=False)
        atomic_json(out/"final_summary.json",final)
        status(out,"complete",**{k:v for k,v in final.items() if k!="status"})
    except Exception as error:
        status(out,"budget_stopped" if isinstance(error,TimeoutError) else "failed",error=repr(error))
        traceback.print_exc();raise
    finally:
        sm.apply_2d_rope=ORIGINAL_ROPE


if __name__=="__main__":
    main()
