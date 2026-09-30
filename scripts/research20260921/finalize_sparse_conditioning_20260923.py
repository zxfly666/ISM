"""Crossed uncertainty, immutable evidence audit, and honest summary figures."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from sparse_conditioning_design import SEEDS,ARMS,GEN_DEFS,COND_DEFS
from fixed_geometry_math import metrics,query_specs,symmetric_distribution
from analyze_fixed_geometry_joint_20260923 import mc_weights,atomic_json,file_hash
from run_fixed_geometry_joint_20260923 import npz
from evaluate_study import write_csv

GROUPS=("matched_continuous48","matched_continuous96","matched_continuous128","unequal_distance_w48")
TMETRICS=("sequential_kl","parallel_kl","oracle_parallel_kl","marginal_kl","order_tv","geometry_probability_mae","response_rmse")
BANDS={"continuous128":((1,8),(9,48),(49,64)),"continuous96":((1,8),(9,24),(25,48)),
       "continuous48":((1,8),(9,24),(1,24)),"held_s10_w48":((10,30),(40,120),(130,470))}


def triplet_scores(pred,p,cases):
    keys=[];values=[]
    for ci,clock in enumerate(("natural","common")):
        mm=metrics(pred[:,:,:,ci],p,query_specs(2304))
        for group in GROUPS:
            ix=np.array([c["geometry"]==group for c in cases])
            for name in TMETRICS:
                v=np.sqrt(mm["response_squared_error"][...,ix].mean(-1)) if name=="response_rmse" else mm[name][...,ix].mean(-1)
                values.append(np.broadcast_to(v,pred.shape[:2]));keys.append(f"{group}/{clock}/{name}")
    return np.stack(values,-1),keys


def load(root):
    cases=json.loads((root/"cases.json").read_text());pred=[];cond=[];generations=[]
    for arm in ARMS:
        ap=[];ac=[];ag=[]
        for seed in SEEDS:
            d=root/"evaluation"/f"s{seed}_{arm}"
            status=json.loads((d/"complete.json").read_text())
            assert status["checkpoint_unchanged"] and status["generation_samples"]==320
            with np.load(d/"triplet_predictions.npz") as z:ap.append(z["probabilities"].astype(float))
            ac.append([dict(np.load(d/(t["name"]+".npz"))) for t in COND_DEFS])
            sg=[]
            for t in GEN_DEFS:
                gd=d/"generation"/t["name"]
                complete=json.loads((gd/"complete.json").read_text())
                assert complete["definition"]["samples"]==t["samples"]
                files=sorted(gd.glob("shard_*.npz"));assert len(files)==t["samples"]//16
                with np.load(gd/"statistics.npz") as z:
                    data={k[6:]:z[k] for k in z.files if k.startswith("model_") and k!="model_G"}
                from ism_diffusion.scale_evaluation import open_energy_density
                data["energy"]=np.concatenate([open_energy_density(np.load(f)["spins"]) for f in files])
                sg.append(data)
            ag.append(sg)
        pred.append(ap);cond.append(ac);generations.append(ag)
    with np.load(root/"reference_evidence/triplets.npz") as z:ref={k:z[k] for k in z.files}
    gr=[dict(np.load(root/"reference_evidence"/(d["name"]+".npz"))) for d in GEN_DEFS]
    pred=np.array(pred);assert pred.shape==(3,6,116,2,20,3)
    # Same physical inputs across all models, not merely the same case labels.
    for filename,key in [("triplet_predictions.npz","input_hash")]+[(d["name"]+".npz","input_sha256") for d in COND_DEFS]:
        values=[np.load(root/"evaluation"/f"s{s}_{a}"/filename)[key] for a in ARMS for s in SEEDS]
        assert all(np.array_equal(values[0],v) for v in values)
    return cases,pred,cond,generations,ref,gr


def scores(cases,pred,cond,gen,ref,gr,weights=None,random=None):
    weights=np.ones(len(ref["chain"])) if weights is None else weights
    counts=np.einsum("cpi,p->ci",ref["counts"],weights,optimize=True)
    p=symmetric_distribution(counts)
    values,keys=triplet_scores(pred,p,cases)
    columns=[values];extra=[];ek=[]
    for ti,t in enumerate(COND_DEFS):
        col=np.empty((3,6))
        for a in range(3):
            for s in range(6):
                rr=cond[a][s][ti];ww=weights[rr["parent"]]
                if ww.sum()==0:raise RuntimeError("No held-out conditional parents in bootstrap; no silent redraw")
                col[a,s]=np.average(rr["ce"],weights=ww)
        extra.append(col);ek.append(t["name"]+"/CE")
    for gi,d in enumerate(GEN_DEFS):
        reference=gr[gi];cw=weights[reference["parent"]]
        refsum=cw@reference["pair_sum"];refcount=cw@reference["pair_count"]
        rg=np.divide(refsum,refcount,out=np.zeros_like(refsum),where=refcount>0)
        cols=np.empty((3,6,7))
        for s in range(6):
            n=d["samples"];nshard=n//16
            iw=np.ones(n) if random is None else np.repeat(np.bincount(random.integers(nshard,size=nshard),minlength=nshard),16)
            for a in range(3):
                g=gen[a][s][gi];cc=iw@g["pair_count"];ss=iw@g["pair_sum"]
                gg=np.divide(ss,cc,out=np.zeros_like(ss),where=cc>0)
                rr=np.arange(len(rg))
                for bi,(lo,hi) in enumerate(BANDS[d["name"]]):
                    ix=(rr>=lo)&(rr<=hi)&(refcount>0)&(cc>0)
                    assert ix.any()
                    cols[a,s,bi]=np.sqrt(np.mean((gg[ix]-rg[ix])**2))/np.sqrt(np.mean(rg[ix]**2))
                for mi,key in enumerate(("m","m2","abs_m","energy"),3):
                    cols[a,s,mi]=np.average(g[key],weights=iw)-np.average(reference[key],weights=cw)
        for bi,(lo,hi) in enumerate(BANDS[d["name"]]):ek.append(f"{d['name']}/G_{lo}_{hi}_NRMSE")
        ek.extend(f"{d['name']}/{key}_bias" for key in ("m","m2","abs_m","energy"))
        extra.extend(np.moveaxis(cols,-1,0))
    result=np.concatenate([values,np.stack(extra,-1)],-1)
    assert np.isfinite(result).all()
    return result,keys+ek


def plot(out,point,keys,uncertainty,pred,ref,cases,gen,gr):
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.spines.top":False,
        "axes.spines.right":False,"axes.linewidth":1.5,"legend.frameon":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    colors=["#767676","#0F4D92","#42949E"];dest=out/"figures";dest.mkdir()
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    primary=["matched_continuous128/natural/sequential_kl","continuous128/G_49_64_NRMSE","w48_t095/CE"]
    for ax,key,title in zip(axs,primary,("W128 two-query KL","W128 long-range G error","W48 ordinary-mask CE (t=.95)")):
        idx=keys.index(key)
        for s in range(6):ax.plot(range(3),point[:,s,idx],color="#CFCECE",lw=.8,zorder=1)
        for a,arm in enumerate(ARMS):ax.scatter(np.full(6,a),point[a,:,idx],color=colors[a],s=35,label=arm,zorder=2)
        ax.set_xticks(range(3),ARMS);ax.set_title(title);ax.set_ylabel("Lower is better")
    fig.suptitle("All six paired training seeds; final EMA, no checkpoint selection")
    fig.tight_layout(pad=1.8)
    for ext in ("png","pdf"):fig.savefig(dest/f"primary_paired_seeds.{ext}",dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,label in ((axs[0],"sequential_kl","Two-query KL (nats)"),(axs[1],"order_tv","Order disagreement (TV)")):
        for a,arm in enumerate(ARMS):
            ix=[keys.index(f"matched_continuous{w}/natural/{metric}") for w in (48,96,128)]
            vals=point[a,:,ix].T
            ax.plot([48,96,128],point[a][:,ix].mean(0),marker="o",color=colors[a],label=arm)
        ax.set(xlabel="Context width",ylabel=label);ax.legend()
    fig.tight_layout(pad=1.8)
    for ext in ("png","pdf"):fig.savefig(dest/f"accuracy_and_consistency.{ext}",dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,gi in zip(axs,(0,1)):
        d=GEN_DEFS[gi];r=np.arange(gr[gi]["pair_sum"].shape[1]);ix=(r>=1)&(r<=d["width"]//2)
        truth=gr[gi]["pair_sum"].sum(0)/np.maximum(gr[gi]["pair_count"].sum(0),1)
        ax.plot(r[ix],truth[ix],color="black",lw=2,label="Independent MC")
        for a,arm in enumerate(ARMS):
            curves=np.stack([gen[a][s][gi]["pair_sum"].sum(0)/np.maximum(gen[a][s][gi]["pair_count"].sum(0),1) for s in range(6)])
            for curve in curves:ax.plot(r[ix],curve[ix],color=colors[a],alpha=.2,lw=.8)
            ax.plot(r[ix],curves.mean(0)[ix],color=colors[a],lw=2,label=arm)
        ax.set(xlabel="Physical distance r",ylabel="Axis correlation G(r)",title=d["name"]);ax.legend()
    fig.suptitle("S0 256-step sampling; thin lines show training-seed variation")
    fig.tight_layout(pad=1.8)
    for ext in ("png","pdf"):fig.savefig(dest/f"generation_correlations.{ext}",dpi=300)
    plt.close(fig)


def main(root,deadline):
    started=time.time();out=root/"analysis";out.mkdir(exist_ok=False)
    cases,pred,cond,gen,ref,gr=load(root)
    point,keys=scores(cases,pred,cond,gen,ref,gr)
    assert len(keys)==len(set(keys))
    rows=[dict(arm=arm,seed=seed,metric=key,value=float(point[a,s,m])) for a,arm in enumerate(ARMS)
          for s,seed in enumerate(SEEDS) for m,key in enumerate(keys)]
    write_csv(out/"per_seed_metrics.csv",rows)
    uncertainty={};effects=[]
    for block,reps in ((8,1000),(4,500),(16,500)):
        random=np.random.default_rng(2026092410+block);boot=[]
        for i in range(reps):
            if time.time()>=deadline:raise TimeoutError("statistics deadline; no silent reduced replicates")
            w=mc_weights(ref["chain"],block,random)
            vals,_=scores(cases,pred,cond,gen,ref,gr,w,random)
            seed_ids=random.integers(6,size=6);boot.append(vals[:,seed_ids].mean(1))
            if i%100==0:print(json.dumps(dict(stage="bootstrap",block=block,rep=i)),flush=True)
        boot=np.stack(boot);npz(out/f"bootstrap_block{block}.npz",values=boot,metrics=np.array(keys),arms=np.array(ARMS))
        contrasts={}
        for label,a,b in (("T1_minus_T0",1,0),("T2_minus_T1",2,1),("T2_minus_T0",2,0)):
            estimates=point[a].mean(0)-point[b].mean(0);ci=np.quantile(boot[:,a]-boot[:,b],[.025,.975],axis=0)
            contrasts[label]={key:dict(estimate=float(estimates[m]),ci95=ci[:,m].tolist()) for m,key in enumerate(keys)}
            effects.extend(dict(block=block,contrast=label,metric=key,estimate=float(estimates[m]),
                                 ci_low=float(ci[0,m]),ci_high=float(ci[1,m])) for m,key in enumerate(keys))
        uncertainty[str(block)]=dict(replicates=reps,contrasts=contrasts)
        atomic_json(out/"crossed_uncertainty.json",uncertainty)
    write_csv(out/"paired_contrasts.csv",effects)
    npz(out/"plot_data.npz",per_seed=point,predictions=pred,reference_p=ref["reference_p"],metrics=np.array(keys))
    plot(out,point,keys,uncertainty,pred,ref,cases,gen,gr)
    primary=uncertainty["8"]["contrasts"]["T1_minus_T0"]
    ck="matched_continuous128/natural/sequential_kl";gk="continuous128/G_49_64_NRMSE"
    retained=all(primary[k]["ci95"][1]<=.01 for k in ("w48_t05/CE","w48_t095/CE"))
    success=primary[ck]["ci95"][1]<0 and primary[gk]["ci95"][1]<0 and retained
    proto=json.loads((root/"run_protocol.json").read_text())
    assert all(file_hash(ROOT/f)==h for f,h in proto["source_hashes"].items())
    assert all(file_hash(Path(f))==h for f,h in proto["base_hashes"].items())
    for seed in SEEDS:
        logs=[root/"training"/f"s{seed}_{a}"/"train.jsonl" for a in ARMS]
        rows=[[json.loads(l) for l in p.read_text().splitlines() if '"main_hash"' in l] for p in logs]
        assert [r["step"] for r in rows[0]]==[r["step"] for r in rows[1]]==[r["step"] for r in rows[2]]
        for rr in zip(*rows):
            for k in ("main_hash","aux_clean_hash","sparse_input_hash"):
                assert rr[0][k]==rr[1][k]==rr[2][k]
    summary=dict(status="remote_complete_requires_backup_and_visual_review",models=18,sampling_tasks=72,
        generation_samples=5760,checkpoint_unchanged=True,training_pairing_verified=True,
        conditional_primary=primary[ck],generation_primary=primary[gk],retention_gate=retained,
        T1_end_to_end_gate_passed=success,T2_comparisons=uncertainty["8"]["contrasts"]["T2_minus_T1"],
        elapsed_hours=(time.time()-proto["started"])/3600,analysis_seconds=time.time()-started,
        limitations=["T2 uses physics-assisted labels; cannot replace T1 primary claim",
          "W96 is trained size; W128 unseen context is not unseen every physical distance",
          "fixed training pool and inherited initial checkpoints; fixed query bank; limited generation shards",
          "MC confidence is chain/time-block based; plotted thin curves show training-seed spread, not CI",
          "conditional accuracy is not automatically joint generation accuracy"])
    atomic_json(root/"final_summary.json",summary)
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=file_hash(p))
             for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("manifest.json","queue.log","status.json","run.lock")]
    atomic_json(root/"manifest.json",dict(files=entries,created=time.time(),scope="new campaign only"))
    return summary


ROOT=Path(__file__).resolve().parents[2]
