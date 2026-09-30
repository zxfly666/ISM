"""Wait for verified main evidence, analyze it, then run bounded diagnostics.

No training or sampler changes. Final status still requires local backup and
human/agent visual review; source-frozen primary comparisons precede routing.
"""
from __future__ import annotations
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import numpy as np
from ism_diffusion import geometry_study as gs
import evaluate_study as ev

OUT=ROOT / "artifacts/adaptive_research_20260922"
REPAIR=OUT / "repair"
DEST=OUT / "final_analysis"
SEEDS=list(range(91001,91007));ARMS=("F0","F1","F2")
PRIMARY=("continuous96","held_gap57_w32","held_s10_w48")


def status(stage,**fields):
    gs.atomic_json(DEST / "status.json",dict(stage=stage,time=time.time(),**fields))
    print(json.dumps(dict(stage=stage,**fields)),flush=True)


def curve(s,c):return s.sum(0)/c.sum(0)


def gather(protocol):
    summary=[];all_data={};support=json.loads((REPAIR / "reference_evidence/support.json").read_text())["supports"]
    for seed in SEEDS:
        for arm in ARMS:
            path=REPAIR / f"evaluation/s{seed}_{arm}/complete.json"
            result=json.loads(path.read_text());assert result["status"]=="complete"
            assert result["protocol_hash"]==protocol["protocol_hash"]
            checkpoint=REPAIR / f"training/s{seed}_{arm}/final.pt"
            assert gs.file_hash(checkpoint)==result["checkpoint_sha256"]
            for definition in protocol["generation"]:
                folder=path.parent / "generation" / definition["name"]
                detail=json.loads((folder / "complete.json").read_text())
                assert detail["definition"]==definition
                expected=list(range(0,definition["samples"],16))
                assert [int(p.stem.split("_")[1]) for p in sorted(folder.glob("shard_*.npz"))]==expected
            summary.append(result)
    for name in PRIMARY:
        with np.load(REPAIR / f"reference_evidence/{name}.npz") as z:
            ref={k:z[k] for k in z.files}
        index=np.array(support[name]["long_r"],int)
        assert len(index)>0
        arrays=[];whole=[]
        for seed in SEEDS:
            rows=[];full=[]
            for arm in ARMS:
                with np.load(REPAIR / f"evaluation/s{seed}_{arm}/generation/{name}/statistics.npz") as z:
                    assert np.array_equal(z["mc_pair_sum"],ref["pair_sum"])
                    assert np.array_equal(z["model_pair_count"],ref["pair_count"])
                    rows.append(z["model_pair_sum"][:,index]);full.append(z["model_G"])
            arrays.append(rows);whole.append(full)
        array=np.array(arrays);counts=ref["pair_count"][:,index]
        assert array.shape[2]%16==0
        blocked=array.reshape(6,3,-1,16,len(index)).sum(3)
        cc=counts.reshape(-1,16,len(index)).sum(1)
        all_data[name]=dict(sums=array,blocks=blocked,counts=counts,block_counts=cc,
            mc=ref["pair_sum"][:,index],parent=ref["parent"],index=index,full_curves=np.array(whole),
            full_mc=np.divide(ref["pair_sum"].sum(0),ref["pair_count"].sum(0),out=np.full(ref["pair_sum"].shape[1],np.nan),where=ref["pair_count"].sum(0)>0))
    return summary,all_data


def mc_multiplicity(random,chain,block_length,batch):
    """Crossed chain bootstrap and circular chronological blocks in 128 parents."""
    unique=np.unique(chain);weights=np.zeros((batch,len(chain)))
    pools=[np.flatnonzero(chain==c) for c in unique]
    for b in range(batch):
        for choice in random.integers(len(unique),size=len(unique)):
            pool=pools[choice];n=len(pool)
            starts=random.integers(n,size=math.ceil(n/block_length))
            positions=((starts[:,None]+np.arange(block_length))%n).ravel()[:n]
            np.add.at(weights[b],pool[positions],1.)
    return weights


def conditional_parent_arrays(chain):
    tasks=("held_gap57_w32","held_s10_w48")
    sums=np.zeros((6,3,2,2,len(chain)));count=np.zeros((2,len(chain)))
    for si,seed in enumerate(SEEDS):
        for ai,arm in enumerate(ARMS):
            current=np.zeros_like(count)
            with (REPAIR / f"evaluation/s{seed}_{arm}/conditional.csv").open() as f:
                for row in csv.DictReader(f):
                    if row["view"]!="full" or row["geometry"] not in tasks:continue
                    ti=tasks.index(row["geometry"]);pid=int(row["parent"])
                    sums[si,ai,0,ti,pid]+=float(row["ce"])
                    sums[si,ai,1,ti,pid]+=float(row["sensitivity"])
                    current[ti,pid]+=1
            if si==ai==0:count=current
            else:assert np.array_equal(count,current)
    return sums,count


def crossed_bootstrap(data,chain,conditional,block_length,reps,seed,deadline):
    random=np.random.default_rng(seed);J=[];curves={k:[] for k in data};CE=[];SENS=[]
    cond_sum,cond_count=conditional
    for start in range(0,reps,64):
        if time.time()>=deadline:raise TimeoutError("bootstrap budget")
        B=min(64,reps-start);mw=mc_multiplicity(random,chain,block_length,B)
        picked=random.integers(6,size=(B,6));total=np.zeros((B,6,3))
        for name,d in data.items():
            mcw=mw[:,d["parent"]];reference=(mcw@d["mc"])/(mcw@d["counts"])
            n=d["blocks"].shape[2]
            sw=np.stack([random.multinomial(n,np.ones(n)/n,size=6) for _ in range(B)])
            num=np.einsum("bsk,sakr->bsar",sw,d["blocks"],optimize=True)
            den=np.einsum("bsk,kr->bsr",sw,d["block_counts"],optimize=True)
            gg=num/den[:,:,None,:]
            err=np.sqrt(((gg-reference[:,None,None,:])**2).mean(-1))/np.sqrt((reference**2).mean(-1))[:,None,None]
            total+=err/len(data)
            selected=gg[np.arange(B)[:,None],picked]
            curves[name].append((selected[:,:,2]-selected[:,:,1]).mean(1))
        J.append(total[np.arange(B)[:,None],picked].mean(1))
        denom=np.einsum("bn,tn->bt",mw,cond_count)
        c=np.einsum("bn,saktn->bsakt",mw,cond_sum,optimize=True)/denom[:,None,None,None,:]
        c=c[np.arange(B)[:,None],picked].mean(1)
        CE.append(c[:,:,0,:]);SENS.append(c[:,:,1,:])
    draws=np.concatenate(J);ce=np.concatenate(CE);sens=np.concatenate(SENS)
    valid=np.isfinite(draws).all(1)&np.isfinite(ce).all((1,2))&np.isfinite(sens).all((1,2))
    if valid.mean()<.99:raise RuntimeError("too many unsupported bootstrap resamples; do not ignore")
    dd=draws[valid,2]-draws[valid,1]
    return dict(block_length=block_length,reps=reps,rejected=int((~valid).sum()),
        F2_minus_F1_mean=float(dd.mean()),ci95=np.quantile(dd,[.025,.975]).tolist(),upper_one_sided95=float(np.quantile(dd,.95)),
        full_ce_F2_minus_F1_ci95_by_task=np.quantile(ce[valid,2]-ce[valid,1],[.025,.975],axis=0).T.tolist(),
        full_ce_one_sided95_upper_by_task=np.quantile(ce[valid,2]-ce[valid,1],.95,axis=0).tolist(),
        sensitivity_ratio_ci95=np.quantile(sens[valid,2].mean(1)/sens[valid,1].mean(1),[.025,.975]).tolist(),
        limitations="6 training seeds and 8 MC chains; intervals exploratory at this cluster count; no simultaneous coverage across tasks"),\
        dict(J=draws,ce=ce,sensitivity=sens,valid=valid,**{"curve_"+k:np.concatenate(v) for k,v in curves.items()})


def figures(results,data,draws,out,include_samples=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"axes.spines.top":False,
        "axes.spines.right":False,"legend.frameon":False,"axes.labelsize":15,"savefig.dpi":300})
    colors=["#747474","#0F4D92","#B64342"]
    out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(out/(name+".png"),dpi=300,bbox_inches="tight")
        fig.savefig(out/(name+".pdf"),bbox_inches="tight");plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.4))
    for ax,(name,d) in zip(axs,data.items()):
        rr=np.arange(d["full_curves"].shape[-1]);valid=(rr>0)&np.isfinite(d["full_mc"])
        ax.plot(rr[valid],d["full_mc"][valid],color="black",label="Independent MC",lw=2.2)
        for ai,arm in enumerate(ARMS):
            values=d["full_curves"][:,ai,valid]
            for ss in values:ax.plot(rr[valid],ss,color=colors[ai],alpha=.12,lw=.8)
            ax.plot(rr[valid],values.mean(0),color=colors[ai],lw=2,label=arm)
        ax.set(xlabel="Physical separation r",ylabel="Raw G(r)",title=name.replace("_"," "))
    axs[0].legend(fontsize=11);fig.tight_layout();save(fig,"generation_correlations")
    fig,axs=plt.subplots(1,3,figsize=(15,4.3));bands={}
    for ax,(name,d) in zip(axs,data.items()):
        gg=d["sums"].sum(2)/d["counts"].sum(0)[None,None,:]
        point=(gg[:,2]-gg[:,1]).mean(0);boot=draws["curve_"+name][draws["valid"]]
        se=boot.std(0,ddof=1);active=se>1e-12
        pivot=np.max(np.abs((boot[:,active]-point[active])/se[active]),axis=1) if active.any() else np.zeros(len(boot))
        crit=float(np.quantile(pivot,.95));low=point-crit*se;high=point+crit*se
        ax.axhline(0,color="black",lw=1,ls="--");ax.plot(d["index"],point,color=colors[2])
        ax.fill_between(d["index"],low,high,color=colors[2],alpha=.2)
        ax.set(title=name.replace("_"," "),xlabel="Physical separation r",ylabel="G F2 − G F1")
        bands[name]=dict(r=d["index"].tolist(),point=point.tolist(),low=low.tolist(),high=high.tolist(),
                         definition="within-task 95% max-standardized crossed-bootstrap band; finite-bootstrap estimate")
    fig.tight_layout();save(fig,"paired_curve_differences");gs.atomic_json(out / "curve_bands.json",bands)
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for seed in SEEDS:
        vals=[next(r["J"] for r in results if r["seed"]==seed and r["arm"]==arm) for arm in ARMS]
        ax.plot(range(3),vals,color="#ADB5BD",lw=1,alpha=.7)
        ax.scatter(range(3),vals,c=colors,s=32)
    ax.set(xticks=range(3),xticklabels=ARMS,ylabel="Mean long-range NRMSE J",title="All six paired training seeds")
    fig.tight_layout();save(fig,"paired_primary_metric")
    if not include_samples:return
    fig,axs=plt.subplots(3,4,figsize=(10,7.6))
    for row,arm in enumerate(("MC","F1","F2")):
        path=REPAIR / f"evaluation/s91001_{'F1' if arm=='MC' else arm}/generation/continuous96/shard_00000.npz"
        with np.load(path) as z:images=z["mc" if arm=="MC" else "spins"][:4]
        for col in range(4):
            axs[row,col].imshow(images[col],cmap="RdBu_r",vmin=-1,vmax=1,interpolation="nearest")
            axs[row,col].set(xticks=[],yticks=[])
            if col==0:axs[row,col].set_ylabel(arm)
    fig.suptitle("First four saved samples — not selected by appearance",fontsize=13)
    fig.tight_layout();save(fig,"unselected_samples")


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    lock=(DEST / "queue.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT / "final_preflight/complete.json").read_text())["status"]=="passed"
    source_paths=["scripts/research20260921/finalize_adaptive_campaign.py",
                  "scripts/research20260921/adaptive_trajectory_diagnostics.py"]
    frozen=DEST / "protocol.json"
    if frozen.exists():
        for p,h in json.loads(frozen.read_text())["source_hashes"].items():
            if gs.file_hash(ROOT / p)!=h:raise RuntimeError("analysis source changed after freeze: "+p)
    else:
        gs.atomic_json(frozen,dict(version="analysis_v1",created=time.time(),
            source_hashes={p:gs.file_hash(ROOT / p) for p in source_paths},
            mc_block_lengths=[8,4,16],bootstrap_reps=[2000,1000,1000],
            generator_cluster="16-image saved shard; common weights across paired arms",
            training_unit="six training seeds; paired arms preserved",
            mc_unit="8 MC chains crossed with seeds, then circular blocks over chronological 128 parents",
            primary="F2-F1 mean three-task long-range NRMSE; conditional CE guardrail reported separately",
            followup="least F2-over-F1 task benefit, all six seeds and F1/F2, 8 trajectories each, only if >=20min remains",
            claim="development-selected architecture; new reference confirmation; follow-up exploratory"))
    campaign=json.loads((OUT / "campaign.json").read_text());deadline=campaign["deadline"]
    status("waiting_main")
    while not (REPAIR / "main_summary.json").exists():
        if time.time()>=deadline:raise TimeoutError("campaign budget reached before final analysis")
        p=REPAIR / "status.json"
        if p.exists() and json.loads(p.read_text())["stage"] in ("failed","budget_stop"):
            raise RuntimeError("upstream stopped; no automatic restart")
        time.sleep(30)
    protocol=json.loads((REPAIR / "protocol.json").read_text())
    status("verify_and_crossed_analysis")
    results,data=gather(protocol)
    refpath=OUT / "reference/fresh_l1024.npz"
    with np.load(refpath) as z:chain=z["test_target_chain_id"]
    conditional=conditional_parent_arrays(chain)
    stats=[];primary_draws=None
    for length,reps in ((8,2000),(4,1000),(16,1000)):
        ss,draws=crossed_bootstrap(data,chain,conditional,length,reps,2026092405+length,deadline)
        stats.append(ss);ev.atomic_npz(DEST / f"bootstrap_block{length}.npz",**draws)
        if length==8:primary_draws=draws
    gs.atomic_json(DEST / "crossed_uncertainty.json",stats)
    table=[]
    for r in results:
        table.append(dict(seed=r["seed"],arm=r["arm"],J=r["J"],
            full_CE=np.mean([r["conditional"][x+"/full"]["ce"] for x in PRIMARY[1:]]),
            sensitivity=np.mean([r["conditional"][x+"/full"]["sensitivity"] for x in PRIMARY[1:]]),
            **{name+"_long_NRMSE":r["primary_support"][name]["nrmse"] for name in PRIMARY}))
    ev.write_csv(DEST / "per_seed_results.csv",table)
    figures(results,data,primary_draws,DEST / "figures")
    # Read main outcomes before routing a new, explicitly exploratory experiment.
    differences={name:float(np.mean([r["primary_support"][name]["nrmse"] for r in results if r["arm"]=="F2"])-
                                 np.mean([r["primary_support"][name]["nrmse"] for r in results if r["arm"]=="F1"])) for name in PRIMARY}
    target=max(differences,key=differences.get)
    reasoning="Probe the task with least F2-over-F1 generation benefit to test whether consistency shifts between MC and generated contexts. All six seeds and both arms, not only a winning checkpoint."
    route=dict(target=target,task_differences=differences,reason=reasoning,adaptive_exploratory=True,
               run_trajectories=deadline-time.time()>1200,source_sha256=gs.file_hash(Path(__file__)))
    gs.atomic_json(DEST / "followup_decision.json",route)
    if route["run_trajectories"]:
        status("adaptive_trajectory_followup",decision=route)
        import torch
        from ism_diffusion.scale_data import load_parent_split
        from adaptive_trajectory_diagnostics import run
        torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
        parent=load_parent_split(refpath,"test_target")
        definition=next(d for d in protocol["generation"] if d["name"]==target)
        for seed in SEEDS:
            for arm in ("F1","F2"):
                run(REPAIR / f"training/s{seed}_{arm}/final.pt",parent,DEST / f"trajectories/s{seed}_{arm}",
                    seed,definition,deadline,samples=8)
        del parent
        trajectory_rows=[]
        for p in sorted((DEST / "trajectories").glob("s*/*/rows.csv")):
            arm=p.parents[1].name.split("_")[1];seed=int(p.parents[1].name[1:6])
            with p.open() as f:
                for rr in csv.DictReader(f):trajectory_rows.append(dict(seed=seed,arm=arm,**rr))
        ev.write_csv(DEST / "trajectory_summary.csv",trajectory_rows)
    base_checks={s:gs.file_hash((ROOT / "artifacts/geometry_alignment_20260921" if protocol["selected_size"]=="S" else OUT / "capacity") /
          f"training/s{s}_{'A' if protocol['selected_size']=='S' else 'M'}/final.pt")==digest for s,digest in protocol["base_hashes"].items()}
    assert all(base_checks.values())
    manifest=[]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json","queue.lock") and path.suffix not in (".tmp",):
            # Include immutable scientific evidence; avoid hashing actively changing queue logs.
            if path.suffix==".log" or path.name=="status.json":continue
            manifest.append(dict(path=str(path.relative_to(OUT)),bytes=path.stat().st_size,sha256=gs.file_hash(path)))
    gs.atomic_json(DEST / "manifest.json",manifest)
    result=dict(status="remote_complete_requires_local_backup_and_visual_review",models=len(results),
        generated_samples=sum(d["samples"] for d in protocol["generation"])*len(results),
        base_checkpoints_unchanged=all(base_checks.values()),crossed_uncertainty=stats,
        followup=route,elapsed_hours=(time.time()-campaign["started"])/3600,
        still_required=["download and hash-check necessary artifacts","visually inspect final figures","researcher interpretation; no automatic claim of success"],
        omitted_from_broader_proposal=["fixed-background two-visible geometry probe","dedicated K=2/8/32 benchmark"],
        claim_boundary="S/M is development selection; F2-F1 is paired repair comparison; adaptive trajectory followup is exploratory")
    gs.atomic_json(OUT / "final_summary.json",result);status(result["status"],summary=result)


if __name__=="__main__":
    try:main()
    except TimeoutError as exc:status("budget_stop",error=str(exc))
    except Exception as exc:status("failed",error=repr(exc),traceback=traceback.format_exc());raise
