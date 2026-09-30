"""Frozen exploratory analysis, crossed seed/MC bootstrap, and report figures."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import sys
import csv
import hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
from fixed_geometry_math import metrics,query_specs,symmetric_distribution


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temporary,path)


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): digest.update(block)
    return digest.hexdigest()

ARMS=["A","B","C","F0","F1","F2"]
SEEDS=list(range(91001,91007))
GROUPS=["unequal_distance_w48","continuous48","s10_w48","continuous96"]
METRICS=["response_rmse","geometry_probability_mae","balanced_conditional_kl","sequential_kl",
         "parallel_kl","oracle_parallel_kl","marginal_kl","order_tv","fixed_clock_order_tv",
         "fixed_clock_sequential_kl","response_gain"]


def aggregate(pred,p,cases):
    mm=metrics(pred,p,query_specs(2304))
    groups=[]
    for group in GROUPS:
        ix=np.array([c["geometry"]==group for c in cases])
        columns=[]
        for name in METRICS:
            if name=="response_rmse": v=np.sqrt(mm["response_squared_error"][...,ix].mean(-1))
            elif name=="response_gain":
                target=mm["reference_response"][ix]
                v=(mm["response"][...,ix]*target).sum(-1)/max(float((target**2).sum()),1e-12)
            else:
                v=mm[name][...,ix].mean(-1)
                v=np.broadcast_to(v,pred.shape[:2])
            columns.append(v)
        groups.append(np.stack(columns,-1))
    return np.stack(groups) # group, arm, seed, metric


def mc_weights(chain,block,random):
    ids=np.unique(chain);draws=[]
    for c in random.choice(ids,len(ids),replace=True):
        members=np.flatnonzero(chain==c)
        starts=random.integers(0,len(members),size=int(np.ceil(len(members)/block)))
        take=((starts[:,None]+np.arange(block))%len(members)).ravel()[:len(members)]
        draws.extend(members[take])
    return np.bincount(draws,minlength=len(chain)).astype(float)


def bootstrap(pred,counts,chain,cases,block,reps,out):
    random=np.random.default_rng(9237000+block)
    values=[]
    matrix=counts.transpose(0,2,1).reshape(-1,len(chain)).astype(np.float64)
    t0=time.time()
    for r in range(reps):
        w=mc_weights(chain,block,random)
        summed=(matrix@w).reshape(len(cases),8)
        p=symmetric_distribution(summed)
        seed_ids=random.integers(0,6,6)
        row=aggregate(pred[:,seed_ids],p,cases).mean(2)
        if not np.isfinite(row).all(): raise AssertionError("Invalid bootstrap replicate; no silent skipping")
        values.append(row)
        if r%100==0:
            print(json.dumps(dict(phase="bootstrap",block=block,replicate=r,reps=reps,elapsed=time.time()-t0)),flush=True)
    values=np.stack(values)
    np.savez_compressed(out/f"bootstrap_block{block}.npz",values=values,groups=np.array(GROUPS),
                        arms=np.array(ARMS),metrics=np.array(METRICS))
    return values


def figures(out,pred,p,cases,per_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.spines.top":False,
        "axes.spines.right":False,"axes.linewidth":1.3,"legend.frameon":False,
        "svg.fonttype":"none","pdf.fonttype":42})
    dest=out/"figures";dest.mkdir()
    colors=["#0F4D92","#767676","#B64342","#767676","#3775BA","#42949E"]
    mm=metrics(pred,p,query_specs(2304))
    for cohort,arms in (("ABC",[0,1,2]),("F012",[3,4,5])):
        fig,axes=plt.subplots(1,3,figsize=(16,4.8))
        for ax,group in zip(axes,["unequal_distance_w48","s10_w48","continuous96"]):
            ix=np.array([c["geometry"]==group for c in cases])
            truth=mm["reference_response"][ix]
            lim=max(.05,float(abs(truth).max())*1.15)
            ax.plot([-lim,lim],[-lim,lim],"--",color="black",lw=1,label="Correct response")
            for a in arms:
                val=mm["response"][a,:,ix] # indexing yields case, seed
                if val.shape!=(ix.sum(),6): raise AssertionError(val.shape)
                ax.scatter(truth,val.mean(1),s=32,color=colors[a],label=ARMS[a],alpha=.8)
            ax.axhline(0,color="#cccccc",lw=.6);ax.axvline(0,color="#cccccc",lw=.6)
            ax.set(title=group,xlabel="MC target probability change",ylabel="Model probability change")
            ax.set_xlim(-lim,lim)
        axes[0].legend(fontsize=10)
        fig.suptitle(f"Fixed background: response amplitude ({cohort}, six-seed means)")
        fig.tight_layout(pad=1.8)
        for ext in ("png","pdf"):fig.savefig(dest/f"geometry_response_{cohort}.{ext}",dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(18,4.8))
    for g,ax in enumerate(axes):
        for m,color,marker in ((3,"#0F4D92","o"),(4,"#B64342","s"),(5,"#767676","^")):
            vals=per_seed[g,:,:,m]
            x=np.arange(6)+(m-4)*.18
            ax.scatter(np.repeat(x,6),vals.reshape(-1),s=9,color=color,alpha=.3)
            ax.scatter(x,vals.mean(1),s=50,color=color,marker=marker,label=METRICS[m])
        ax.set_xticks(range(6),ARMS);ax.set(title=GROUPS[g],ylabel="KL (nats), lower is better")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    fig.suptitle("Two-query joint accuracy; small dots are individual training seeds")
    fig.tight_layout(pad=1.8)
    for ext in ("png","pdf"):fig.savefig(dest/f"joint_accuracy.{ext}",dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    args=parser.parse_args();root=Path(args.root)
    out=root/"analysis";out.mkdir(exist_ok=False)
    cases=json.loads((root/"cases.json").read_text())
    with np.load(root/"reference.npz") as z:
        counts=z["counts"];chain=z["chain"];p=z["reference_p"]
    predictions=[]
    for arm in ARMS:
        seeds=[]
        for seed in SEEDS:
            dest=root/"models"/f"s{seed}_{arm}"
            complete=json.loads((dest/"complete.json").read_text())
            assert complete["checkpoint_unchanged"] and complete["cases"]==112
            assert file_hash(dest/"predictions.npz")==complete["predictions_sha256"]
            with np.load(dest/"predictions.npz") as z: seeds.append(z["probabilities"].astype(float))
        predictions.append(seeds)
    pred=np.array(predictions)
    assert pred.shape==(6,6,112,20,3)
    point=aggregate(pred,p,cases)
    rows=[]
    for g,group in enumerate(GROUPS):
        for a,arm in enumerate(ARMS):
            for s,seed in enumerate(SEEDS):
                rows.append(dict(geometry=group,arm=arm,seed=seed,**dict(zip(METRICS,point[g,a,s].tolist()))))
    with (out/"per_seed_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={group:{arm:dict(zip(METRICS,point[g,a].mean(0).tolist())) for a,arm in enumerate(ARMS)}
             for g,group in enumerate(GROUPS)}
    atomic_json(out/"point_estimates.json",summary)
    uncertainties={}
    for block,reps in ((8,1000),(4,500),(16,500)):
        boot=bootstrap(pred,counts,chain,cases,block,reps,out)
        block_data={}
        for label,a,b in (("A_minus_B",0,1),("A_minus_C",0,2),("F2_minus_F1",5,4)):
            diff=boot[:,:,a]-boot[:,:,b]
            estimates=point[:,a].mean(1)-point[:,b].mean(1)
            block_data[label]={group:{name:dict(estimate=float(estimates[g,m]),
                ci95=np.quantile(diff[:,g,m],[.025,.975]).tolist()) for m,name in enumerate(METRICS)}
                for g,group in enumerate(GROUPS)}
        uncertainties[str(block)]=dict(replicates=reps,paired_contrasts=block_data)
    atomic_json(out/"crossed_uncertainty.json",dict(status="complete",blocks=uncertainties,
        scope="Exploratory; six paired training seeds crossed with eight MC chains and parent time blocks; fixed cases/origins; no familywise correction"))
    mm=metrics(pred,p,query_specs(2304))
    diagnostics={k:float(np.max(mm[k])) for k in ("parallel_identity_residual","sequential_identity_residual","swap_permutation_error")}
    assert max(diagnostics.values())<2e-5
    np.savez_compressed(out/"plot_data.npz",predictions=pred,reference_p=p,per_seed=point,
        arms=np.array(ARMS),seeds=np.array(SEEDS),metrics=np.array(METRICS),groups=np.array(GROUPS))
    figures(out,pred,p,cases,point)
    atomic_json(out/"complete.json",dict(status="complete",diagnostic_max_errors=diagnostics,
        figures=sorted(f.name for f in (out/"figures").iterdir()),models=36,
        warning="Low-dimensional accuracy and incompatibility do not attribute whole-window error fractions"))
    protocol=json.loads((root/"run_protocol.json").read_text())
    final=dict(status="remote_complete_requires_local_backup_and_visual_review",models=36,cases=112,
        forward_inputs=80640,checkpoint_unchanged=True,finished=time.time(),elapsed_seconds=time.time()-protocol["started"],
        main_geometry="unequal_distance_w48",main_point=summary["unequal_distance_w48"],
        main_contrasts=uncertainties["8"]["paired_contrasts"],
        no_new_training=True,limits=["reused MC: exploratory","fixed case bank","C single false-geometry draw",
        "ABC 24k separate from F012 36k","two-query joint not full-generation error attribution"])
    atomic_json(root/"final_summary.json",final)
    manifest=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=file_hash(f))
              for f in sorted(root.rglob("*")) if f.is_file() and f.name not in ("manifest.json","queue.log","status.json","run.lock")]
    atomic_json(root/"manifest.json",dict(files=manifest))
    print(json.dumps(dict(status=final["status"],elapsed_seconds=final["elapsed_seconds"])),flush=True)


if __name__=="__main__": main()
