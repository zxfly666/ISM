"""Read-only trajectory instrumentation of the unchanged S0 sampler.

On-policy contexts have no MC label. Only teacher-MC counterparts have CE;
on-policy numbers are sensitivity/entropy, never presented as true error.
"""
from pathlib import Path
import time
import numpy as np
import torch
from ism_diffusion.context_repair_core import pack_views,tensor_batch,query_logits
from ism_diffusion.scale_evaluation import load_scale_model
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion import geometry_study as gs
from adaptive_repair_evaluation import confirmation_batch
import evaluate_study as ev


class Recorder(torch.nn.Module):
    def __init__(self,model,batch,seed,out):
        super().__init__();self.inner=model;self.batch=batch;self.seed=seed;self.out=out
        self.calls=0;self.rows=[]

    def forward(self,x,t,c,valid=None):
        logits=self.inner(x,t,c,valid);call=self.calls;self.calls+=1
        if call not in (1,32,64,128,192,240):return logits
        noisy=x.cpu().numpy();mask=noisy==2;tt=t.cpu().numpy()
        # Instrumentation is spin-blind and does not consume the sampler RNG.
        for mode in ("on_policy","teacher_mc"):
            xx=noisy if mode=="on_policy" else np.where(mask,2,self.batch["clean"])
            views=pack_views(self.batch,tt,mask,xx,self.seed,call,qmax=64,rho=1/3)
            data=tensor_batch(views)
            lp=query_logits(self.inner,data,"full");lq=query_logits(self.inner,data,"sub")
            p=lp.softmax(-1).cpu().numpy();q=lq.softmax(-1).cpu().numpy()
            mid=.5*(p+q);js=.5*(p*(np.log(np.clip(p,1e-8,1))-np.log(np.clip(mid,1e-8,1)))+
                                   q*(np.log(np.clip(q,1e-8,1))-np.log(np.clip(mid,1e-8,1)))).sum(-1)
            for j in range(len(tt)):
                good=views["qvalid"][j]
                if not good.any():continue
                pp=np.clip(p[j,good,1],1e-7,1-1e-7);yy=views["labels"][j,good]
                row=dict(step=call,t=float(tt[j]),mode=mode,sample=j,n_query=int(good.sum()),
                    visible=int(views["original_visible"][j]),parent=int(self.batch["parent"][j]),
                    chain=int(self.batch["chain"][j]),js=float(js[j,good].mean()),
                    abs_probability_change=float(np.abs(p[j,good,1]-q[j,good,1]).mean()),
                    entropy=float((-pp*np.log(pp)-(1-pp)*np.log1p(-pp)).mean()),
                    teacher_ce=float(-(yy*np.log(pp)+(1-yy)*np.log1p(-pp)).mean()) if mode=="teacher_mc" else float("nan"))
                self.rows.append(row)
            ev.atomic_npz(self.out / f"step{call:03d}_{mode}.npz",tokens=xx,t=tt,full_p=p,sub_p=q,
                          query=views["fullq"],qvalid=views["qvalid"],mc_labels=views["labels"],
                          label_is_truth=np.array(mode=="teacher_mc"),coordinates=c.cpu().numpy(),
                          parent=self.batch["parent"],chain=self.batch["chain"])
        return logits


def run(model_path,parent,out,seed,definition,deadline,samples=8):
    out.mkdir(parents=True,exist_ok=True)
    if (out / "complete.json").exists():return
    model,_=load_scale_model(model_path,torch.device("cuda"));diffusion=CoordinateAbsorbingDiffusion()
    rows=[];began=time.time()
    for start in range(0,samples,4):
        if time.time()>=deadline:raise TimeoutError("trajectory deadline")
        folder=out / f"batch{start:03d}";folder.mkdir(exist_ok=True)
        if (folder / "complete.json").exists():continue
        b=confirmation_batch(parent,min(4,samples-start),definition["width"],definition["kind"],
                             definition["gaps"],2026092401,seed+start)
        c=torch.tensor(b["coords"]["A"],device="cuda");valid=torch.ones(b["clean"].shape,device="cuda",dtype=torch.bool)
        recorder=Recorder(model,b,2026092402+seed+start,folder)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            tokens=diffusion.sample(recorder,c,valid,steps=256,generator=torch.Generator(device="cuda").manual_seed(seed+start+2026092403))
        ev.atomic_npz(folder / "final.npz",spins=(tokens.cpu().numpy()*2-1).astype(np.int8),mc=(2*b["clean"]-1).astype(np.int8))
        ev.write_csv(folder / "rows.csv",recorder.rows)
        gs.atomic_json(folder / "complete.json",dict(status="complete",rows=len(recorder.rows)))
    gs.atomic_json(out / "complete.json",dict(status="complete",samples=samples,elapsed=time.time()-began,
        checkpoint_sha256=gs.file_hash(model_path),definition=definition,
        limitation="adaptive exploratory follow-up; teacher CE is not on-policy ground truth; no sampler change"))
    del model;torch.cuda.empty_cache()


def smoke_test(model,parent,out):
    """The instrumented and original sampler must give identical final tokens."""
    out.mkdir(parents=True,exist_ok=True)
    b=confirmation_batch(parent,1,16,"continuous",[1],2026092401,1)
    c=torch.tensor(b["coords"]["A"],device="cuda");valid=torch.ones((1,16,16),device="cuda",dtype=torch.bool)
    recorder=Recorder(model,b,400,out);sampler=CoordinateAbsorbingDiffusion()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        a=sampler.sample(model,c,valid,steps=8,generator=torch.Generator(device="cuda").manual_seed(401))
        bb=sampler.sample(recorder,c,valid,steps=8,generator=torch.Generator(device="cuda").manual_seed(401))
    assert torch.equal(a,bb),"instrumentation changed sampling"
    assert recorder.rows,"missing instrumentation"
    return dict(status="passed",sampler_bitwise_identical=True,rows=len(recorder.rows))
