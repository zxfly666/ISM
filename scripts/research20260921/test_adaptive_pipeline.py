"""Small real-device checks in an isolated directory, never scientific runs."""
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.context_repair_core import self_test,pack_views,tensor_batch,query_logits,update
from ism_diffusion.scale_data import load_parent_split
from run_adaptive_repair_20260922 import restore,save_checkpoint
import adaptive_repair_evaluation as ae
import evaluate_study as ev


def main():
    out=ROOT / "artifacts/adaptive_research_20260922/repair_preflight"
    out.mkdir(parents=True,exist_ok=True)
    test=self_test("cuda")
    torch.set_num_threads(1);torch.set_float32_matmul_precision("high")
    base=ROOT / "artifacts/geometry_alignment_20260921/training/s91001_A/final.pt"
    parent=load_parent_split(ROOT / "data/level1/parents_l1024.npz","train")
    model,ema,opt,p=restore(base,dict(gs.MODEL))
    b=gs.make_batch(parent,2,16,"gap",gs.GAPS,819220,24001,True)
    result=update(model,ema,opt,b,819220,1,8000,"F2",1.)
    save_checkpoint(out / "resume_test.pt",model,ema,opt,p,1,dict(gs.MODEL),
                    {"protocol_hash":"preflight_only"},819220,"F2",0.)
    b2=gs.make_batch(parent,2,16,"gap",gs.GAPS,819220,24002,True)
    result2=update(model,ema,opt,b2,819220,2,8000,"F2",1.)
    second,second_ema,second_opt,_=restore(out / "resume_test.pt",dict(gs.MODEL))
    result3=update(second,second_ema,second_opt,b2,819220,2,8000,"F2",1.)
    error=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),second.parameters()))
    ema_error=max(float((a-b).abs().max()) for a,b in zip(ema.parameters(),second_ema.parameters()))
    assert error<2e-6 and ema_error<2e-6,(error,ema_error)
    t,mask,noisy=gs.corrupt_batch(b["clean"],818,2,.8)
    views=pack_views(b,t,mask,noisy,818,2,qmax=8,rho=.25)
    inputs=tensor_batch(views)
    with torch.inference_mode():
        original=query_logits(model,inputs,"sub")
        padded={k:v.clone() for k,v in inputs.items()}
        padded["sub"]=torch.cat([padded["sub"],torch.full((2,5),3,device="cuda")],1)
        padded["sub_coords"]=torch.cat([padded["sub_coords"],torch.zeros((2,5,2),device="cuda")],1)
        padded["sub_valid"]=torch.cat([padded["sub_valid"],torch.zeros((2,5),device="cuda",dtype=torch.bool)],1)
        pad_error=float((original-query_logits(model,padded,"sub")).abs().max())
    assert pad_error<5e-5,pad_error
    # A spin change at a hidden site cannot alter view membership or coordinates.
    altered=copy.deepcopy(b);altered["clean"][mask]=1-altered["clean"][mask]
    changed=pack_views(altered,t,mask,noisy,818,2,qmax=8,rho=.25)
    for key in ("fullq","subq","sub","sub_coords","sub_valid"):
        assert np.array_equal(views[key],changed[key]),key
    irr=ae.confirmation_batch(parent,8,48,"irregular",[9,10,11],982,1)
    assert np.all(irr["axes"][0][:,-1]==470) and np.all(irr["axes"][1][:,-1]==470)
    assert np.array_equal(irr["coords"]["A"][...,0],np.broadcast_to(irr["axes"][0][:,:,None],(8,48,48)))
    (out / "conditional").mkdir(exist_ok=True)
    records=ae.conditionals(ema,parent,out / "conditional",819220,[ae.GEN_DEFS[-1]],samples=2,deadline=time.time()+120)
    assert len(records)==32,len(records)
    previous=ev.make_batch;ev.make_batch=ae.confirmation_batch
    try:
        generation=ev.generate_and_score(ema,parent,"A",819220,dict(ae.GEN_DEFS[-1],samples=2),out / "generation_smoke",time.time()+120,shard_size=2)
    finally:ev.make_batch=previous
    with np.load(out / "generation_smoke/shard_00000.npz") as z:
        assert np.isin(z["spins"],[-1,1]).all()
    test.update(resume_max_parameter_error=error,resume_max_ema_error=ema_error,pad_logit_error=pad_error,
                hidden_value_selection_independence=True,irregular_physical_span=470,
                conditional_records=len(records),generation_smoke=generation,
                training_loss=result["loss"],step2_loss_difference=abs(result2["loss"]-result3["loss"]),
                config=dict(gs.MODEL),checkpoint_sha256=gs.file_hash(base))
    gs.atomic_json(out / "complete.json",test)
    print(json.dumps(test),flush=True)


if __name__=="__main__":main()
