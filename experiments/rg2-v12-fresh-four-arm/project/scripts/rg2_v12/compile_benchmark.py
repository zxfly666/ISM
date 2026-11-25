"""Optional training compiler benchmark; isolated from all scientific outputs."""
import os
from pathlib import Path
ROOT=Path(__file__).absolute().parents[2]
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'runs/compiler_cache/inductor')
os.environ['TRITON_CACHE_DIR']=str(ROOT/'runs/compiler_cache/triton')
import copy
import json
import time
import numpy as np
import torch
import phase_b as b

def main():
    out=ROOT/'runs/compile_benchmark';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    fields,_,_=b.parent('train');eager=b.initialize(112501);fast=b.initialize(112501)
    # Nonzero output weights make the numerical comparison informative.
    entry=next(e for e in json.loads((ROOT/'inputs/manifest.json').read_text())['models'] if e['key']=='s92601_fine_plus_rg_final')
    state=torch.load(ROOT/entry['file'],weights_only=True,map_location='cpu')
    eager['model'].load_state_dict(state);fast['model'].load_state_dict(state)
    fast['model'].forward=torch.compile(fast['model'].forward,mode='reduce-overhead',dynamic=False)
    checks=[];timings=[]
    b.a.write_json(out/'thresholds.json',dict(max_absolute_loss_error=.001,max_relative_gradient_error=.05,
                   purpose='implementation-equivalence diagnostic; these are not scientific acceptance thresholds'))
    for cycle in range(1,5):
        for role in ('common','aux'):
            z=b.batch(fields,112501,cycle,role,'M')
            reference=None
            for label,s in [('eager',eager),('compiled',fast)]:
                s['model'].zero_grad(set_to_none=True)
                begin=time.monotonic();value=b.loss(s['model'],z);value.backward();torch.cuda.synchronize()
                grad=torch.cat([p.grad.flatten() for p in s['model'].parameters() if p.grad is not None])
                if reference is None:reference=(float(value),grad.clone())
                else:
                    diff=abs(float(value)-reference[0]);rel=float(torch.linalg.vector_norm(grad-reference[1])/torch.linalg.vector_norm(reference[1]))
                    checks.append(dict(width=z['width'],role=role,loss_error=diff,gradient_relative_error=rel,compile_seconds=time.monotonic()-begin))
                    b.a.write_json(out/'progress.json',checks)
                    if diff>.001 or rel>.05:raise RuntimeError('Numerical equivalence threshold exceeded')
                # Compare identical forward/backward workloads without optimizers.
                for rep in range(6):
                    s['model'].zero_grad(set_to_none=True)
                    torch.cuda.synchronize();started=time.monotonic();v=b.loss(s['model'],z);v.backward();torch.cuda.synchronize()
                    if rep>=2:timings.append(dict(mode=label,width=z['width'],role=role,seconds=time.monotonic()-started))
            print('COMPILER CHECK',cycle,role,checks[-1],flush=True)
    b.a.write_json(out/'result.json',dict(status='passed',checks=checks,timings=timings,
                caveat='A0 retention ran concurrently; absolute times are preliminary, paired speed ratios are more informative'))

if __name__=='__main__':main()
