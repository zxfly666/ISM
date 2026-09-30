"""Independent bounded 2x2 campaign; frozen old sources are never edited."""
import argparse,fcntl,hashlib,json,os,random,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split
from run_adaptive_repair_20260922 import restore
from mask_query_factorial_design import ARMS,SEEDS,GEN_DEFS,COND_DEFS,MC_SEED,make_cases,self_test
from mask_query_factorial_core import views,update
from evaluate_mask_query_factorial_20260924 import log,budget,prepare_reference,evaluate_model
DOC=ROOT/"docs/research_reboot_20260921/MASK_QUERY_FACTORIAL_PROTOCOL_20260924_ZH.md"
def base(seed):return ROOT/f"artifacts/adaptive_research_20260922/repair/training/s{seed}_F0/final.pt"

def save(path,model,ema,opt,p,step,protocol,seed,arm,elapsed,digest):
    data=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        config=dict(model=gs.MODEL,variant="A",seed=seed),step=36000+step,factorial_step=step,arm=arm,
        protocol_hash=protocol["protocol_hash"],base_checkpoint_sha256=protocol["base_hashes"][str(base(seed))],
        elapsed_seconds=elapsed,stream_digests=digest,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_suffix(".tmp");torch.save(data,tmp);os.replace(tmp,path)


def train_block(out,parent,validation,protocol,seed,arm,target):
    dest=out/"training"/f"s{seed}_{arm}";dest.mkdir(parents=True,exist_ok=True)
    last=dest/"last.pt";model,ema,opt,p=restore(last if last.exists() else base(seed),gs.MODEL)
    start=int(p.get("factorial_step",0))+1
    previous=float(p["elapsed_seconds"]) if "factorial_step" in p else 0.
    digest=p.get("stream_digests",dict(main_hash="",aux_clean_hash="",all_views_hash=""))
    if "factorial_step" in p:
        assert p["protocol_hash"]==protocol["protocol_hash"]
        torch.set_rng_state(p["torch_rng"]);torch.cuda.set_rng_state_all(p["cuda_rng"])
        np.random.set_state(p["numpy_rng"]);random.setstate(p["python_rng"])
    began=time.monotonic();model.train();completed=start-1
    try:
        with (dest/"train.jsonl").open("a",encoding="utf-8",buffering=1) as logfile:
            for step in range(start,target+1):
                budget(protocol["deadline"])
                packed=views(parent,seed,step)
                stats=update(model,ema,opt,packed,step,protocol["steps"],arm)
                completed=step
                for key in digest:digest[key]=hashlib.sha256((digest[key]+str(step)+stats[key]).encode()).hexdigest()
                if step==1 or step%20==0:
                    logfile.write(json.dumps(dict(step=step,seed=seed,arm=arm,width=packed["main"]["width"],
                        kind=packed["main"]["kind"],elapsed=previous+time.monotonic()-began,**stats))+"\n")
            budget(protocol["deadline"])
            risk=gs.balanced_risk(gs.evaluate(ema,validation,"A",gs.VAL_DEFS))
            logfile.write(json.dumps(dict(step=target,validation=risk))+"\n")
    except Exception:
        save(dest/"stopped_state.pt",model,ema,opt,p,completed,protocol,seed,arm,
             previous+time.monotonic()-began,digest);raise
    elapsed=previous+time.monotonic()-began
    save(last,model,ema,opt,p,target,protocol,seed,arm,elapsed,digest)
    hp=dest/"validation_history.json";history=json.loads(hp.read_text()) if hp.exists() else []
    history.append(dict(step=target,ce=risk["mean_ce"]));gs.atomic_json(hp,history)
    basece=protocol["base_validation"][str(seed)]["mean_ce"]
    if len(history)>=2 and all(x["ce"]>basece+.05 for x in history[-2:]):
        raise RuntimeError(f"Whole campaign CE safety stop {seed} {arm}; two consecutive >base+0.05")
    if target==protocol["steps"]:
        save(dest/"final.pt",model,ema,opt,p,target,protocol,seed,arm,elapsed,digest)
        gs.atomic_json(dest/"complete.json",dict(status="trained",steps=target,elapsed_seconds=elapsed,
            stream_digests=digest,validation=risk,sha256=gs.file_hash(dest/"final.pt")))
    del model,ema,opt,p;torch.cuda.empty_cache()
    return dict(seed=seed,arm=arm,step=target,validation=risk,elapsed_seconds=elapsed)


def preflight(out):
    out.mkdir(parents=True,exist_ok=False);pre=out/"preflight";pre.mkdir()
    began=time.time()
    gs.atomic_json(pre/"started.json",dict(started=began,deadline=began+12*3600))
    parent=load_parent_split(ROOT/"data/level1/parents_l1024.npz","train")
    val=load_parent_split(ROOT/"data/level1/parents_l1024.npz","val")
    tests=self_test();digest=gs.file_hash(base(91001))
    # Boundary tests on real train-only inputs; no reference MC is opened.
    pack=views(parent,91001,1);again=views(parent,91001,1)
    assert pack["all_views_hash"]==again["all_views_hash"]
    for a,b in (("R00","R01"),("R10","R11")):
        for k in ("noisy","coords","t"):
            assert np.array_equal(pack["choices"][a][k],pack["choices"][b][k])
    torch.use_deterministic_algorithms(True)
    same=dict(pack);same["choices"]={a:pack["choices"]["R10"] for a in ARMS}
    states=[]
    for arm in ARMS:
        m,e,o,p=restore(base(91001),gs.MODEL)
        update(m,e,o,same,1,4000,arm)
        states.append({k:v.cpu().clone() for k,v in m.state_dict().items()})
        del m,e,o,p;torch.cuda.empty_cache()
    identity=max(float((s[k]-states[0][k]).abs().max()) for s in states for k in s)
    assert identity<1e-6
    del states
    scratch=dict(protocol_hash="SCRATCH_ONLY",steps=4,deadline=began+900,
        base_hashes={str(base(91001)):digest},base_validation={"91001":dict(mean_ce=10.)})
    train_block(pre/"resume",parent,val,scratch,91001,"R11",2)
    saved=torch.load(pre/"resume/training/s91001_R11/last.pt",map_location="cpu",weights_only=False)
    rm,re,ro,rp=restore(pre/"resume/training/s91001_R11/last.pt",gs.MODEL)
    assert all(torch.equal(v.cpu(),saved["model"][k]) for k,v in rm.state_dict().items())
    for k,state in ro.state_dict()["state"].items():
        for name,v in state.items():
            other=saved["optimizer"]["state"][k][name]
            assert torch.equal(v.cpu(),other) if torch.is_tensor(v) else v==other
    del saved,rm,re,ro,rp;torch.cuda.empty_cache()
    train_block(pre/"resume",parent,val,scratch,91001,"R11",4)
    train_block(pre/"direct",parent,val,scratch,91001,"R11",4)
    pp=torch.load(pre/"resume/training/s91001_R11/final.pt",map_location="cpu",weights_only=False)
    qq=torch.load(pre/"direct/training/s91001_R11/final.pt",map_location="cpu",weights_only=False)
    err=max(float((pp["model"][k]-qq["model"][k]).abs().max()) for k in pp["model"])
    assert err==0 and pp["stream_digests"]==qq["stream_digests"]
    del pp,qq;torch.use_deterministic_algorithms(False);torch.cuda.empty_cache()
    tests.update(identical_view_arm_error=identity,resume_weight_error=err,paired_views=True,
                 exact_optimizer_roundtrip=True,heldout_used=False)
    from test_mask_query_factorial_20260924 import test_scores_and_banks
    tests.update(test_scores_and_banks())
    bench={}
    for arm in ARMS:
        m,e,o,p=restore(base(91001),gs.MODEL);m.train()
        records=[];torch.cuda.reset_peak_memory_stats()
        for step in range(1,25):
            torch.cuda.synchronize();t=time.perf_counter()
            packed=views(parent,91001,step);stats=update(m,e,o,packed,step,4000,arm)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            assert stats["tokens"]==18432
            if step>6:records.append(seconds)
        bench[arm]=dict(seconds=float(np.mean(records)),peak_gib=torch.cuda.max_memory_allocated()/2**30)
        print(json.dumps(dict(stage="preflight_train",arm=arm,**bench[arm])),flush=True)
        del m,e,o,p;torch.cuda.empty_cache()
    m,e,o,p=restore(base(91001),gs.MODEL);e.eval()
    from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
    sampler=CoordinateAbsorbingDiffusion();cc=gs.coordinate_arrays(16,128,"continuous",(1,),2026092417,0)[0]["A"]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        x=sampler.sample(e,torch.as_tensor(cc,device="cuda"),
             torch.ones((16,128,128),device="cuda",dtype=torch.bool),steps=4,
             generator=torch.Generator(device="cuda").manual_seed(2026092417))
    assert ((x==0)|(x==1)).all();del x
    t=time.perf_counter();gs.evaluate(e,val,"A",gs.VAL_DEFS);valcost=time.perf_counter()-t
    del m,e,o,p;torch.cuda.empty_cache()
    # Actual preceding full 256-step timings, same hardware/model/sampler.
    oldroot=ROOT/"artifacts/sparse_conditioning_20260923"
    genrate={}
    for d in GEN_DEFS:
        rates=[]
        for f in (oldroot/"evaluation").glob(f"s*/generation/{d['name']}/complete.json"):
            result=json.loads(f.read_text());rates.append(result["sampling_seconds"]/result["definition"]["samples"])
        assert len(rates)==18
        genrate[d["name"]]=max(rates)
    train=6*4000*sum(v["seconds"] for v in bench.values())*1.2
    gen=24*sum(d["samples"]*genrate[d["name"]] for d in GEN_DEFS)*1.2
    # 240 sec/model conservatively covers preceding full two-clock cases+conditional banks.
    conditional=24*240*1.25
    expected=(time.time()-began)+max(train,3000)+gen+conditional+(96+6)*valcost+900
    assert gs.file_hash(base(91001))==digest
    answer=dict(status="passed" if expected<11.5*3600 else "budget_gate_failed",tests=tests,training=bench,
        generation_seconds_per_image=genrate,forecast_hours=expected/3600,val_seconds=valcost,
        components_seconds=dict(training=train,generation=gen,conditional=conditional,reference_and_statistics=900),
        base_unchanged=True,started=began,deadline=began+12*3600,steps=4000)
    gs.atomic_json(pre/"preflight.json",answer);print(json.dumps(answer),flush=True)


def freeze(out,validation):
    pre=json.loads((out/"preflight/preflight.json").read_text())
    assert pre["status"]=="passed" and pre["tests"]["status"]=="passed"
    baseline={}
    for seed in SEEDS:
        m,e,o,p=restore(base(seed),gs.MODEL);assert p["step"]==36000
        baseline[str(seed)]=gs.balanced_risk(gs.evaluate(e,validation,"A",gs.VAL_DEFS))
        del m,e,o,p;torch.cuda.empty_cache()
    sources=sorted(list((ROOT/"ism_diffusion").glob("*.py"))+list(Path(__file__).parent.glob("*.py")))
    snapshot=out/"source";snapshot.mkdir(exist_ok=False);hashes={}
    for p in sources:
        rel=p.relative_to(ROOT);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);hashes[str(rel)]=gs.file_hash(p)
    shutil.copy2(DOC,out/"protocol.md")
    proto=dict(version="mask_query_factorial_v1",started=pre["started"],deadline=pre["deadline"],
        training_started=time.time(),steps=4000,model=gs.MODEL,seeds=SEEDS,arms=ARMS,
        generation=GEN_DEFS,conditional=COND_DEFS,source_hashes=hashes,
        base_hashes={str(base(s)):gs.file_hash(base(s)) for s in SEEDS},base_validation=baseline,
        train_source_sha256=gs.file_hash(ROOT/"data/level1/parents_l1024.npz"),
        protocol_sha256=gs.file_hash(DOC),mc_seed=MC_SEED,forecast_hours=pre["forecast_hours"],
        microbatch={24:8,48:4,96:1},generation_shard=16,aux_loss="window_mean_64_hard_label_slots_no_t_weight",
        training_stream_seed_offset=10000,primary_family_size=4,primary_ci_level=.9875)
    proto["protocol_hash"]=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest()
    gs.atomic_json(out/"run_protocol.json",proto);gs.atomic_json(out/"cases.json",make_cases())
    return proto


def run(out):
    lock=(out/"run.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"run_protocol.json").exists() or (out/"training").exists():raise RuntimeError("Existing campaign; no restart")
    parent=load_parent_split(ROOT/"data/level1/parents_l1024.npz","train")
    val=load_parent_split(ROOT/"data/level1/parents_l1024.npz","val")
    proto=freeze(out,val);budget(proto["deadline"])
    log(out,"protocol_frozen",steps=4000,forecast_hours=proto["forecast_hours"],deadline=proto["deadline"])
    mc=None
    try:
        ref=out/"reference";ref.mkdir(exist_ok=False)
        with (ref/"queue.log").open("w") as f:
            mc=subprocess.Popen([sys.executable,"-u",str(Path(__file__).parent/"prepare_mask_query_reference_20260924.py"),
                 "--out",str(ref)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                 env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"))
        gs.atomic_json(out/"processes.json",dict(main_pid=os.getpid(),reference_pid=mc.pid,started=proto["started"]))
        for target in range(1000,4001,1000):
            for seed in SEEDS:
                for arm in ARMS:
                    budget(proto["deadline"])
                    if mc.poll() not in (None,0):raise RuntimeError("MC failed")
                    if (ref/"complete.json").exists() and json.loads((ref/"complete.json").read_text())["status"]!="passed":
                        raise RuntimeError("MC QA failed")
                    log(out,"training",seed=seed,arm=arm,target_step=target)
                    result=train_block(out,parent,val,proto,seed,arm,target)
                    print(json.dumps(dict(stage="training_block_complete",**result)),flush=True)
        del parent,val
        log(out,"waiting_reference")
        while mc.poll() is None:budget(proto["deadline"]);time.sleep(10)
        assert mc.returncode==0
        qa=json.loads((ref/"complete.json").read_text());assert qa["status"]=="passed"
        assert gs.file_hash(ref/"fresh_l1024.npz")==qa["sha256"]
        fresh=load_parent_split(ref/"fresh_l1024.npz","test_target")
        prepare_reference(fresh,out,make_cases(),proto["deadline"])
        for seed in SEEDS:
            for arm in ARMS:
                budget(proto["deadline"]);log(out,"evaluation",seed=seed,arm=arm)
                evaluate_model(out/"training"/f"s{seed}_{arm}"/"final.pt",fresh,out/"evaluation"/f"s{seed}_{arm}",out,seed,arm,proto["deadline"])
        del fresh;log(out,"statistical_analysis")
        from finalize_mask_query_factorial_20260924 import main as finalize
        summary=finalize(out,proto["deadline"]);log(out,"remote_complete_requires_backup_and_visual_review",elapsed_hours=summary["elapsed_hours"])
    finally:
        if mc is not None and mc.poll() is None:os.killpg(mc.pid,signal.SIGTERM);mc.wait(timeout=30)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--mode",choices=["preflight","run"],required=True)
    a=p.parse_args();out=Path(a.out).resolve()
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    try:(preflight if a.mode=="preflight" else run)(out)
    except Exception as e:
        path=out/("preflight/failure.json" if a.mode=="preflight" else "failure.json")
        gs.atomic_json(path,dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),automatic_restart=False))
        raise
