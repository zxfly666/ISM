"""CPU evidence audits, exclusive archives, verified union coverage and monitoring."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
import numpy as np
import endpoint_common as c
import endpoint_data as d


def archive(root, export, name, paths):
    root, export=Path(root).resolve(),Path(export).resolve(); export.mkdir(parents=True,exist_ok=True)
    tarpath=export/(name+".tar.gz"); manifestpath=export/(name+".manifest.json")
    if tarpath.exists() or manifestpath.exists(): raise FileExistsError(name)
    paths=sorted(set(Path(p).resolve() for p in paths))
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p)) for p in paths]
    with tarfile.open(tarpath,"x:gz",compresslevel=3,dereference=True) as tar:
        for p in paths: tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    # Sources must not change while being packed. Growing stdout is explicitly excluded.
    for p,row in zip(paths,files): assert p.stat().st_size==row["bytes"] and c.sha(p)==row["sha256"]
    result=dict(archive=tarpath.name,archive_bytes=tarpath.stat().st_size,archive_sha256=c.sha(tarpath),
                members=len(files),files=files,created=time.time())
    c.write(manifestpath,result); return result


def verify(archive_path,manifest_path,receipt):
    path=Path(archive_path); manifest=c.read(manifest_path)
    archive_bytes=manifest.get("archive_bytes",manifest.get("bytes"))
    archive_sha=manifest.get("archive_sha256",manifest.get("sha256"))
    assert path.stat().st_size==archive_bytes and c.sha(path)==archive_sha
    expected={r["path"]:r for r in manifest["files"]}; seen=set()
    with tarfile.open(path,"r:gz") as tar:
        for entry in tar:
            assert entry.isfile() and entry.name in expected and entry.name not in seen
            assert not entry.name.startswith("/") and ".." not in Path(entry.name).parts
            h=hashlib.sha256()
            with tar.extractfile(entry) as f:
                for block in iter(lambda:f.read(4*1024**2),b""): h.update(block)
            assert h.hexdigest()==expected[entry.name]["sha256"] and entry.size==expected[entry.name]["bytes"]
            seen.add(entry.name)
    assert seen==set(expected)
    result=dict(status="passed",time=time.time(),archive=path.name,archive_bytes=path.stat().st_size,
                archive_path=str(path.resolve()),manifest_path=str(Path(manifest_path).resolve()),
                archive_sha256=archive_sha,manifest_sha256=c.sha(manifest_path),members=len(seen))
    c.write(receipt,result); return result


def union(manifest_path,backup_dirs,receipt):
    science=c.read(manifest_path); covered={}; packages=[]
    for folder in map(Path,backup_dirs):
        for rec in sorted(folder.glob("*.verification.json")):
            r=c.read(rec)
            if r.get("status")!="passed":continue
            name=r["archive"][:-7]
            mp=folder/(name+".manifest.json"); ap=folder/r["archive"]
            if not mp.exists():mp=Path(r["manifest_path"])
            if not ap.exists():ap=Path(r["archive_path"])
            assert c.sha(mp)==r["manifest_sha256"] and c.sha(ap)==r["archive_sha256"]
            mm=c.read(mp); packages.append(str(ap))
            for row in mm["files"]: covered[(row["path"],row["sha256"],row["bytes"])]=str(ap)
    missing=[r for r in science["files"] if (r["path"],r["sha256"],r["bytes"]) not in covered]
    result=dict(status="passed" if not missing else "failed",time=time.time(),science_paths=len(science["files"]),
                covered=len(science["files"])-len(missing),missing=missing,packages=packages,
                manifest_sha256=c.sha(manifest_path),same_disk_not_offsite=True)
    c.write(receipt,result); assert not missing; return result


def record_paths(paths):
    return [dict(path=p.relative_to(c.ROOT).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(set(paths))]


def package_science():
    proto=c.read(c.OUT/"run_protocol.json"); c.check_sources(proto["sources"])
    # Only changing console/status administration is omitted pending an end-of-process closure delta.
    omitted=("last.pt","formal.stdout","reference.stdout","status.json","run.jsonl")
    paths=[p for p in c.OUT.rglob("*") if p.is_file() and p.name not in omitted
           and not any(x in p.parts for x in ("__pycache__","preflight")) and not p.name.endswith(".tmp")]
    inherited=[c.ROOT/p for p in proto["sources"]]+[c.base_path(s) for s in c.SEEDS]
    science=dict(files=record_paths(paths+inherited),time=time.time(),
                 exclusions=[str(p.relative_to(c.ROOT)) for p in c.OUT.rglob("last.pt")],
                 administrative_delta_required=[x for x in omitted if x!="last.pt"],
                 inherited_bases_reused_verified_archives_not_redownloaded=True)
    c.write(c.OUT/"science_manifest.json",science)
    return archive(c.ROOT,c.EXPORT,"complete_science_v1",paths+[c.OUT/"science_manifest.json"])


def closure():
    """Post-exit administration only: no GPU, resampling, training or statistical recomputation."""
    assert (c.OUT/"remote_export_complete.json").exists()
    begun=c.read(c.OUT/"formal_started.json")
    process=Path(f"/proc/{begun['pid']}/cmdline")
    if process.exists() and "run_endpoint.py" in process.read_bytes().decode(errors="replace"):
        raise RuntimeError("Formal driver still alive; cannot freeze final console records")
    science=c.read(c.OUT/"science_manifest.json")
    c.check_sources(c.read(c.OUT/"run_protocol.json")["sources"],enforce_deadline=False)
    for row in science["files"]:
        path=c.ROOT/row["path"]
        assert path.stat().st_size==row["bytes"] and c.sha(path)==row["sha256"]
    late=[c.OUT/name for name in ["formal.stdout","reference.stdout","status.json","run.jsonl","remote_export_complete.json","science_manifest.json"] if (c.OUT/name).exists()]
    combined={r["path"]:r for r in science["files"]}
    for row in record_paths(late):combined[row["path"]]=row
    destination=c.EXPORT/"closure_manifest_v1.json"
    c.write(destination,dict(files=list(combined.values()),time=time.time(),status="remote_bytes_closed_needs_local_union_and_visual_review",
                             inherited_science_manifest_sha256=c.sha(c.OUT/"science_manifest.json"),no_scientific_recomputation=True))
    return archive(c.ROOT,c.EXPORT,"administrative_closure_v1",late+[destination])


def audit_training(reconstruct=True):
    parent=d.load_parent(c.DATA,"train") if reconstruct else None
    totals=0; finals={}; snapshots={}; initial={}
    import torch
    from endpoint_training import check_optimizer, base_hashes
    def finite(value):
        if torch.is_tensor(value): assert bool(torch.isfinite(value).all())
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(tuple,list)):
            for v in value:finite(v)
    for seed in c.SEEDS:
        files={a:(c.OUT/"training"/f"s{seed}_{a}"/"log.jsonl").open(encoding="utf-8") for a in c.ARMS}
        digests={a:{k:"" for k in ["physical","native","supervised"]} for a in c.ARMS}
        try:
            for step in range(1,8001):
                if step%128==1:c.check()
                rows=[]
                for arm, handle in files.items():
                    row=json.loads(next(handle)); assert (row["seed"],row["arm"],row["step"])==(seed,arm,step)
                    assert np.isfinite([row["loss"],row["grad_norm"],row["lr"],*row["model_t"]]).all()
                    assert row["input_tokens"]==18432
                    if reconstruct:
                        batch=d.training_batch(parent,seed,step,arm)
                        for key in ["physical_hash","native_hash","supervision_hash"]:assert row[key]==batch[key]
                        assert row["M"]==batch["mask_counts"].tolist() and row["model_t"]==batch["t"].tolist()
                    for field,key in [("physical","physical_hash"),("native","native_hash"),("supervised","supervision_hash")]:
                        digests[arm][field]=c.increment(digests[arm][field],row[key])
                    rows.append(row);totals+=1
                assert len({r["physical_hash"] for r in rows})==1
                if rows[2]["repeat"]!=2:assert rows[1]["native_hash"]==rows[2]["native_hash"]
                if rows[0]["sparse"]:assert len({r["native_hash"] for r in rows})==1
            assert all(h.read()=="" for h in files.values())
        finally:
            for h in files.values():h.close()
        for arm in c.ARMS:
            folder=c.OUT/"training"/f"s{seed}_{arm}"
            init=c.read(folder/"initial.json");initial[seed,arm]=init
            path=folder/"final.pt"; ck=torch.load(path,map_location="cpu",weights_only=False)
            assert ck["step"]==8000 and ck["global_step"]==20000 and ck["seed"]==seed and ck["arm"]==arm
            assert ck["protocol_hash"]==c.read(c.OUT/"run_protocol.json")["protocol_hash"]
            assert ck["metadata"]["base_sha256"]==base_hashes()[seed]
            for field in digests[arm]: assert ck["metadata"][field+"_digest"]==digests[arm][field]
            for key in ["model","ema","optimizer"]:finite(ck[key])
            assert ck["optimizer"]["state"] and all(int(v["step"])==20000 for v in ck["optimizer"]["state"].values())
            assert all(k in ck for k in ["torch_rng","cuda_rng","numpy_rng","python_rng"])
            finals[path.relative_to(c.ROOT).as_posix()]=c.sha(path)
            assert c.read(folder/"complete.json")["final_sha256"]==c.sha(path)
            for step in [4000,6000]:
                path=folder/f"ema_{step}.pt";state=torch.load(path,map_location="cpu",weights_only=False)
                assert (state["step"],state["seed"],state["arm"])==(step,seed,arm);finite(state["ema"])
                snapshots[path.relative_to(c.ROOT).as_posix()]=c.sha(path)
        for key in ["base_sha256","initial_raw_hash","initial_ema_hash","base_initial_hash"]:
            assert len({initial[seed,a][key] for a in c.ARMS})==1
    assert totals==144000
    return dict(status="passed",updates=totals,input_reconstruction=reconstruct,finite=True,
                paired_lineages=6,finals=finals,ema_snapshots=snapshots,time=time.time())


def audit_predictions():
    from endpoint_training import metrics
    counts=0; inputs=0; banks={}; sources={}
    tasks=[]
    for seed in c.SEEDS:
        for arm in c.ARMS:
            cell=c.OUT/"evaluation"/f"s{seed}_{arm}"; train=c.OUT/"training"/f"s{seed}_{arm}"
            for p in sorted((cell/"conditional").glob("*.npz")):tasks.append((p,c.OUT/"banks"/p.name,train/"final.pt"))
            for p in sorted((cell/"learning").glob("*.npz")):
                step=int(p.name.split("_",1)[0][4:]);bankname=p.name.split("_",1)[1]
                tasks.append((p,c.OUT/"validation_banks"/bankname,train/("final.pt" if step==8000 else f"ema_{step}.pt")))
        for p in sorted((c.OUT/"phase0"/f"s{seed}"/"conditional").glob("*.npz")):
            tasks.append((p,c.OUT/"banks"/p.name,c.base_path(seed)))
    assert len(tasks)==546
    for path, bankpath, source in tasks:
        c.check()
        if bankpath not in banks:banks[bankpath]=c.load(bankpath)
        b=banks[bankpath];d.validate_bank(b);z=c.load(path)
        for key in ["target","parent","chain"]:assert np.array_equal(z[key],b[key])
        assert str(z["common_native_hash"])==d.bank_hash(b) and str(z["bank_sha256"])==c.sha(bankpath)
        if source not in sources:sources[source]=c.sha(source)
        assert str(z["source_sha256"])==sources[source]
        expected=metrics(z["probability"],z["target"])
        for key,value in expected.items():
            if key=="kl" and not str(b["target_type"]).startswith("exact"): assert key not in z;continue
            assert np.allclose(value,z[key],atol=2e-12,rtol=0)
        counts+=1;inputs+=len(z["parent"])
    assert inputs==82176
    return dict(status="passed",prediction_files=counts,inputs=inputs,banks=len(banks),sources=len(sources),time=time.time())


def audit_reference():
    parent=d.load_parent(c.OUT/"reference/fresh_l1024.npz","test_target")
    assert len(parent.spins)==2048 and np.array_equal(np.bincount(parent.chain_ids),np.full(16,128))
    assert c.read(c.OUT/"reference/qa.json")["status"]=="passed"
    crop=c.load(c.OUT/"reference/crop96.npz");assert np.array_equal(crop["chain"],parent.chain_ids)
    pieces={}
    for start in range(0,2048,16):
        c.check();ids=np.arange(start,start+16)
        origin=np.stack([c.rng(c.CFG["role_seeds"]["banks"],int(i),"reference_crop96").integers(1024,size=2) for i in ids])
        axes=np.broadcast_to(np.arange(96),(16,2,96)).copy()
        fields=(2*d.sample(parent,ids,axes,origin)-1).astype(np.int8)
        assert np.array_equal(crop["spins"][ids],fields) and np.array_equal(crop["origin"][ids],origin)
        values=d.multiscale_stats(fields);values["w48"]=d.physical_stats(fields[:,24:72,24:72])
        for key,val in values.items():pieces.setdefault(key,[]).append(val)
    for view,parts in pieces.items():
        actual=c.load(c.OUT/"reference"/f"stats_{view}.npz")
        for key,val in d.concatenate(parts).items():assert np.array_equal(val,actual[key])
    from ism_diffusion.ising import energy_density,magnetization
    from ism_diffusion.diagnostics import split_rhat,integrated_autocorrelation_time
    traces=c.load(c.OUT/"reference/chain_traces.npz");m=magnetization(parent.spins)
    for name,values in [("energy",energy_density(parent.spins)),("m",m),("m2",m*m),("abs_m",np.abs(m))]:
        assert np.array_equal(traces[name],values)
    full=d.concatenate(pieces["full"])
    assert np.array_equal(traces["G25_48"],(full["pair_sum"][:,25:49]/full["pair_count"][:,25:49]).mean(1))
    for name in ["energy","abs_m","m2","G25_48"]:
        series=[traces[name][parent.chain_ids==j] for j in range(16)]
        assert split_rhat(series)<=1.1 and all(integrated_autocorrelation_time(x)["ess"]>=16 for x in series)
    # Rebuild all bank identity/geometry/targets from their frozen parent data.
    ids=d.select_parents(parent,256);checked=0
    for width in [48,96]:
        for masks in [1,2,8,32]:
            b=d.local_bank(parent,ids,width,masks,"exact_test")
            assert d.bank_hash(b)==d.bank_hash(c.load(c.OUT/"banks"/f"exact_w{width}_m{masks}.npz"));checked+=1
        b=d.stress_bank(parent,width)
        assert d.bank_hash(b)==d.bank_hash(c.load(c.OUT/"banks"/f"stress_w{width}.npz"));checked+=1
    for name,kind,k in [("C48_K115","continuous",115),("C48_K1152","continuous",1152),("H48_K512","held_gap",512)]:
        b=d.retention_bank(parent,ids,kind,k)
        assert d.bank_hash(b)==d.bank_hash(c.load(c.OUT/"banks"/(name+".npz")));checked+=1
    return dict(status="passed",parents=2048,chains=16,crop_and_physics_recomputed=True,formal_banks_rebuilt=checked,time=time.time())


def audit_generation():
    import endpoint_sampling as sam
    total=0; phase0=0; folders=0; rng_pairing={}; checked_repair=0
    roots=list(c.OUT.glob("evaluation/*/generation/w*"))+list(c.OUT.glob("phase0/*/floor*"))
    assert len(roots)==84
    for folder in roots:
        c.check(); complete=c.read(folder/"complete.json"); arrays={};ids=[]
        frozen="phase0" in folder.parts
        for path in sorted(folder.glob("shard_*.npz")):
            z=c.load(path); sam.audit_repair(z)
            seed=int(z["seed"]);width=int(z["width"]);images=z["image_ids"]
            assert int(z["network_calls"])==256 and len(images)==16
            assert str(z["source_sha256"])==complete["source_sha256"]
            source=c.base_path(seed) if frozen else c.OUT/"training"/folder.parent.parent.name/"final.pt"
            assert c.sha(source)==str(z["source_sha256"])
            assert np.array_equal(z["reveal_rng_seeds"],sam.image_seeds(seed,width,images,"reveal"))
            counts=z["M_before"]; revealed=z["newly_revealed"]
            assert np.all(counts[:,0]==width**2) and np.all(revealed>=0) and np.all(revealed<=counts)
            assert np.array_equal(counts[:,1:],counts[:,:-1]-revealed[:,:-1])
            assert np.all(counts[:,-1]-revealed[:,-1]==0)
            if "repair_positions" in z:
                p,u,seeds=sam.repair_design(seed,width,images)
                assert np.array_equal(z["repair_positions"],p.transpose(1,0,2))
                assert np.array_equal(z["repair_uniforms"],u.transpose(1,0,2))
                assert np.array_equal(z["repair_rng_seeds"],seeds);checked_repair+=len(images)
            for i,image_id in enumerate(images):
                key=(seed,width,int(image_id));r=int(z["reveal_rng_seeds"][i])
                if key in rng_pairing:assert rng_pairing[key]==r
                rng_pairing[key]=r
            for typ,key in [("final","spins"),("prefix","prefix_spins"),("oracle","oracle_spins")]:
                if key not in z:continue
                for view,values in d.multiscale_stats(z[key]).items():arrays.setdefault((typ,view),[]).append(values)
            ids.extend(images.tolist());total+=0 if frozen else len(images);phase0+=len(images) if frozen else 0
        assert ids==list(range(1000000 if frozen else 0,(1000000 if frozen else 0)+complete["images"]))
        for (typ,view),parts in arrays.items():
            expected=d.concatenate(parts);actual=c.load(folder/f"stats_{typ}_{view}.npz")
            for key,value in expected.items():assert np.array_equal(value,actual[key]),(folder,key)
        folders+=1
    assert total==6912 and phase0==384 and checked_repair==3456
    return dict(status="passed",formal_images=total,frozen_phase0_images=phase0,
                correction_trajectories_replayed=checked_repair,oracle_trajectories_replayed=checked_repair,
                statistics_recomputed_from_spins=True,paired_rng_keys=len(rng_pairing),folders=folders,time=time.time())


def audit_all():
    c.check_sources(c.read(c.OUT/"run_protocol.json")["sources"])
    results={}
    for name,fun in [("training",audit_training),("reference",audit_reference),("predictions",audit_predictions),("generation",audit_generation)]:
        start=time.time();results[name]=fun();results[name]["elapsed_seconds"]=time.time()-start
        c.write(c.OUT/"audit"/(name+".json"),results[name]);c.log("audit_completed",part=name)
    assert len(list((c.OUT/"analysis").glob("*.png")))==6 and len(list((c.OUT/"analysis").glob("*.pdf")))==6
    c.check_sources(c.read(c.OUT/"run_protocol.json")["sources"])
    c.write(c.OUT/"audit/complete.json",dict(status="passed",results=results,time=time.time()))
    return results


def monitor():
    result=dict(time=time.time(),resources=c.resources(),status=None,budget=None,failures={},training_complete=0,
                evaluation_complete=0,final_summary=False,source_hashes="not_frozen")
    for key,name in [("status","status.json"),("budget","budget.json"),("forecast","preflight/summary.json")]:
        if (c.OUT/name).exists():result[key]=c.read(c.OUT/name)
    result["failures"]={p.name:c.read(p) for p in c.OUT.glob("*failure*.json")}
    result["training_complete"]=len(list(c.OUT.glob("training/*/complete.json")))
    result["evaluation_complete"]=len(list(c.OUT.glob("evaluation/*/complete.json")))
    result["final_summary"]=(c.OUT/"final_summary.json").exists()
    if os.name!="nt":
        ps=subprocess.run(["ps","-eo","pid,ppid,pgid,etime,%cpu,args"],capture_output=True,text=True).stdout.splitlines()
        result["processes"]=[line for line in ps if "research20261001/run_endpoint.py" in line]
    result["tails"]={}
    for path in c.OUT.glob("training/*/log.jsonl"):
        with path.open("rb") as f:f.seek(max(0,path.stat().st_size-16384));lines=f.read().splitlines()
        if lines:
            try:result["tails"][path.parent.name]=json.loads(lines[-1])
            except json.JSONDecodeError:pass
    if (c.OUT/"run_protocol.json").exists():
        c.check_sources(c.read(c.OUT/"run_protocol.json")["sources"], enforce_deadline=False);result["source_hashes"]="passed"
    for name in ["reference/qa.json","reference/complete.json","final_lock.json"]:
        if (c.OUT/name).exists():result[name]=c.read(c.OUT/name)
    dest=c.EXPORT/("monitor_"+time.strftime("%Y%m%d_%H%M%S",time.gmtime())+".json")
    c.write(dest,result);return dict(path=str(dest),**result)


if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="mode",required=True)
    sub.add_parser("monitor");sub.add_parser("audit");sub.add_parser("closure")
    v=sub.add_parser("verify");v.add_argument("--archive",required=True);v.add_argument("--manifest",required=True);v.add_argument("--receipt",required=True)
    u=sub.add_parser("union");u.add_argument("--manifest",required=True);u.add_argument("--backup-dir",required=True,action="append");u.add_argument("--receipt",required=True)
    a=p.parse_args(); c.set_clock()
    print(json.dumps(monitor() if a.mode=="monitor" else audit_all() if a.mode=="audit" else closure() if a.mode=="closure" else verify(a.archive,a.manifest,a.receipt) if a.mode=="verify" else union(a.manifest,a.backup_dir,a.receipt),ensure_ascii=False))
