"""Verify a completed diagnostic archive and safely extract its small evidence.

No experiment mutation. This is closure-time auditing, not a new model run.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/"artifacts/fixed_geometry_joint_20260923_remote"
ARCHIVE=LOCAL/"fixed_geometry_joint_20260923_evidence_v1.tar.gz"
PREFIX="fixed_geometry_joint_20260923"
SHA="35c142269f0f0f6fef772df57256c2e5c82a0f5d7575ec8dcb837c05e3a411d7"


def digest(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1048576),b""): h.update(chunk)
    return h.hexdigest()


def main():
    assert ARCHIVE.stat().st_size==9365671
    with ARCHIVE.open("rb") as f: assert digest(f)==SHA
    target=LOCAL/"evidence"
    assert not target.exists(), "No overwrite of an earlier extraction"
    with tarfile.open(ARCHIVE,"r:gz") as tar:
        members=tar.getmembers()
        files={}
        for member in members:
            path=PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in member.name
            assert path.parts[0] in (PREFIX,PREFIX+"_queue.log")
            assert member.isfile() or member.isdir(), "No links/devices"
            assert (target/member.name).resolve().is_relative_to(target.resolve())
            if member.isfile():
                assert member.name not in files
                with tar.extractfile(member) as f: files[member.name]=dict(bytes=member.size,sha256=digest(f))
        with tar.extractfile(PREFIX+"/manifest.json") as f: manifest=json.load(f)
        assert len({x['path'] for x in manifest['files']})==len(manifest['files'])
        for entry in manifest["files"]:
            assert files[PREFIX+"/"+entry["path"]]=={k:entry[k] for k in ("bytes","sha256")}
        extras=set(files)-{PREFIX+"/"+e["path"] for e in manifest["files"]}
        assert extras=={PREFIX+"/manifest.json",PREFIX+"/status.json",PREFIX+"/run.lock",PREFIX+"_queue.log"},extras
        target.mkdir()
        for member in members:
            dest=target/member.name
            if member.isdir(): dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,dest.open("xb") as f: shutil.copyfileobj(src,f)
    base=target/PREFIX
    load=lambda name:json.loads((base/name).read_text(encoding="utf-8"))
    final=load("final_summary.json");protocol=load("run_protocol.json")
    assert final["models"]==36 and final["forward_inputs"]==80640 and final["checkpoint_unchanged"]
    assert final["status"]=="remote_complete_requires_local_backup_and_visual_review"
    assert load("inference_complete.json")["status"]=="complete"
    assert load("reference_complete.json")["status"]=="complete"
    assert load("analysis/complete.json")["status"]=="complete"
    assert not (base/"failure.json").exists()
    total=0
    for seed in range(91001,91007):
        for arm in ("A","B","C","F0","F1","F2"):
            cell=f"s{seed}_{arm}"
            c=load(f"models/{cell}/complete.json")
            assert c["checkpoint_unchanged"] and c["checkpoint_sha256"]==protocol["checkpoints"][cell]
            assert c["cases"]==112 and c["inputs"]==2240
            assert len(load(f"models/{cell}/metrics.json"))==112
            with np.load(base/f"models/{cell}/predictions.npz") as z:
                assert z["probabilities"].shape==(112,20,3)
                assert z["input_sha256"].shape==(112,)
                assert np.isfinite(z["probabilities"]).all()
            total+=c["inputs"]
    assert total==80640
    for block,reps in ((8,1000),(4,500),(16,500)):
        with np.load(base/f"analysis/bootstrap_block{block}.npz") as z:
            assert z["values"].shape==(reps,4,6,11) and np.isfinite(z["values"]).all()
    for name in ("geometry_response_ABC","geometry_response_F012","joint_accuracy"):
        for ext in ("png","pdf"): assert (base/f"analysis/figures/{name}.{ext}").is_file()
    result=dict(status="passed",archive=ARCHIVE.name,archive_sha256=SHA,bytes=ARCHIVE.stat().st_size,
        archive_files=len(files),manifest_files=len(manifest["files"]),models=36,forward_inputs=total,
        checkpoint_unchanged=True,elapsed_seconds=final["elapsed_seconds"],
        scope="All newly produced diagnostic evidence including reference sufficient statistics and raw predictions; not a new backup of original checkpoints or full parent fields; no GitHub upload",
        visual_review="pending")
    (LOCAL/"backup_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":main()
