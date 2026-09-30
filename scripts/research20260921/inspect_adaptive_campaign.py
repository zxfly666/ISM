"""Read-only scientific checks; writes only a timestamped monitoring snapshot."""
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT / "artifacts/adaptive_research_20260922"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    return dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)


def main():
    protocol=read(OUT / "repair/protocol.json");campaign=read(OUT / "campaign.json")
    rows=[]
    for folder in sorted((OUT / "repair/training").glob("s*")):
        history=read(folder / "validation_history.json") or []
        seed=folder.name.split("_")[0][1:]
        last=history[-1] if history else None
        tail=[]
        path=folder / "train.jsonl"
        if path.exists():
            with path.open() as f:tail=list(collections.deque(f,maxlen=3))
        logrows=[]
        for line in tail:
            try:logrows.append(json.loads(line))
            except json.JSONDecodeError:pass  # Writer may be appending last line.
        checkpoints={p.name:dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)
                     for p in folder.glob("*.pt")}
        rows.append(dict(cell=folder.name,last_validation=last,checkpoint_files=checkpoints,
            completed=read(folder / "complete.json"),log_tail=logrows,
            validation_change_vs_base=(last["ce"]-protocol["base_validation"][seed]["mean_ce"]) if last and protocol else None))
    source_checks={rel:sha(ROOT / rel)==digest for rel,digest in protocol["source_hashes"].items()} if protocol else {}
    base_checks={}
    if protocol:
        base=ROOT / "artifacts/geometry_alignment_20260921" if protocol["selected_size"]=="S" else OUT / "capacity"
        tag="A" if protocol["selected_size"]=="S" else "M"
        base_checks={seed:sha(base / f"training/s{seed}_{tag}/final.pt")==digest for seed,digest in protocol["base_hashes"].items()}
    now=time.time();stamp=dt.datetime.fromtimestamp(now,dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    statuses={stage:read(OUT / stage / "status.json") for stage in ("capacity","repair","final_analysis")}
    result=dict(observed_unix=now,observed_utc=stamp,remaining_hours=(campaign["deadline"]-now)/3600,
        statuses=statuses,reference_complete=read(OUT / "reference/complete.json"),
        final_summary=read(OUT / "final_summary.json"),training=rows,
        source_unchanged=source_checks,base_unchanged=base_checks,
        processes=command(["ps","-p","121147,121354,121615,121782","-o","pid,etime,pcpu,args"]),
        gpu=command(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw","--format=csv,noheader"]),
        disk=command(["df","-h",str(ROOT)]))
    folder=OUT / "monitoring";folder.mkdir(exist_ok=True)
    path=folder / (stamp+".json")
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(snapshot=str(path),rows=len(rows),remaining_hours=result["remaining_hours"],
                         sources_ok=all(source_checks.values()),bases_ok=all(base_checks.values())),indent=2))


if __name__=="__main__":main()
