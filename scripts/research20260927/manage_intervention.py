"""J-only monitor, immutable staged export and complete local SHA verification.

No science starts here. No deletion, no overwrite of archives, no old job control.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path
import intervention_common as c


def latest(path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size-12000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue


def monitor():
    result = dict(time=time.time(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for key in ("budget", "status", "processes", "preflight_process", "reference_process", "final_summary", "failure", "reference_failure"):
        p = c.OUT/(key+".json"); result[key] = c.read(p) if p.exists() else None
    for key, command in dict(process_table=["ps", "-eo", "pid,ppid,pgid,stat,pcpu,pmem,etime,args"],
                             gpu=["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                             disk=["df", "-B1", str(c.ROOT)]).items():
        value = subprocess.run(command, capture_output=True, text=True, timeout=15).stdout
        if key == "process_table":
            value = "\n".join(line for line in value.splitlines() if "research20260927" in line or "multiprocessing" in line)
        result[key] = value
    result["training"] = {p.name: dict(last=latest(p/"train.jsonl"), complete=(p/"complete.json").exists())
                          for p in (c.OUT/"training").glob("s*")}
    result["counts"] = dict(training_complete=len(list((c.OUT/"training").glob("*/complete.json"))),
        evaluation_complete=len(list((c.OUT/"evaluation").glob("*/complete.json"))),
        predictions=len(list((c.OUT/"evaluation").glob("*/*/*.npz"))),
        validation_predictions=len(list((c.OUT/"training").glob("*/validation/*/*.npz"))))
    if (c.OUT/"run_protocol.json").exists():
        p = c.read(c.OUT/"run_protocol.json")
        result["source_mismatch"] = [name for name, sha in p["files"].items() if c.sha(c.ROOT/name) != sha]
        result["final_mismatch"] = [x.parent.name for x in (c.OUT/"training").glob("*/complete.json")
                                    if c.sha(x.parent/"final.pt") != c.read(x)["final_sha256"]]
        result["run_protocol_sha256"] = c.sha(c.OUT/"run_protocol.json")
    result["reference_QA"] = c.read(c.OUT/"reference/qa.json") if (c.OUT/"reference/qa.json").exists() else None
    if result["budget"]:
        result["remaining_hours"] = (result["budget"]["deadline"]-time.time())/3600
    c.EXPORTS.mkdir(exist_ok=True)
    path = c.EXPORTS/("monitor_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")
    c.write(path, result)
    print(json.dumps(dict(path=str(path), counts=result["counts"], status=result["status"], failures=[result["failure"], result["reference_failure"]])))


EXCLUDE = {"last.pt", "restore_fixture.pt", "timing.pt", "fixture_dense.pt", "fixture_observed_only.pt"}


def export(stage, cell, name):
    if not name or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in name):
        raise ValueError("Safe unique archive name required")
    if stage == "training":
        if cell not in {x["cell"] for x in c.cells()}:
            raise ValueError(cell)
        folder = c.OUT/"training"/cell
        complete = c.read(folder/"complete.json")
        if complete["status"] != "complete" or c.sha(folder/"final.pt") != complete["final_sha256"]:
            raise RuntimeError("Incomplete training export")
        paths = list(folder.rglob("*"))
    elif stage == "evaluation":
        if cell not in {x["cell"] for x in c.evaluation_identities()}:
            raise ValueError(cell)
        folder = c.OUT/"evaluation"/cell
        if c.read(folder/"complete.json")["status"] != "complete":
            raise RuntimeError("Incomplete evaluation export")
        paths = list(folder.rglob("*"))
    elif stage in ("reference", "banks"):
        folder = c.OUT/stage
        if not (folder/"complete.json").exists():
            raise RuntimeError("Incomplete stage export")
        paths = list(folder.rglob("*"))
        if stage == "banks":
            paths += list((c.OUT/"diagnostic_banks").rglob("*"))
    elif stage == "initial":
        if not (c.OUT/"run_protocol.json").exists():
            raise RuntimeError("Freeze first")
        paths = c.source_files()+list(c.OUT.glob("*.json"))+list((c.OUT/"design").rglob("*"))+list((c.OUT/'cpu_preparation').rglob('*'))
        for folder in c.OUT.glob("preflight*"):
            if folder.is_dir():
                paths += list(folder.rglob("*"))
        paths += list((c.EXPORTS/'software_scratch_cpu_v1').rglob('*'))
    elif stage == "final":
        if not (c.OUT/"final_summary.json").exists():
            raise RuntimeError("Remote summary absent")
        paths = list(c.OUT.glob("*"))+list((c.OUT/"analysis").rglob("*"))+list((c.OUT/"validation_banks").rglob("*"))
    else:
        raise ValueError(stage)
    paths = sorted(set(p for p in paths if p.is_file() and p.name not in EXCLUDE and not p.name.endswith(".tmp")))
    c.EXPORTS.mkdir(exist_ok=True)
    archive = c.EXPORTS/(name+".tar.gz"); manifest = c.EXPORTS/(name+".manifest.json")
    if archive.exists() or manifest.exists():
        raise FileExistsError(name)
    rows = []
    began = time.perf_counter()
    with archive.open("xb") as f:
        with tarfile.open(fileobj=f, mode="w:gz", compresslevel=2, dereference=True) as tar:
            for p in paths:
                h = c.sha(p); rel = p.relative_to(c.ROOT).as_posix()
                rows.append(dict(path=rel, bytes=p.stat().st_size, sha256=h))
                tar.add(p, arcname=rel, recursive=False)
                if c.sha(p) != h:
                    raise RuntimeError("Source changed during backup")
    result = dict(stage=stage, cell=cell, files=rows, archive=archive.name, members=len(rows),
                  bytes=archive.stat().st_size, sha256=c.sha(archive), seconds=time.perf_counter()-began)
    c.write(manifest, result)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


def verify_archive(archive, manifest):
    ar = Path(archive); m = c.read(manifest)
    if ar.stat().st_size != m["bytes"] or c.sha(ar) != m["sha256"]:
        raise RuntimeError("Archive SHA/size mismatch")
    rows = {x["path"]: x for x in m["files"]}
    if len(rows) != len(m["files"]):
        raise RuntimeError("Duplicate member in manifest")
    seen = set()
    with tarfile.open(ar, "r:gz") as tar:
        for member in tar:
            name = member.name
            if not member.isfile() or name not in rows or name in seen or name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError("Unexpected or unsafe archive member")
            seen.add(name); h = hashlib.sha256()
            with tar.extractfile(member) as f:
                for part in iter(lambda: f.read(4*1024**2), b""):
                    h.update(part)
            if member.size != rows[name]["bytes"] or h.hexdigest() != rows[name]["sha256"]:
                raise RuntimeError("Archive member mismatch: "+name)
    if seen != set(rows):
        raise RuntimeError("Missing archived members")
    return rows, m


def verify(archive, manifest, receipt):
    rows, m = verify_archive(archive, manifest)
    c.write(receipt, dict(status="passed", archive=str(Path(archive).resolve()), archive_sha256=m["sha256"],
                          verified_members=len(rows), time=time.time()))
    print("VERIFIED", len(rows), m["sha256"])


def union(science_manifest, backup_dir, receipt):
    science = c.read(science_manifest); backup_dir = Path(backup_dir)
    available = {}; packages = []
    for p in sorted(backup_dir.glob("*.manifest.json")):
        receipt_path = p.with_name(p.name.replace(".manifest.json", ".verification.json"))
        if not receipt_path.exists() or c.read(receipt_path)["status"] != "passed":
            continue
        ar = p.with_name(p.name.replace(".manifest.json", ".tar.gz"))
        rows, m = verify_archive(ar, p)
        packages.append(dict(archive=ar.name, members=len(rows), sha256=m["sha256"]))
        for name, row in rows.items():
            available.setdefault(name, set()).add((row["bytes"], row["sha256"]))
    missing = []
    prefix = "artifacts/"+c.CONFIG["study"]+"/"
    for name, row in science.items():
        if (row["bytes"], row["sha256"]) not in available.get(prefix+name, set()):
            missing.append(name)
    c.write(receipt, dict(status="passed" if not missing else "failed_missing_science", science_files=len(science),
                          covered=len(science)-len(missing), missing=missing, packages=packages, time=time.time()))
    if missing:
        raise RuntimeError("Incomplete scientific backup coverage")
    print("UNION_VERIFIED", len(science))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["monitor", "export", "verify", "union"])
    for flag in ("stage", "cell", "name", "archive", "manifest", "receipt", "backup_dir"):
        p.add_argument("--"+flag.replace("_", "-"))
    a = p.parse_args()
    if a.mode == "monitor": monitor()
    elif a.mode == "export": export(a.stage, a.cell, a.name)
    elif a.mode == "verify": verify(a.archive, a.manifest, a.receipt)
    else: union(a.manifest, a.backup_dir, a.receipt)
