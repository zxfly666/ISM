"""Isolated protocol, immutable outputs and addressed random streams."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/"docs/research_reboot_20260921"
PLAN = DOCS/"DENSE_MULTISIZE_CONTROL_PLAN_20260928_ZH.md"
CONFIG_PATH = DOCS/"DENSE_MULTISIZE_CONTROL_CONFIG_20260928.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT/"artifacts"/CFG["study"]
PARENT = ROOT/CFG["parent_study"]
EXPORT = ROOT/"artifacts/dense_multisize_control_exports_20260928"
BETA = math.log1p(math.sqrt(2))/2


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ahash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        if a.dtype.hasobject:
            raise TypeError("Object arrays forbidden")
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    os.replace(str(tmp), str(path))


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    assert not any(np.asarray(a).dtype.hasobject for a in arrays.values())
    with path.with_name(path.name+".tmp").open("xb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(str(path.with_name(path.name+".tmp")), str(path))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def rng(label, step, role):
    tag = np.frombuffer(hashlib.sha256(role.encode()).digest()[:16], dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([CFG["root_seed"], int(label), int(step), *map(int, tag)]))


def initial_seed(label):
    return int(rng(label, 0, "initialization").integers(0, 2**31))


def deadline():
    b = read(OUT/"budget.json")
    if time.time() >= b["deadline"]:
        raise TimeoutError("Original two-hour diagnostic deadline exhausted; no extension")
    return b["deadline"]


def disk():
    free = {k: shutil.disk_usage(k+":/").free for k in ("C", "D")}
    if min(free.values()) < 2*1024**3:
        raise RuntimeError("Keep at least 2GiB on each drive; no deletion authorized")
    return free


def metrics(p, y):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    assert p.shape == y.shape and np.isfinite(p).all() and np.isfinite(y).all()
    assert (p >= 0).all() and (p <= 1).all() and (y > 0).all() and (y < 1).all()
    pc = p.clip(1e-12, 1-1e-12)
    entropy = -(y*np.log(y)+(1-y)*np.log1p(-y))
    ce = -(y*np.log(pc)+(1-y)*np.log1p(-pc))
    assert (ce-entropy).min() >= -1e-12
    return dict(ce=ce, kl=np.maximum(ce-entropy, 0), error=p-y,
                brier=y*(1-p)**2+(1-y)*p**2, excess_brier=(p-y)**2)


def summary(p, y):
    m = metrics(p, y)
    d = dict(n=len(y), mean_ce=float(m["ce"].mean()), mean_kl=float(m["kl"].mean()),
             max_kl=float(m["kl"].max()), probability_mae=float(np.abs(m["error"]).mean()),
             max_probability_error=float(np.abs(m["error"]).max()),
             expected_brier=float(m["brier"].mean()), excess_brier=float(m["excess_brier"].mean()))
    d["ability_passed"] = bool(d["max_kl"] <= CFG["max_mode_kl"] and d["max_probability_error"] <= CFG["max_probability_error"])
    return d


def source_snapshot():
    paths = [PLAN, CONFIG_PATH]+sorted(Path(__file__).parent.glob("*.py"))
    previous = read(PARENT/"run_protocol.json")["sources"]
    paths += [ROOT/p for p in previous]
    paths += [ROOT/"ism_diffusion/__init__.py", ROOT/"ism_diffusion/ising.py",
              ROOT/"docs/TEACHER_RESEARCH_IDEAS_QUOTES_ZH.md"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}


def check_sources(files):
    bad = [p for p, digest in files.items() if sha(ROOT/p) != digest]
    if bad:
        raise RuntimeError("Frozen source mismatch: "+repr(bad))


def parent_verification():
    prior_exports = ROOT/"artifacts/basic_capability_stages12_exports_20260928"
    receipt = read(prior_exports/"stages12_complete_v1.verification.json")
    manifest = read(prior_exports/"stages12_complete_v1.manifest.json")
    assert receipt["status"] == "passed"
    assert sha(prior_exports/manifest["archive"]) == receipt["archive_sha256"] == manifest["archive_sha256"]
    for row in manifest["files"]:
        path = ROOT/row["path"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    protocol = read(PARENT/"run_protocol.json")
    finals = {}
    for seed in CFG["parent_seeds"]:
        folder = PARENT/("training/seed_%d" % seed)
        digest = sha(folder/"final.pt")
        assert digest == read(folder/"complete.json")["final_sha256"]
        finals[str(seed)] = digest
    return dict(status="passed", previous_members=len(manifest["files"]),
        previous_archive_sha256=receipt["archive_sha256"], previous_protocol_hash=protocol["protocol_hash"],
        bank_sha256=sha(PARENT/"bank.npz"), old_final_sha256=finals, original_files_read_only=True)
