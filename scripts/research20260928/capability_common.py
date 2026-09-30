"""Shared, isolated I/O, streams and scoring for authorized stages 1 and 2."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research_reboot_20260921"
CONFIG_PATH = DOCS / "BASIC_CAPABILITY_STAGE12_CONFIG_20260928.json"
PLAN_PATH = DOCS / "BASIC_CAPABILITY_STAGE12_PLAN_20260928_ZH.md"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / "artifacts" / CONFIG["study"]
BETA = math.log1p(math.sqrt(2.0)) / 2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not allowed")
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    os.replace(str(temp), str(path))


def save(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for value in values.values():
        if np.asarray(value).dtype.hasobject:
            raise TypeError("Object arrays are not allowed")
    temp = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError(str(path))
    with temp.open("xb") as f:
        np.savez_compressed(f, **values)
    if path.exists():
        raise FileExistsError(str(path))
    os.replace(str(temp), str(path))


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def stream(label, step, role):
    tag = np.frombuffer(hashlib.sha256(role.encode()).digest()[:16], dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(
        [CONFIG["root_seed"], int(label), int(step), *map(int, tag)]))


def model_seed(label, role="training"):
    return int(stream(label, 0, "model_initialization_" + role).integers(0, 2**31))


def metrics(probability, target):
    p, y = np.asarray(probability, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Invalid probability/target shape or finite status")
    if not ((0 <= p).all() and (p <= 1).all() and (0 < y).all() and (y < 1).all()):
        raise ValueError("Invalid probability bounds")
    pc = np.clip(p, 1e-12, 1-1e-12)
    entropy = -(y*np.log(y) + (1-y)*np.log1p(-y))
    ce = -(y*np.log(pc)+(1-y)*np.log1p(-pc))
    kl = ce-entropy
    if kl.min() < -1e-12:
        raise ValueError("Negative KL beyond rounding")
    return dict(ce=ce, kl=np.maximum(kl, 0), error=p-y,
                brier=y*(1-p)**2+(1-y)*p**2, excess_brier=(p-y)**2)


def summarize(probability, target):
    m = metrics(probability, target)
    result = dict(n=int(np.size(target)), mean_ce=float(m["ce"].mean()),
                  mean_kl=float(m["kl"].mean()), max_kl=float(m["kl"].max()),
                  probability_mae=float(np.abs(m["error"]).mean()),
                  max_probability_error=float(np.abs(m["error"]).max()),
                  expected_brier=float(m["brier"].mean()),
                  excess_brier=float(m["excess_brier"].mean()))
    result["diagnostic_capability_passed"] = bool(
        result["max_kl"] <= CONFIG["max_mode_kl"] and
        result["max_probability_error"] <= CONFIG["max_probability_error"])
    return result


def check_deadline():
    path = OUT / "budget.json"
    if path.exists() and time.time() >= read(path)["deadline"]:
        raise TimeoutError("Fixed stage-1/2 diagnostic budget exhausted")


def source_files():
    files = [CONFIG_PATH, PLAN_PATH]
    files += sorted(Path(__file__).parent.glob("*.py"))
    files += [ROOT/"scripts/research20260927/intervention_model.py",
              ROOT/"ism_diffusion/model.py", ROOT/"ism_diffusion/scale_model.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def check_frozen(files):
    mismatch = [p for p, digest in files.items() if sha(ROOT/p) != digest]
    if mismatch:
        raise RuntimeError("Source changed after freeze: " + repr(mismatch))

