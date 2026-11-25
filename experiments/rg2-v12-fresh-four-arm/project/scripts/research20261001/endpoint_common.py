"""New endpoint campaign identities/utilities. Import has no compute/write side effects."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research_reboot_20260921"
CONFIG_PATH = DOCS / "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_CONFIG_20261001.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
STUDY = CFG["experiment_id"]
OUT = ROOT / "artifacts" / STUDY
EXPORT = ROOT / "artifacts" / (STUDY + "_exports")
DATA = ROOT / "data/level1/parents_l1024.npz"
ARMS = ("original-support", "lower-time-floor", "explicit-late-mask")
SEEDS = tuple(CFG["base"]["seeds"])
WIDTHS = tuple(CFG["training"]["widths"])
TOKENS = CFG["training"]["tokens_per_update"]
MODEL = dict(d_model=128, n_heads=4, n_blocks=7, mlp_ratio=4.0,
             dropout=0., vocab_size=4, output_classes=2, rope_base=10000.)
BETA = .5 * np.log(1 + np.sqrt(2))
DEADLINE = float("inf")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as f:
        f.write(text)
    os.replace(temp, path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(*values):
    h = hashlib.sha256()
    for v in values:
        a = np.ascontiguousarray(v)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def increment(old, value):
    return hashlib.sha256((old + value).encode()).hexdigest()


def rng(root, index, role):
    tag = np.frombuffer(hashlib.sha256(str(role).encode()).digest()[:16], dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([2026100101, int(root), int(index), *map(int, tag)]))


def stream(root, index, role):
    return int(rng(root, index, role).integers(0, 2**63 - 1))


def check():
    if time.time() >= DEADLINE:
        raise TimeoutError("Immutable endpoint campaign deadline reached")


def set_clock():
    global DEADLINE
    if (OUT / "budget.json").exists():
        DEADLINE = read(OUT / "budget.json")["deadline"]


def log(stage, **values):
    row = dict(time=time.time(), stage=stage, **values)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write(OUT / "status.json", row, exclusive=False)
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def base_path(seed):
    return ROOT / "artifacts/geometry_identification_20260926/training" / f"s{seed}_I-F/final.pt"


def source_paths():
    paths = list((ROOT / "scripts/research20261001").glob("*.py"))
    paths += list((ROOT / "ism_diffusion").glob("*.py"))
    paths += [CONFIG_PATH, DOCS / "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_PLAN_20261001_ZH.md",
              DOCS / "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_BUDGET_20261001.json",
              DOCS / "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_AUTHORIZATION_20261001_ZH.md",
              DOCS / "GEOMETRY_IDENTIFICATION_CONFIG_20260926.json",
              ROOT / "scripts/research20260926/identification_common.py",
              ROOT / "scripts/research20260926/identification_data.py",
              ROOT / "scripts/research20260926/identification_training.py", DATA]
    return sorted(set(paths))


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths()}


def check_sources(expected, enforce_deadline=True):
    for name, value in expected.items():
        if enforce_deadline: check()
        if sha(ROOT / name) != value:
            raise RuntimeError("Frozen source/data drift: " + name)


def resources():
    result = dict(time=time.time(), disk_free=shutil.disk_usage(ROOT).free)
    if os.name != "nt":
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
        result["gpu"] = p.stdout.strip()
    return result


def import_old():
    for p in (ROOT, ROOT / "scripts/research20260926"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

