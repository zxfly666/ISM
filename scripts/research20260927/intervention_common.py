"""Isolated J18h identities, deterministic streams and immutable I/O.

No imports from I-run modules, no mutation of their CONFIG/OUT/DEADLINE.
Importing this module never creates an experiment directory or budget clock.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research_reboot_20260921"
CONFIG_PATH = DOCS / "OBSERVED_CONTEXT_INTERVENTION_CONFIG_V2_18H_20260927.json"
PLAN_PATH = DOCS / "OBSERVED_CONTEXT_INTERVENTION_PLAN_V2_18H_20260927_ZH.md"
AUTH_PATH = DOCS / "OBSERVED_CONTEXT_INTERVENTION_AUTHORIZATION_20260927.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / "artifacts" / CONFIG["study"]
EXPORTS = ROOT / "artifacts/observed_context_intervention_18h_exports_20260927"
DATA = ROOT / CONFIG["training_common"]["parent_file"]
COHORTS = {row["id"]: row for row in CONFIG["cohorts"]}
TOKENS = CONFIG["training_common"]["tokens_per_update"]
WIDTHS = tuple(CONFIG["training_common"]["widths"])
MODEL = {k: v for k, v in CONFIG["model"].items() if k != "parameters"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise TypeError("Object arrays have no stable byte identity")
        h.update(str((a.shape, a.dtype.str)).encode("utf-8"))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def rng(seed, index, role):
    """Independent role paths; neither arm nor Python hash advances data RNG."""
    tag = np.frombuffer(hashlib.sha256(role.encode("utf-8")).digest()[:16], dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(
        [CONFIG["seeds"]["root"], int(seed), int(index), *map(int, tag)]))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, exclusive=True):
    """Administrative updates opt into replacement; science defaults immutable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    if exclusive and path.exists():
        raise FileExistsError(path)
    os.replace(tmp, path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as f:
        np.savez_compressed(f, **arrays)
    if path.exists():
        raise FileExistsError(path)
    os.replace(tmp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def identity(cohort, index, mode):
    spec = COHORTS[cohort]
    if mode not in ("D", "O") or not 0 <= index < 6:
        raise ValueError((cohort, index, mode))
    prefix = "C" if cohort == "continuation" else "S"
    seed = spec["base_seeds" if prefix == "C" else "initialization_seeds"][index]
    return dict(cell=f"s{seed}_{prefix}-{mode}", cohort=cohort, pair_index=index,
                seed=seed, data_seed=spec["data_seeds"][index], arm=f"{prefix}-{mode}",
                attention_mode="dense" if mode == "D" else "observed_only",
                base_step=spec["base_step"], updates=spec["updates"])


def cells():
    return [identity(cohort, index, mode) for cohort in COHORTS
            for index in range(6) for mode in ("D", "O")]


def block_order():
    for cohort, spec in COHORTS.items():
        for block in range(spec["updates"] // 1000):
            for index in range(6):
                for mode in (("D", "O") if (index + block) % 2 == 0 else ("O", "D")):
                    yield identity(cohort, index, mode), block * 1000, (block + 1) * 1000


def evaluation_identities():
    rows = []
    for index, seed in enumerate(CONFIG["base_seeds"]):
        rows.append(dict(cell=f"s{seed}_B0", seed=seed, pair_index=index, arm="B0",
                         attention_mode="dense", kind="base", snapshot=12000,
                         checkpoint=f"artifacts/geometry_identification_20260926/training/s{seed}_I-F/final.pt"))
    for ident in cells():
        rows.append({**ident, "kind": "final", "snapshot": ident["base_step"]+ident["updates"],
                     "checkpoint": f"artifacts/{CONFIG['study']}/training/{ident['cell']}/final.pt"})
    for ident in cells():
        if ident["cohort"] == "fresh":
            rows.append({**ident, "cell": ident["cell"]+"_12000", "kind": "intermediate", "snapshot": 12000,
                         "checkpoint": f"artifacts/{CONFIG['study']}/training/{ident['cell']}/ema_12000.pt"})
    return rows


def base_checkpoint(index):
    seed = CONFIG["base_seeds"][index]
    return ROOT / f"artifacts/geometry_identification_20260926/training/s{seed}_I-F/final.pt"


def source_files():
    files = [p for directory in ("ism_diffusion", "scripts/research20260921", "scripts/research20260927")
             for p in (ROOT/directory).glob("*.py")]
    files += [PLAN_PATH, CONFIG_PATH, AUTH_PATH,
        DOCS/"OBSERVED_CONTEXT_INTERVENTION_PLAN_20260927_ZH.md",
        DOCS/"OBSERVED_CONTEXT_INTERVENTION_AUTHORIZATION_20260927_ZH.md", DATA,
        ROOT/"artifacts/geometry_identification_20260926/design/low_layouts.npz",
        ROOT/"artifacts/geometry_identification_20260926/run_protocol.json"]
    return sorted(set(files))


def verify_sources(protocol):
    for name, digest in protocol["files"].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError("Frozen source/data changed: "+name)
    for i, digest in enumerate(CONFIG["base_final_sha256"]):
        if sha(base_checkpoint(i)) != digest:
            raise RuntimeError("Frozen I-F base changed")


def verified_authorization():
    auth = read(AUTH_PATH)
    if (auth["study"] != CONFIG["study"] or not auth["authorized_to_implement_and_preflight"]
            or not auth["authorized_to_start_once_only_if_all_gates_pass"]
            or auth["plan_sha256"] != sha(PLAN_PATH)
            or auth["config_sha256"] != sha(CONFIG_PATH)
            or auth["hard_seconds"] != 64800):
        raise RuntimeError("J18h authorization/design mismatch")
    return auth


class Deadline:
    """Explicit, immutable per-process deadline, never an old module global."""
    def __init__(self, epoch):
        self.epoch = float(epoch)
        if not np.isfinite(self.epoch):
            raise ValueError("A finite absolute deadline is required")

    def __call__(self):
        if time.time() >= self.epoch:
            raise TimeoutError("J18h immutable absolute deadline reached")


def budget_estimate(E, T, M, F, L, A, B):
    values = (E, T, M, F, L, A, B)
    if not all(np.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Missing/nonfinite/negative timing evidence")
    return E + max(1.25*T, 1.25*M) + 1.25*(F+L) + max(2700, 1.25*A) + 1.25*B + 900 + 1800
