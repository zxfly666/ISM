"""Read-only staged RG-publication audit; optional management receipt only.

No GPU, training, new sampling, bootstrap, checkpoint loading, network or staging.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = "fd9da6381298b087250d0c99b0b1b9713376eaa3"
SLUG = "rg-coarse-data-training-transfer"
DOCS = {"README.md", ".gitattributes", "experiments/README.md", "experiments/REPORT_ZH.md",
        "experiments/DATA_AVAILABILITY.md", "experiments/REPRODUCING.md", "experiments/registry.json",
        "experiments/source-manifest.json", "experiments/publication-allowlist.json",
        "experiments/validation-rgpilot-20261001.json", "scripts/archive20261001/check_rgpilot_publication.py",
        "docs/research_reboot_20260921/RG_DATA_TRAINING_PILOT_CONFIG_20261001.json",
        "docs/research_reboot_20260921/RG_DATA_TRAINING_PILOT_PLAN_20261001_ZH.md"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt")
    args = ap.parse_args()
    entries = {}
    for line in git("ls-files", "--stage", "-z").split(b"\0"):
        if line:
            meta, path = line.split(b"\t", 1)
            _, oid, stage = meta.split()
            assert stage == b"0"
            entries[path.decode()] = oid.decode()
    delta = [p.decode() for p in git("diff", "--cached", "--name-only", BASE, "-z").split(b"\0") if p]
    errors, links, total, largest = [], 0, 0, 0
    pub = ROOT / "experiments" / SLUG
    protocol = read(pub / "evidence/run_protocol.json")
    frozen_checked = 0
    for row in protocol["sources"]:
        rel = row["path"]
        if rel.startswith("scripts/research20261001_rgpilot/") or rel in DOCS:
            raw = git("cat-file", "blob", entries[rel])
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                errors.append(dict(kind="frozen source index mismatch", path=rel))
            frozen_checked += 1
    provenance = read(pub / "provenance.json")["files"]
    for row in provenance:
        rel = "experiments/" + row["published"]
        raw = (ROOT / rel).read_bytes()
        oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if entries.get(rel) != oid or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            errors.append(dict(kind="public provenance/index mismatch", path=rel))
    patterns = [re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{25,}\b|github_pat_[A-Za-z0-9_]{35,}"),
                re.compile(rb"(?i)[\"']?(?:password|passwd|api_key|access_token|client_secret)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9/+_-]{16,}[\"']"),
                re.compile(rb"(?i)[A-Z]:[/\\]+Users[/\\]+[A-Za-z0-9]|/root/shared[-]nvme/|219\.146\.211\.42")]
    for rel in delta:
        if not (rel in DOCS or rel.startswith("experiments/" + SLUG + "/")
                or rel.startswith("scripts/research20261001_rgpilot/") and rel.endswith(".py")):
            errors.append(dict(kind="outside publication scope", path=rel))
        if rel not in entries:
            errors.append(dict(kind="unexpected deletion", path=rel))
            continue
        raw = (ROOT / rel).read_bytes()
        oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if entries[rel] != oid:
            raw = git("cat-file", "blob", entries[rel])
        total += len(raw)
        largest = max(largest, len(raw))
        if rel.endswith((".pt", ".pth", ".tar.gz", ".zip")) or len(raw) > 20 * 1024**2:
            errors.append(dict(kind="unexpected weight/archive/large file", path=rel))
        if rel.endswith((".py", ".md", ".json", ".txt", ".jsonl")) and any(p.search(raw) for p in patterns):
            errors.append(dict(kind="possible private infrastructure or credential; value suppressed", path=rel))
        if rel.endswith(".md"):
            text = re.sub(r"```.*?```", "", raw.decode("utf-8-sig"), flags=re.S)
            for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                target = match.group(1).strip("<>").split(' "')[0]
                if target.startswith(("https:", "http:", "mailto:", "#")):
                    continue
                links += 1
                dest = ((ROOT / rel).parent / unquote(target.split("#")[0])).resolve()
                try:
                    name = dest.relative_to(ROOT).as_posix()
                except ValueError:
                    errors.append(dict(kind="external local link", path=rel))
                    continue
                if name not in entries and not any(x.startswith(name.rstrip("/") + "/") for x in entries):
                    errors.append(dict(kind="missing link target", path=rel, target=target))
    predictions = list((pub / "evidence/predictions").rglob("*.npz"))
    for path in predictions:
        with np.load(path, allow_pickle=False) as z:
            p, y = z["probabilities"], z["labels"]
            ce = -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean(1)
            if not np.allclose(ce, z["empirical_ce"], atol=1e-12, rtol=0):
                errors.append(dict(kind="public CE reconstruction mismatch", path=str(path.relative_to(ROOT))))
    summary = read(pub / "evidence/analysis/summary.json")
    arrays = {}
    for arm in ("fine_only", "fine_plus_rg", "fine_replay"):
        means = []
        for seed in (92601, 92602, 92603):
            vals = []
            for mask in (1, 8, 32):
                path = pub / f"evidence/predictions/s{seed}_{arm}/final/test/fine_W96_M{mask}.npz"
                with np.load(path, allow_pickle=False) as z:
                    vals.append(z["exact_kl"])
            means.append(float(np.stack(vals).mean()))
        arrays[arm] = np.array(means)
        expected = summary["summaries"]["primary_fine_W96_KL"]["arms"][arm]["seed_means"]
        assert np.allclose(means, expected, atol=1e-14, rtol=0)
    point = float((arrays["fine_plus_rg"] - arrays["fine_replay"]).mean())
    assert abs(point - summary["summaries"]["primary_fine_W96_KL"]["comparisons"]["fine_plus_rg_minus_fine_replay"]["mean"]) < 1e-14
    assert len(predictions) == 297 and frozen_checked == 8
    result = dict(passed=not errors, comparison_base=BASE, delta_files=len(delta), delta_bytes=total,
        largest_delta_bytes=largest, links_checked=links, frozen_sources_and_documents_checked=frozen_checked,
        provenance_records=len(provenance), public_prediction_files=297, all_public_empirical_CE_recomputed=True,
        saved_primary_KL_means_reproduced=True, primary_difference=point, bootstrap_rerun=False,
        oracle_KL_reconstruction_requires_withheld_banks=True, errors=errors,
        scope="Exact staged RG publication; byte checks and existing-value arithmetic only, no new scientific run.")
    if args.receipt:
        target = ROOT / args.receipt
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
