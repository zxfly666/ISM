"""Read-only check of this evaluation's exact Git index delta.

No scientific computation, checkpoint loading, network, staging or publication.
An optional receipt is the only output file written by this management check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
BASE = "b8dee83a77e14049c866cbc77a6a4ff0859aea5a"
SLUG = "late-mask-coverage-and-committed-spin-correction"
DOCS = {"README.md", ".gitattributes", "experiments/README.md", "experiments/REPORT_ZH.md",
        "experiments/CORRECTIONS.md", "experiments/DATA_AVAILABILITY.md", "experiments/REPRODUCING.md",
        "experiments/NEWCOMER_TEST.md", "experiments/registry.json", "experiments/source-manifest.json",
        "experiments/publication-allowlist.json", "experiments/validation-evaluation-20261001.json",
        "experiments/index-validation-evaluation-20261001.json",
        "scripts/archive20261001/check_evaluation_publication.py"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt")
    args = ap.parse_args()
    entries = {}
    for line in git("ls-files", "--stage", "-z").split(b"\0"):
        if not line:
            continue
        meta, path = line.split(b"\t", 1)
        _, oid, stage = meta.split()
        assert stage == b"0"
        entries[path.decode()] = oid.decode()
    delta = [v.decode() for v in git("diff", "--cached", "--name-only", BASE, "-z").split(b"\0") if v]
    errors = []
    code_sources = json.loads((ROOT / "experiments/source-manifest.json").read_text())
    for row in code_sources:
        rel = row["path"]
        raw = git("cat-file", "blob", entries[rel])
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            errors.append(dict(kind="frozen source index hash mismatch", path=rel))
    for provenance in (ROOT / "experiments").rglob("provenance.json"):
        for row in json.loads(provenance.read_text())["files"]:
            rel = "experiments/" + row["published"]
            if rel not in entries:
                errors.append(dict(kind="evidence absent from index", path=rel))
                continue
            raw = (ROOT / rel).read_bytes()
            oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if oid != entries[rel] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                errors.append(dict(kind="evidence index identity mismatch", path=rel))
    patterns = [re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{25,}\b|github_pat_[A-Za-z0-9_]{35,}"),
                re.compile(rb"(?i)[\"']?(?:password|passwd|api_key|access_token|client_secret)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9/+_-]{16,}[\"']")]
    total = largest = links = 0
    for rel in delta:
        allowed = rel in DOCS or rel.startswith("experiments/" + SLUG + "/") or rel.startswith("scripts/research20261001_eval/") and rel.endswith(".py")
        if not allowed:
            errors.append(dict(kind="outside this publication allowlist", path=rel))
        if rel not in entries:
            errors.append(dict(kind="unexpected deletion", path=rel))
            continue
        raw = (ROOT / rel).read_bytes()
        oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if oid != entries[rel]:
            # Management/README text may be normalized by Git. Audit actual staged bytes.
            raw = git("cat-file", "blob", entries[rel])
        total += len(raw)
        largest = max(largest, len(raw))
        if len(raw) > 50 * 1024**2 or rel.endswith((".pt", ".pth", ".tar.gz", ".zip")):
            errors.append(dict(kind="unexpected large/binary archive or checkpoint payload", path=rel))
        if any(p.search(raw) for p in patterns):
            errors.append(dict(kind="possible credential; value suppressed", path=rel))
        if rel.endswith(".md"):
            content = re.sub(r"```.*?```", "", raw.decode("utf-8-sig"), flags=re.S)
            for m in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", content):
                target = m.group(1).strip("<>").split(' "')[0]
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                links += 1
                dest = ((ROOT / rel).parent / unquote(target.split("#")[0])).resolve()
                try:
                    name = dest.relative_to(ROOT).as_posix()
                except ValueError:
                    errors.append(dict(kind="link outside repo", path=rel))
                    continue
                if name not in entries and not any(x.startswith(name.rstrip("/") + "/") for x in entries):
                    errors.append(dict(kind="link missing from index", path=rel, target=target))
    result = dict(passed=not errors, comparison_base=BASE, indexed_files=len(entries),
                  delta_files=len(delta), delta_bytes=total, largest_delta_bytes=largest,
                  changed_markdown_links_checked=links, frozen_sources_checked=len(code_sources),
                  errors=errors, scope="Exact staged evaluation-publication delta, not all local untracked files. No scientific computation.")
    if args.receipt:
        (ROOT / args.receipt).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
