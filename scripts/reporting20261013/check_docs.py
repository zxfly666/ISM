"""Read-only checks for the 2026-10-13 research documentation update.

No training, inference, resampling, network or writes. Check selected primary
table numbers against frozen summaries, local/public link closure, and evidence
hashes. This is editorial QA, not a replacement for scientific interpretation.
Use --index before publishing and optionally --base COMMIT for change scope.
"""
import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
FOCUS = (
    "size-spacing-factorial",
    "fine-geometry-identification",
    "canonical-context-size-generalization",
    "rg-coarse-data-training-transfer",
)
COMMON = (
    "README.md", "docs/README.md", "docs/L64_IMPLEMENTATION_GUIDE.md",
    "docs/SCALE_AWARE_CONTEXT_EXPERIMENT_REPORT_ZH.md",
    "results/README.md", "results/scale_aware_context/README.md",
    "experiments/README.md", "experiments/REPORT_ZH.md",
    "experiments/METHODS_ZH.md", "experiments/STUDY_GUIDE_ZH.md",
    "experiments/QUESTIONS_ZH.md", "experiments/HISTORY_ZH.md",
    "experiments/REPRODUCING.md",
)
SELF = "scripts/reporting20261013/check_docs.py"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unfenced(text):
    return re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*(?:\n|$)", "", text)


def anchors(text):
    """GitHub-style anchors for the simple ATX headings used in these docs."""
    result, used = set(), {}
    for line in unfenced(text).splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        title = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", match[1])
        title = re.sub(r"<[^>]+>", "", title).replace("`", "").lower()
        title = "".join(c for c in title if c in " -_" or
                        unicodedata.category(c)[0] in "LNM")
        slug = title.replace(" ", "-")
        count = used.get(slug, 0)
        used[slug] = count + 1
        result.add(slug if not count else f"{slug}-{count}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", action="store_true",
                    help="require all reviewed files in index and equal to working text")
    ap.add_argument("--base", help="also restrict tracked changes since this commit")
    args = ap.parse_args()
    registry = read_json("experiments/registry.json")
    docs = set(COMMON)
    docs.update(f"experiments/{r['slug']}/README.md" for r in registry)
    docs.update(f"experiments/{slug}/REPORT_ZH.md" for slug in FOCUS)
    allowed = docs | {SELF}
    tracked = set(git("ls-files", "-z").decode("utf-8").split("\0")) - {""}
    # Pending editorial files are allowed only without --index. Other untracked
    # files never count as public targets, even if they exist on this machine.
    public = tracked if args.index else tracked | allowed
    errors, link_count, anchor_count, numeric_count, hash_count = [], 0, 0, 0, 0

    def fail(kind, path):
        errors.append({"kind": kind, "path": path})

    if args.base:
        git("rev-parse", "--verify", f"{args.base}^{{commit}}")
        changed = set(git("diff", "--name-only", "-z", args.base).decode("utf-8").split("\0")) - {""}
        for path in sorted(changed - allowed):
            fail("outside documentation change scope", path)
    if sum(bool(r["formal_completed"]) for r in registry) != 14:
        fail("expected 14 completed research units; review guide", "experiments/registry.json")

    texts = {}
    private_patterns = [
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{35,})\b",
        r"(?i)[\"']?(?:password|passwd|api_key|access_token|client_secret)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9/+_-]{16,}[\"']",
        r"/root/shared-nvme/", r"(?i)D:[/\\]ISM",
        r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b",
        r"TEACHER_RESEARCH_IDEAS_QUOTES_ZH|WECHAT_TEACHER_BRIEFING_REFERENCE_ZH",
    ]
    for rel in sorted(docs):
        path = ROOT / rel
        if not path.is_file():
            fail("missing reviewed document", rel)
            continue
        text = path.read_text(encoding="utf-8-sig")
        texts[rel] = text
        if args.index:
            if rel not in tracked:
                fail("not included in index", rel)
            elif git("show", f":{rel}").decode("utf-8-sig").replace("\r\n", "\n") != text:
                fail("index differs from reviewed working text", rel)
        if any(re.search(pattern, text) for pattern in private_patterns):
            fail("possible private material; review without echo", rel)
        lines = unfenced(text).splitlines()
        for index, line in enumerate(lines):
            if re.match(r"^#{1,6}\s", line):
                if index and lines[index - 1].strip():
                    fail(f"heading needs preceding blank line {index + 1}", rel)
                if index + 1 < len(lines) and lines[index + 1].strip():
                    fail(f"heading needs following blank line {index + 1}", rel)
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", unfenced(text)):
            target = match[1].strip().split(' "')[0].strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            link_count += 1
            resolved = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            try:
                target_rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                fail("local link escapes repository", rel)
                continue
            is_public = target_rel in public or (
                resolved.is_dir() and any(p.startswith(target_rel + "/") for p in public))
            if not resolved.exists() or not is_public:
                fail("link target absent from publication: " + target_rel, rel)
            if parts.fragment and resolved.is_file() and resolved.suffix == ".md":
                anchor_count += 1
                if unquote(parts.fragment) not in anchors(resolved.read_text(encoding="utf-8-sig")):
                    fail("unknown heading anchor: " + target_rel + "#" + unquote(parts.fragment), rel)

    guide = texts.get("experiments/STUDY_GUIDE_ZH.md", "")
    for entry in registry:
        if entry["slug"] + "/" not in guide:
            fail("experiment missing from reading priorities", entry["slug"])

    def table_numbers(doc, label, expected):
        """Compare the numeric data cells, not arbitrary occurrences in prose."""
        nonlocal numeric_count
        lines = [s for s in texts[doc].splitlines() if s.startswith("| " + label + " |")]
        if len(lines) != 1:
            fail("missing or ambiguous result row: " + label, doc)
            return
        cells = lines[0].split("|")[2:]
        values = [float(s.replace("−", "-")) for s in re.findall(
            r"[+−-]?(?:\d*\.\d+)", " ".join(cells))]
        if len(values) != len(expected) or any(
                abs(a - b) > 0.00000051 for a, b in zip(values, expected)):
            fail("rounded numbers differ from frozen summary: " + label, doc)
        numeric_count += len(expected)

    g = read_json("experiments/size-spacing-factorial/evidence/analysis/crossed_uncertainty.json")
    for label, item in zip(("H1 间距覆盖", "H2 尺寸覆盖", "H3 细排列相对跨度"), g["primary"]):
        table_numbers(f"experiments/{FOCUS[0]}/REPORT_ZH.md", label,
                      [item["mean"], *item["ci98_333"]])
    i = read_json("experiments/fine-geometry-identification/evidence/analysis/summary.json")
    for label, item in zip(("P1 F−S", "P2 F−R"), i["primary"]):
        table_numbers(f"experiments/{FOCUS[1]}/REPORT_ZH.md", label,
                      [item["estimate"], *item["primary_ci"]])
    n = read_json("experiments/canonical-context-size-generalization/evidence/final_summary.json")
    ndoc = f"experiments/{FOCUS[2]}/REPORT_ZH.md"
    for arm in ("N", "W"):
        table_numbers(ndoc, arm + "平均KL", [sum(n["primary_arm_loss"][arm]) / 6])
    table_numbers(ndoc, "配对差", [n["primary"]["estimate"]])
    table_numbers(ndoc, "六seed配对t的95%区间", n["primary"]["ci"])
    rg = read_json("experiments/rg-coarse-data-training-transfer/evidence/analysis/summary.json")["summaries"]
    primary = rg["primary_fine_W96_KL"]
    arms = ("fine_only", "fine_plus_rg", "fine_replay")
    compare = primary["comparisons"]["fine_plus_rg_minus_fine_replay"]
    rdoc = f"experiments/{FOCUS[3]}/REPORT_ZH.md"
    for index, seed in enumerate((92601, 92602, 92603)):
        table_numbers(rdoc, str(seed),
                      [primary["arms"][a]["seed_means"][index] for a in arms] +
                      [compare["paired_seed_values"][index]])
    table_numbers(rdoc, "平均", [primary["arms"][a]["mean"] for a in arms] + [compare["mean"]])
    table_numbers("experiments/REPORT_ZH.md", "主细W96局部KL",
                  [compare["mean"], *compare["t95"], *compare["joint_seed_chain_block4_bootstrap95"]])

    # Frozen scientific sources and published evidence must remain byte-identical.
    for item in read_json("experiments/source-manifest.json"):
        target = ROOT / item["path"]
        if not target.is_file() or sha(target) != item["sha256"]:
            fail("frozen source hash mismatch", item["path"])
        hash_count += 1
    for provenance in sorted((ROOT / "experiments").rglob("provenance.json")):
        for item in json.loads(provenance.read_text(encoding="utf-8-sig"))["files"]:
            target = ROOT / "experiments" / item["published"]
            if (not target.is_file() or target.stat().st_size != item["bytes"] or
                    sha(target) != item["sha256"]):
                fail("published evidence hash mismatch", item["published"])
            hash_count += 1
    if args.index and SELF not in tracked:
        fail("checker not included in index", SELF)

    print(json.dumps({
        "passed": not errors, "reviewed_markdown_files": len(docs),
        "completed_research_units": sum(bool(r["formal_completed"]) for r in registry),
        "local_links_checked": link_count, "heading_anchors_checked": anchor_count,
        "primary_table_numbers_checked": numeric_count, "frozen_hashes_checked": hash_count,
        "index_checked": args.index, "base": args.base, "errors": errors,
        "limits": ["External URLs not requested; local public targets and selected anchors checked.",
                   "Selected main tables only; interpretation and other numbers require human review.",
                   "Pattern privacy checks are not proof of absence; inspect the staged payload.",
                   "No new scientific result, inference, resampling or experiment performed."],
    }, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
