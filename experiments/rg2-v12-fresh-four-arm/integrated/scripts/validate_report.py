"""Check the integrated report package without changing scientific evidence.

Run from the experiment root after rendering REPORT_ZH.html:
    python integrated/scripts/validate_report.py

Dependencies: beautifulsoup4, markdown-it-py, Pillow, pypdf.
The JSON audit is derived output. No network requests or model runs are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
MD = MarkdownIt("commonmark", {"html": True})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def without_fences(text: str) -> str:
    return re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", text)


def markdown_headings(text: str) -> list[str]:
    return re.findall(r"(?m)^##[ \t]+(.+?)\s*$", without_fences(text))


def github_ids(text: str) -> set[str]:
    counts: Counter[str] = Counter()
    result = set()
    for title in re.findall(r"(?m)^#{1,6}[ \t]+(.+?)\s*$", without_fences(text)):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        result.add(slug + suffix)
        counts[slug] += 1
    return result


def soup_for(path: Path) -> BeautifulSoup:
    text = path.read_text(encoding="utf-8-sig")
    return BeautifulSoup(MD.render(text) if path.suffix.lower() == ".md" else text, "html.parser")


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.errors: list[str] = []
        self.checks: dict = {}
        self.references: list[dict] = []
        self._anchor_cache: dict[Path, set[str]] = {}
        self._css_seen: set[Path] = set()

    def check(self, condition: bool, error: str):
        if not condition:
            self.errors.append(error)

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def hash_check(self, path: Path, expected: str, label: str) -> bool:
        if not path.is_file():
            self.errors.append(f"Missing hash target: {label}: {self.relative(path)}")
            return False
        ok = sha256(path) == expected
        self.check(ok, f"SHA mismatch: {label}: {self.relative(path)}")
        return ok

    def anchors(self, path: Path) -> set[str]:
        if path not in self._anchor_cache:
            soup = soup_for(path)
            ids = {str(x["id"]) for x in soup.find_all(id=True)}
            ids.update(str(x["name"]) for x in soup.find_all("a", attrs={"name": True}))
            if path.suffix.lower() == ".md":
                ids.update(github_ids(path.read_text(encoding="utf-8-sig")))
            self._anchor_cache[path] = ids
        return self._anchor_cache[path]

    def reference(self, owner: Path, url: str, attribute: str):
        if not url or url.isspace():
            self.errors.append(f"Empty {attribute}: {self.relative(owner)}")
            return
        parsed = urlsplit(url.strip())
        if parsed.scheme or parsed.netloc:
            return
        rawpath = unquote(parsed.path)
        target = (owner.parent / rawpath).resolve() if rawpath else owner.resolve()
        record = {"owner": self.relative(owner), "url": url, "attribute": attribute,
                  "target": self.relative(target), "exists": target.exists()}
        self.references.append(record)
        self.check(target.exists(), f"Broken local {attribute}: {self.relative(owner)} -> {url}")
        if not target.is_file():
            return
        if parsed.fragment and target.suffix.lower() in (".html", ".htm", ".md"):
            anchor = unquote(parsed.fragment)
            record["anchor_exists"] = anchor in self.anchors(target)
            self.check(record["anchor_exists"], f"Broken anchor: {self.relative(owner)} -> {url}")
        if target.suffix.lower() == ".css" and target not in self._css_seen:
            self._css_seen.add(target)
            css = target.read_text(encoding="utf-8-sig")
            for match in re.finditer(r"url\(\s*(['\"]?)(.*?)\1\s*\)", css):
                self.reference(target, match[2], "css-url")

    def document_links(self, path: Path) -> list[str]:
        soup = soup_for(path)
        images = []
        for node in soup.find_all(True):
            for attribute in ("href", "src", "poster", "data"):
                if node.has_attr(attribute):
                    self.reference(path, str(node[attribute]), attribute)
            if node.has_attr("srcset"):
                for item in str(node["srcset"]).split(","):
                    self.reference(path, item.strip().split()[0], "srcset")
            if node.name == "img" and node.has_attr("src"):
                images.append(str(node["src"]))
        return images

    def sections(self, titles: list[str], label: str):
        numbers = [int(m[1]) for t in titles if (m := re.match(r"^(\d+)\s*[.．、]\s*", t))]
        appendices = [m[1].upper() for t in titles if (m := re.match(r"^附录\s*([A-Da-d])(?:\b|\s|[：:])", t))]
        self.check(numbers == list(range(1, 13)), f"{label}: expected ordered sections 1–12, found {numbers}")
        self.check(appendices == list("ABCD"), f"{label}: expected appendices A–D, found {appendices}")
        return {"sections": numbers, "appendices": appendices, "h2_titles": titles}

    def reports(self, mdpath: Path, htmlpath: Path, markdown_only: bool = False):
        required = (mdpath, self.root / "integrated/FIGURES.md") if markdown_only else (mdpath, htmlpath, self.root / "integrated/FIGURES.md")
        for path in required:
            self.check(path.is_file(), f"Missing publication document: {self.relative(path)}")
        if not mdpath.is_file() or (not markdown_only and not htmlpath.is_file()):
            return
        mdtext = mdpath.read_text(encoding="utf-8-sig")
        self.check("@@" not in mdtext, "Unresolved @@ placeholder in Markdown")
        mdh = markdown_headings(mdtext)
        self.checks["markdown_structure"] = self.sections(mdh, "Markdown")
        mdimages = self.document_links(mdpath)
        index = self.root / "integrated/FIGURES.md"
        if index.is_file():
            self.document_links(index)
        if markdown_only:
            self.checks["report_images"] = {"markdown": len(mdimages)}
            self.checks["documents"] = {self.relative(p): sha256(p) for p in (mdpath, index) if p.is_file()}
            return
        htmltext = htmlpath.read_text(encoding="utf-8-sig")
        self.check("@@" not in htmltext, "Unresolved @@ placeholder in HTML")
        soup = soup_for(htmlpath)
        htmlh = [x.get_text(" ", strip=True) for x in soup.find_all("h2")]
        self.checks["html_structure"] = self.sections(htmlh, "HTML")
        htmlimages = self.document_links(htmlpath)
        self.check(Counter(mdimages) == Counter(htmlimages), "Markdown and HTML image references differ")
        self.checks["report_images"] = {"markdown": len(mdimages), "html": len(htmlimages)}
        error_nodes = soup.select(".katex-error")
        math_nodes = soup.select(".katex")
        mathml_nodes = soup.select(".katex-mathml")
        htmlmath_nodes = soup.select(".katex-html")
        self.check(not error_nodes, "Rendered HTML contains .katex-error")
        self.check(bool(math_nodes), "No prerendered KaTeX formulas found in HTML")
        self.check(len(math_nodes) == len(mathml_nodes) == len(htmlmath_nodes), "KaTeX MathML/HTML node counts disagree")
        for node in math_nodes:
            self.check(node.find("annotation", attrs={"encoding": "application/x-tex"}) is not None,
                       "KaTeX node has no source TeX annotation")
        self.checks["katex"] = {"rendered_formulas": len(math_nodes), "error_nodes": len(error_nodes),
                                "mathml_nodes": len(mathml_nodes), "html_nodes": len(htmlmath_nodes),
                                "mode": "static prerendered HTML verification"}
        build_path = self.root / "integrated/provenance/html_build.json"
        self.check(build_path.is_file(), "Missing HTML build provenance")
        if build_path.is_file():
            build = json_read(build_path)
            self.hash_check(mdpath, build["input_sha256"], "HTML authoritative Markdown")
            self.hash_check(htmlpath, build["output_sha256"], "built HTML output")
            self.hash_check(self.root / "integrated/scripts/build_html.py", build["build_script_sha256"], "HTML build script")
            article = soup.select_one(".report-article")
            article_math = len(article.select(".katex")) if article else 0
            self.check(article_math == build["formula_count"], "Article formula count differs from HTML build record")
            self.checks["katex"]["article_formula_count"] = article_math
            self.checks["katex"]["non_article_formula_count"] = len(math_nodes) - article_math
            vendor_path = self.root / "integrated/assets/katex/VENDOR.json"
            self.hash_check(vendor_path, build["katex_vendor_manifest_sha256"], "KaTeX vendor manifest")
            vendor = json_read(vendor_path)
            for rel, digest in vendor["files"].items():
                self.hash_check(vendor_path.parent / rel, digest, "vendored KaTeX asset")
            self.checks["katex"]["vendor_files_verified"] = len(vendor["files"])
        self.checks["documents"] = {self.relative(p): sha256(p) for p in (mdpath, htmlpath, index) if p.is_file()}

    def figure_package(self):
        directory = self.root / "integrated"
        physical = json_read(directory / "provenance/physical_figure_sources.json")
        method = json_read(directory / "provenance/method_figure_sources.json")
        review = json_read(directory / "provenance/physical_figure_review.json")
        self.hash_check(directory / "scripts/physical_figures.py", physical["script_sha256"], "physical script provenance")
        self.hash_check(self.root / method["script"], method["script_sha256"], "method script provenance")
        self.check(physical["script_sha256"] == review["script_sha256"], "Physical review belongs to a different script")
        self.check(physical["source_commit"] == method["source_commit"] == review["source_commit"], "Figure source commits differ")
        for rel, item in physical["inputs"].items():
            self.hash_check(self.root / rel, item["sha256"], "physical input")
        for rel, digest in method["source_sha256"].items():
            self.hash_check(self.root / rel, digest, "method source")
        self.hash_check(self.root / method["toy_data"], method["toy_data_sha256"], "explicit method example")
        expected = set(physical["figures"]) | set(method["figures"])
        self.check(len(expected) == 36, f"Expected 36 distinct figures, found {len(expected)}")
        pngs = {p.stem for p in (directory / "figures").glob("*.png")}
        pdfs = {p.stem for p in (directory / "figures").glob("*.pdf")}
        self.check(pngs == pdfs == expected, f"PNG/PDF/provenance stem sets differ: PNG-only={sorted(pngs-pdfs)}, PDF-only={sorted(pdfs-pngs)}, absent={sorted(expected-pngs)}")
        details = {}
        for stem in sorted(expected):
            png = directory / "figures" / f"{stem}.png"
            pdf = png.with_suffix(".pdf")
            if not png.is_file() or not pdf.is_file():
                continue
            if stem in method["figures"]:
                item = method["figures"][stem]
                self.hash_check(png, item["png_sha256"], stem + " PNG")
                self.hash_check(pdf, item["pdf_sha256"], stem + " PDF")
            else:
                item = review["files"].get(stem)
                self.check(item is not None, f"Physical figure has no reviewed output hashes: {stem}")
                if item:
                    self.hash_check(png, item["png"]["sha256"], stem + " PNG")
                    self.hash_check(pdf, item["pdf"]["sha256"], stem + " PDF")
            try:
                with Image.open(png) as image:
                    dimensions = list(image.size)
                    image.verify()
                with Image.open(png) as image:
                    image.load()
                document = PdfReader(pdf)
                self.check(len(document.pages) == 1, f"Expected one-page PDF: {stem}")
                box = document.pages[0].mediabox
                pdf_dimensions = [float(box.width), float(box.height)]
                ratio_difference = abs((dimensions[0] / dimensions[1]) / (pdf_dimensions[0] / pdf_dimensions[1]) - 1)
                self.check(ratio_difference < .01, f"PNG/PDF aspect ratio differs by >=1%: {stem}")
                details[stem] = {"png_dimensions": dimensions, "pdf_dimensions_pt": pdf_dimensions,
                                 "pdf_pages": len(document.pages), "aspect_ratio_relative_difference": ratio_difference,
                                 "png_sha256": sha256(png), "pdf_sha256": sha256(pdf)}
            except Exception as exc:
                self.errors.append(f"Unreadable figure pair {stem}: {type(exc).__name__}: {exc}")
        index = directory / "FIGURES.md"
        if index.is_file():
            text = index.read_text(encoding="utf-8-sig")
            for stem in sorted(expected):
                self.check(f"figures/{stem}.png" in text and f"figures/{stem}.pdf" in text,
                           f"Figure index lacks PNG/PDF pair: {stem}")
        self.checks["figure_package"] = {"physical_figures": len(physical["figures"]), "method_figures": len(method["figures"]),
                                         "physical_input_hashes": len(physical["inputs"]), "method_source_hashes": len(method["source_sha256"]),
                                         "figures": details}
        trace_path = directory / "provenance/physical_caption_traceability.json"
        if trace_path.is_file():
            trace = json_read(trace_path)
            self.hash_check(directory / "provenance/physical_figure_sources.json", trace["source_provenance_sha256"], "reviewed physical caption sources")
            report_path = self.root / "REPORT_ZH.md"
            if report_path.is_file():
                text = report_path.read_text(encoding="utf-8-sig")
                current = {}
                for match in re.finditer(r"!\[[^\]]*\]\(integrated/figures/([^)]*)\.png\)", text):
                    if match[1] in physical["figures"]:
                        caption = text[match.end():].lstrip().split("\n\n", 1)[0]
                        current[match[1]] = hashlib.sha256(caption.encode()).hexdigest()
                self.check(set(current) == set(trace["captions"]) == set(physical["figures"]), "Physical caption review is not complete")
                for stem, digest in current.items():
                    self.check(digest == trace["captions"][stem]["report_caption_sha256"], f"Caption changed after semantic review: {stem}")
            for rel, digest in trace["supporting_legacy_sample_identity_files"].items():
                self.hash_check(self.root / rel, digest, "legacy sample identity")
            self.checks["physical_caption_traceability"] = {"reviewed_figures": trace["figure_count"], "path": self.relative(trace_path), "sha256": sha256(trace_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--markdown", default="REPORT_ZH.md")
    parser.add_argument("--html", default="REPORT_ZH.html")
    parser.add_argument("--markdown-only", action="store_true", help="Check the finished Markdown while HTML is still being built")
    parser.add_argument("--output", default="integrated/provenance/report_validation.json")
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit(root)
    audit.reports(root / args.markdown, root / args.html, args.markdown_only)
    audit.figure_package()
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not audit.errors else "failed",
              "mode": "markdown-only" if args.markdown_only else "complete-publication",
              "validator_sha256": sha256(Path(__file__)), "checks": audit.checks,
              "local_reference_count": len(audit.references), "local_references": audit.references, "errors": audit.errors,
              "scope_limit": "Checks document structure, local resources, prerendered math markup and frozen input/output integrity; scientific claims and page layout require their separate audits."}
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "errors": audit.errors, "local_references": len(audit.references),
                      "output": str(destination)}, ensure_ascii=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    sys.exit(main())
