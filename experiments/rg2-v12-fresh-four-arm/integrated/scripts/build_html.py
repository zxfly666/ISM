#!/usr/bin/env python3
"""Build the offline report from REPORT_ZH.md and bundled KaTeX 0.16.22.

Requires Python-Markdown and Node.js. No network access or browser is used.
The final build reads only REPORT_ZH.md as scientific prose; --source and
--allow-draft exist solely for previewing an unfinished authoring draft.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import markdown
from markdown.extensions.toc import slugify_unicode


EXP = Path(__file__).resolve().parents[2]
ASSETS = EXP / "integrated/assets/katex"
TOKEN_PREFIX = "RG2MATHPLACEHOLDER"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(EXP).as_posix()
    except ValueError:
        # Preview provenance identifies an unpublished draft without exposing a
        # workstation-specific absolute path in eventual public build records.
        return "external-preview/" + path.name


def escaped(text: str, pos: int) -> bool:
    count = 0
    pos -= 1
    while pos >= 0 and text[pos] == "\\":
        count += 1
        pos -= 1
    return count % 2 == 1


def closing_delimiter(text: str, start: int, delimiter: str, single_line: bool) -> int:
    pos = start
    while True:
        end = text.find(delimiter, pos)
        if end < 0 or (single_line and "\n" in text[start:end]):
            return -1
        if not escaped(text, end):
            return end
        pos = end + len(delimiter)


def protect_math(source: str) -> tuple[str, list[dict]]:
    """Protect TeX, including table pipes, without parsing code as mathematics."""
    if TOKEN_PREFIX in source:
        raise ValueError("Reserved math placeholder already occurs in the source")
    pieces: list[str] = []
    formulas: list[dict] = []
    i = 0
    while i < len(source):
        # Fenced code is opaque, including literal dollar examples.
        if i == 0 or source[i - 1] == "\n":
            opening = re.match(r" {0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)", source[i:])
            if opening:
                fence = opening.group(1)
                start = i + opening.end()
                closing = re.search(
                    rf"(?m)^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*(?:\n|$)",
                    source[start:],
                )
                end = start + closing.end() if closing else len(source)
                pieces.append(source[i:end])
                i = end
                continue
        if source[i] == "`" and not escaped(source, i):
            ticks = re.match(r"`+", source[i:]).group(0)
            end = source.find(ticks, i + len(ticks))
            if end >= 0:
                end += len(ticks)
                pieces.append(source[i:end])
                i = end
                continue
        delimiter = ""
        closer = ""
        display = False
        if source[i] == "$" and not escaped(source, i):
            display = source.startswith("$$", i)
            delimiter = "$$" if display else "$"
            closer = delimiter
        elif source.startswith(r"\[", i) and not escaped(source, i):
            delimiter, closer, display = r"\[", r"\]", True
        elif source.startswith(r"\(", i) and not escaped(source, i):
            delimiter, closer = r"\(", r"\)"
        if delimiter:
            start = i + len(delimiter)
            end = closing_delimiter(source, start, closer, not display)
            if end < 0:
                raise ValueError(f"Unclosed math delimiter near source character {i}: {source[i:i+70]!r}")
            tex = source[start:end].strip()
            if not tex:
                raise ValueError(f"Empty math expression at source character {i}")
            token = f"{TOKEN_PREFIX}{len(formulas):06d}END"
            formulas.append({"token": token, "tex": tex, "display": display})
            pieces.append(f"\n\n{token}\n\n" if display else token)
            i = end + len(closer)
            continue
        pieces.append(source[i])
        i += 1
    return "".join(pieces), formulas


NODE_RENDER = r"""
const fs = require('fs');
const katex = require(process.argv[1]);
const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
const rendered = formulas.map((f, i) => {
  try {
    return katex.renderToString(f.tex, {
      displayMode: f.display, throwOnError: true, strict: 'error',
      trust: false, output: 'htmlAndMathml', maxExpand: 1000
    });
  } catch (err) {
    throw new Error(`Formula ${i}: ${f.tex}\n${err.message}`);
  }
});
process.stdout.write(JSON.stringify({version: katex.version, rendered}));
"""


def render_math(formulas: list[dict], node: str) -> tuple[list[str], str]:
    result = subprocess.run(
        [node, "-e", NODE_RENDER, str(ASSETS / "katex.min.js")],
        input=json.dumps(formulas, ensure_ascii=False),
        text=True, encoding="utf-8", capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("KaTeX rendering failed:\n" + result.stderr)
    payload = json.loads(result.stdout)
    if payload["version"] != "0.16.22":
        raise ValueError("Unexpected vendored KaTeX version: " + payload["version"])
    return payload["rendered"], payload["version"]


def restore_math(fragment: str, formulas: list[dict], rendered: list[str]) -> str:
    for formula, value in zip(formulas, rendered):
        token = formula["token"]
        if formula["display"]:
            block = '<div class="report-math-display">' + value + "</div>"
            paragraph = "<p>" + token + "</p>"
            if token in fragment and paragraph not in fragment:
                raise ValueError("Display formula was not isolated as a paragraph: " + formula["tex"])
            fragment = fragment.replace(paragraph, block)
        else:
            fragment = fragment.replace(token, '<span class="report-math-inline">' + value + "</span>")
    return fragment


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images: list[str] = []
        self.links: list[str] = []
        self.heading_counts = {str(i): 0 for i in range(1, 7)}
        self.ids: list[str] = []
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs.get("src", ""))
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_counts[tag[1:]] += 1
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "table":
            self.tables += 1


CSS = r"""
:root { color-scheme: light; --ink:#18242f; --muted:#536475; --line:#dce4ea;
  --paper:#fff; --wash:#f3f6f8; --accent:#145b80; }
html { scroll-behavior:smooth; scroll-padding-top:1.5rem; }
body { margin:0; color:var(--ink); background:var(--wash);
  font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',system-ui,sans-serif;
  font-size:16px; line-height:1.9; }
.report-shell { display:grid; grid-template-columns:260px minmax(0,1160px);
  gap:38px; max-width:1500px; padding:32px 34px 72px; margin:0 auto; }
.report-sidebar { min-width:0; }
.report-nav { position:sticky; top:24px; max-height:calc(100vh - 48px); overflow:auto;
  padding:16px 14px; border:1px solid var(--line); border-radius:10px; background:var(--paper);
  scrollbar-width:thin; font-size:13px; line-height:1.65; }
.report-nav summary { color:var(--accent); cursor:pointer; font-size:16px; font-weight:700; }
.report-nav ul { list-style:none; margin:10px 0 0; padding-left:0; }
.report-nav ul ul { padding-left:13px; margin-top:4px; border-left:1px solid var(--line); }
.report-nav li { margin:7px 0; }
.report-nav a { color:#405569; text-decoration:none; }
.report-nav a:hover { color:var(--accent); text-decoration:underline; }
.report-main { min-width:0; background:var(--paper); border:1px solid var(--line);
  border-radius:12px; padding:38px 48px 50px; box-shadow:0 4px 24px #12324906; }
.report-tools { display:flex; flex-wrap:wrap; gap:8px 20px; font-size:13px; color:var(--muted);
  border-bottom:1px solid var(--line); margin-bottom:22px; padding-bottom:14px; }
.report-article { overflow-wrap:anywhere; }
.report-article > h1 { font-size:clamp(25px,2.4vw,36px); line-height:1.5; margin:0 0 20px; letter-spacing:.015em; }
.report-article h2 { font-size:25px; line-height:1.55; margin:55px 0 20px;
  padding-top:24px; border-top:2px solid var(--line); }
.report-article h3 { font-size:20px; line-height:1.6; margin:34px 0 14px; }
.report-article h4 { font-size:17px; margin:27px 0 12px; }
.report-article > p { margin:15px 0; text-align:justify; }
.report-article a, .report-tools a { color:var(--accent); text-underline-offset:3px; }
.report-article a:hover, .report-tools a:hover { color:#9a5327; }
.report-article .headerlink { font-size:.65em; margin-left:9px; text-decoration:none; opacity:0; }
.report-article h2:hover .headerlink, .report-article h3:hover .headerlink,
.report-article .headerlink:focus { opacity:1; }
.report-figure { margin:28px -24px 12px; }
.report-figure img { width:100%; height:auto; display:block; background:white; }
.report-caption { font-size:14px; color:#405262; line-height:1.8; margin:12px 0 26px; }
.report-table-scroll { box-sizing:border-box; width:100%; max-width:100%; overflow-x:auto; margin:24px 0; border:1px solid var(--line);
  border-radius:7px; overscroll-behavior-x:contain; }
.report-table-scroll table { border-collapse:collapse; width:100%; font-size:14px; line-height:1.65; }
.report-table-scroll th, .report-table-scroll td { padding:11px 13px; vertical-align:top;
  text-align:left; border-bottom:1px solid var(--line); min-width:90px; }
.report-table-scroll th { background:#edf3f7; font-weight:700; color:#243f52; }
.report-table-scroll tr:last-child td { border-bottom:0; }
.report-table-scroll tbody tr:nth-child(even) { background:#fafcfd; }
.report-article > ul, .report-article > ol { margin:16px 0; padding-left:1.6em; }
.report-article li { margin:7px 0; }
.report-article blockquote { border-left:4px solid #9db9cc; padding:5px 20px;
  margin:22px 0; background:#f5f8fa; }
.report-article > pre { overflow-x:auto; padding:16px; border-radius:6px; background:#f1f4f6; font-size:13px; }
.report-article code { font-family:Consolas,'SFMono-Regular',monospace; font-size:.87em;
  background:#eef3f6; padding:.12em .28em; border-radius:3px; }
.report-article pre code { background:none; padding:0; }
.report-math-display { box-sizing:border-box; width:100%; max-width:100%; overflow-x:auto;
  overflow-y:hidden; padding:6px 3px; margin:20px 0; overscroll-behavior-x:contain; }
.report-math-display > .katex-display { width:max-content; min-width:100%; margin:.55em 0; }
.report-math-inline { display:inline-block; box-sizing:border-box; max-width:100%;
  overflow-x:auto; overflow-y:hidden; white-space:nowrap; vertical-align:middle;
  padding:2px 1px; overscroll-behavior-x:contain; }
.report-footer { margin-top:45px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
.report-draft { background:#fff3da; color:#6f4600; border:1px solid #e8ce92; border-radius:6px;
  margin:0 0 20px; padding:12px 16px; font-size:14px; }
@media (max-width:1160px) {
  .report-shell { grid-template-columns:220px minmax(0,1fr); gap:24px; padding:24px; }
  .report-main { padding:30px; }
  .report-figure { margin-left:-15px; margin-right:-15px; }
}
@media (max-width:850px) {
  .report-shell { display:block; padding:14px; }
  .report-nav { position:static; max-height:42vh; margin-bottom:16px; }
  .report-main { padding:24px 22px; border-radius:8px; }
  .report-article > h1 { font-size:26px; }
  .report-article h2 { font-size:23px; margin-top:42px; }
  .report-figure { margin-left:-12px; margin-right:-12px; }
}
@media (max-width:480px) {
  body { font-size:15px; }
  .report-shell { padding:8px; }
  .report-main { padding:22px 15px; }
  .report-article > h1 { font-size:23px; }
  .report-article h2 { font-size:21px; }
  .report-article h3 { font-size:18px; }
  .report-table-scroll table { font-size:13px; }
  .report-table-scroll th, .report-table-scroll td { padding:9px; }
  .report-figure { margin-left:-9px; margin-right:-9px; }
}
@media print {
  @page { size:A4; margin:16mm 14mm; }
  html { scroll-behavior:auto; }
  body { font-size:10pt; line-height:1.65; background:white; color:black; }
  .report-shell { display:block; padding:0; margin:0; max-width:none; }
  .report-sidebar, .report-tools, .headerlink { display:none !important; }
  .report-main { border:0; box-shadow:none; border-radius:0; padding:0; }
  .report-article > h1 { font-size:21pt; }
  .report-article h2 { font-size:16pt; margin-top:22pt; padding-top:12pt; }
  .report-article h3 { font-size:12pt; margin-top:16pt; }
  .report-article h1, .report-article h2, .report-article h3, .report-article h4 { break-after:avoid; }
  .report-article > p { orphans:3; widows:3; }
  .report-article a { color:inherit; text-decoration:none; }
  .report-figure { margin:16pt 0 6pt; break-inside:avoid; }
  .report-figure img { max-height:225mm; object-fit:contain; }
  .report-caption { font-size:9pt; break-before:avoid; }
  .report-table-scroll { width:100%; max-width:100%; overflow:visible; border-radius:0; }
  .report-table-scroll table { font-size:8pt; table-layout:fixed; }
  .report-table-scroll th, .report-table-scroll td { min-width:0; padding:5pt; overflow-wrap:anywhere; }
  .report-table-scroll thead { display:table-header-group; }
  .report-table-scroll tr { break-inside:avoid; }
  .report-math-display { overflow:visible; break-inside:avoid; font-size:8.5pt; padding:4pt 0; }
  .report-math-display > .katex-display { width:auto; min-width:0; }
  .report-math-display > .katex-display > .katex { white-space:normal; }
  .report-math-inline { display:inline; max-width:none; overflow:visible; white-space:normal;
    vertical-align:baseline; padding:0; }
  .report-footer { font-size:8pt; }
}
"""


def local_asset(reference: str, base: Path) -> Path | None:
    url = urlsplit(html.unescape(reference))
    if url.scheme or url.netloc or not url.path:
        return None
    return (base / unquote(url.path)).resolve()


def build(args) -> dict:
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    source = source_path.read_text(encoding="utf-8-sig")
    placeholders = re.findall(r"@@[^@\n]+@@", source)
    if placeholders and not args.allow_draft:
        raise ValueError("Unresolved authoring placeholders: " + ", ".join(placeholders))
    if not args.allow_draft and source_path != (EXP / "REPORT_ZH.md").resolve():
        raise ValueError("Final builds must use EXP/REPORT_ZH.md as their sole prose source")
    vendor_path = ASSETS / "VENDOR.json"
    vendor = json.loads(vendor_path.read_text(encoding="utf-8"))
    for relative, expected in vendor["files"].items():
        asset = ASSETS / relative
        if digest(asset) != expected:
            raise ValueError("Vendored KaTeX asset differs from its manifest: " + relative)
    protected, formulas = protect_math(source)
    node = args.node or shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required to render the local KaTeX assets")
    rendered, katex_version = render_math(formulas, node)
    token_tex = {f["token"]: f["tex"] for f in formulas}

    def slug(value, separator):
        for token, tex in token_tex.items():
            value = value.replace(token, tex)
        return slugify_unicode(value, separator)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"toc_depth": "2-3", "permalink": "¶", "slugify": slug}},
        output_format="html5",
    )
    body = restore_math(md.convert(protected), formulas, rendered)
    toc = restore_math(md.toc, formulas, rendered)
    body = re.sub(r"<p>(<img\b[^>]*>)</p>",
                  lambda m: '<figure class="report-figure">' + m.group(1).replace('<img ', '<img loading="lazy" decoding="async" ') + '</figure>',
                  body)
    body = re.sub(r"<p>(?=<strong>(?:附)?图)", '<p class="report-caption">', body)
    body = re.sub(r"(<table>.*?</table>)", r'<div class="report-table-scroll" role="region" aria-label="数据表，可横向滚动" tabindex="0">\1</div>', body, flags=re.S)
    inventory = Inventory()
    inventory.feed(body)
    if len(inventory.ids) != len(set(inventory.ids)):
        raise ValueError("Duplicate generated HTML element IDs")
    missing_images = [x for x in inventory.images if local_asset(x, EXP) is not None and not local_asset(x, EXP).is_file()]
    if missing_images and not args.allow_draft:
        raise ValueError("Missing local image files: " + ", ".join(missing_images))
    missing_local_links = [x for x in inventory.links if local_asset(x, EXP) is not None
                           and local_asset(x, EXP) != output_path and not local_asset(x, EXP).exists()]
    missing_fragments = [x for x in inventory.links if x.startswith("#") and unquote(x[1:]) not in inventory.ids]
    if missing_fragments:
        raise ValueError("Broken heading fragment links: " + ", ".join(missing_fragments))
    source_sha = digest(source_path)
    title_match = re.search(r"(?m)^#\s+(.+)$", source)
    title = title_match.group(1) if title_match else "ISM RG-2 整合研究报告"
    stylesheet = Path(os.path.relpath(ASSETS / "katex.min.css", output_path.parent)).as_posix()
    # Relative scientific image links retain the same meaning for the normal output.
    base_tag = ""
    if output_path.parent != EXP:
        relative_base = Path(os.path.relpath(EXP, output_path.parent)).as_posix().rstrip("/") + "/"
        base_tag = f'<base href="{html.escape(relative_base, quote=True)}">'
        stylesheet = "integrated/assets/katex/katex.min.css"
    draft_notice = '<div class="report-draft">编写中预览：正文尚未定稿，不能作为最终报告引用。</div>' if args.allow_draft else ""
    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="ISM RG-2 正式中文整合报告：真实物理统计、四臂机制设计、模型训练与跨尺度生成。">
<meta name="report-source-sha256" content="{source_sha}">
<title>{html.escape(title)}</title>
{base_tag}
<link rel="stylesheet" href="{html.escape(stylesheet, quote=True)}">
<style>{CSS}</style>
</head>
<body>
<div class="report-shell">
<aside class="report-sidebar"><nav class="report-nav" aria-label="报告目录"><details open><summary>报告目录</summary>{toc}</details></nav></aside>
<main class="report-main" id="report-main">
<div class="report-tools"><span>ISM RG-2 · 离线图文阅读版</span><a href="REPORT_ZH.md">Markdown 原文</a><a href="README.md">实验入口</a><span>支持浏览器打印</span></div>
{draft_notice}
<article class="report-article">{body}</article>
<footer class="report-footer">本页由同一 Markdown 正文生成。公式已离线排版，阅读时无需脚本或外部 CDN。图像与字体使用同目录内的相对路径。<br>正文 SHA-256：{source_sha}</footer>
</main>
</div>
</body>
</html>
'''
    if TOKEN_PREFIX in document:
        raise ValueError("A math placeholder escaped restoration")
    if '<script' in document.lower():
        raise ValueError("Offline report unexpectedly contains a script element")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8", newline="\n")
    provenance = {
        "status": "draft_preview" if args.allow_draft else "passed",
        "authoritative_input": "REPORT_ZH.md",
        "actual_input": portable_reference(source_path),
        "input_sha256": source_sha,
        "output": portable_reference(output_path),
        "output_sha256": digest(output_path),
        "build_script_sha256": digest(Path(__file__)),
        "python_markdown_version": markdown.__version__,
        "katex_version": katex_version,
        "katex_vendor_manifest_sha256": digest(vendor_path),
        "rendering": "build-time local KaTeX HTML+MathML; no client JavaScript; no CDN",
        "formula_count": len(formulas),
        "inline_formula_count": sum(not f["display"] for f in formulas),
        "display_formula_count": sum(f["display"] for f in formulas),
        "image_count": len(inventory.images),
        "unique_image_count": len(set(inventory.images)),
        "image_sources": inventory.images,
        "heading_counts": inventory.heading_counts,
        "section_count": inventory.heading_counts["2"],
        "numbered_chapter_count": len(re.findall(r"(?m)^##\s+\d+[.．、]\s*", source)),
        "appendix_count": len(re.findall(r"(?m)^##\s+附录\s*[A-Z]", source)),
        "table_count": inventory.tables,
        "unresolved_draft_placeholders": placeholders,
        "missing_images": missing_images,
        "missing_local_links": missing_local_links,
        "broken_fragments": missing_fragments,
        "asset_integrity": "all vendored KaTeX files match VENDOR.json",
        "css_static_review": {
            "inline_math": "bounded inline-block on screen with horizontal overflow; restored inline flow and permitted breakpoints for print",
            "display_math": "border-box scroll container on screen; content-width inner box prevents a centered negative overflow origin; print permits KaTeX breakpoints",
            "tables": "bounded horizontal scrolling on screen; fixed print table width, wrapping cells, repeated header rows and no row splitting where feasible",
            "scientific_body_edited_by_builder": False,
            "layout_verified_in_browser": False,
            "desktop_visual_verified": False,
            "mobile_visual_verified": False,
            "print_visual_verified": False
        },
        "content_equivalence": "The article body and table of contents are generated from the same decoded Markdown input; presentation wrappers add no scientific prose.",
        "review_limit": "Parsing, KaTeX rendering, source identity and container CSS were reviewed statically. No actual desktop, mobile or print browser visual validation was performed: the available browser rejected local-file access, and no alternate browser or indirect route was used. Native print wrapping cannot guarantee that every unbreakable TeX box fits the page; long formulas require source-level aligned rows when necessary."
    }
    evidence = EXP / "integrated/provenance/html_build.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EXP / "REPORT_ZH.md")
    parser.add_argument("--output", type=Path, default=EXP / "REPORT_ZH.html")
    parser.add_argument("--node", help="Optional absolute path to the Node.js executable")
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({k: result[k] for k in ["status", "input_sha256", "formula_count", "image_count", "section_count", "table_count", "missing_images", "missing_local_links"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
