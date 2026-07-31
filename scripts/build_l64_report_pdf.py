"""Build and audit the polished PDF version of the L=64 experiment report."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.md"
TMP = ROOT / "tmp" / "pdfs"
TEMP_MARKDOWN = TMP / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.pdf-source.md"
HTML = TMP / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.html"
RAW_PDF = TMP / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.raw.pdf"
FINAL_PDF = ROOT / "output" / "pdf" / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.pdf"
AUDIT_JSON = TMP / "pdf_audit.json"
RENDER_DIR = TMP / "render"
CONTACT_DIR = TMP / "contact_sheets"


CSS = r"""
@page {
  size: A4;
  margin: 17mm 17mm 21mm 17mm;
}

:root {
  --ink: #182433;
  --muted: #5c6b7a;
  --navy: #143f62;
  --blue: #176aa5;
  --cyan: #2b91b8;
  --line: #d4dde5;
  --soft: #eef4f7;
  --warm: #f8f5ed;
}

html {
  color: var(--ink);
  background: white;
  font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
  font-size: 10pt;
  line-height: 1.72;
  text-rendering: optimizeLegibility;
}

body {
  margin: 0;
  padding: 0;
  counter-reset: figure;
}

p {
  margin: 0.36em 0 0.72em;
  text-align: justify;
  text-justify: inter-ideograph;
  orphans: 3;
  widows: 3;
}

a {
  color: var(--blue);
  text-decoration: none;
}

strong {
  color: #102f49;
}

.cover {
  box-sizing: border-box;
  min-height: 252mm;
  margin: -4mm -1mm 0;
  padding: 18mm 10mm 8mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  break-after: page;
  page-break-after: always;
  position: relative;
  overflow: hidden;
}

.cover::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 14mm;
  height: 100%;
  background: linear-gradient(180deg, var(--navy), var(--cyan));
}

.cover::after {
  content: "";
  position: absolute;
  right: -32mm;
  top: -30mm;
  width: 112mm;
  height: 112mm;
  border: 16mm solid rgba(43, 145, 184, 0.10);
  border-radius: 50%;
}

.cover-kicker {
  margin: 0 0 8mm 12mm;
  color: var(--blue);
  font-size: 10pt;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.cover h1 {
  margin: 0 0 7mm 12mm;
  max-width: 145mm;
  color: var(--navy);
  font-size: 25pt;
  line-height: 1.28;
  font-weight: 760;
  letter-spacing: 0.01em;
}

.cover-subtitle {
  margin: 0 0 12mm 12mm;
  color: var(--muted);
  font-size: 12pt;
  text-align: left;
}

.cover-metrics {
  margin: 0 0 12mm 12mm;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 3mm;
}

.metric-card {
  box-sizing: border-box;
  min-height: 25mm;
  padding: 4mm 3.5mm;
  border-top: 2.2pt solid var(--cyan);
  background: var(--soft);
}

.metric-value {
  display: block;
  color: var(--navy);
  font-family: "Segoe UI", sans-serif;
  font-size: 15pt;
  font-weight: 750;
  line-height: 1.15;
}

.metric-label {
  display: block;
  margin-top: 1.6mm;
  color: var(--muted);
  font-size: 7.6pt;
  line-height: 1.35;
}

.cover-meta {
  margin: 0 0 0 12mm;
  padding: 4mm 5mm;
  border-left: 2.2pt solid var(--blue);
  background: #f7fafb;
  color: #405264;
  font-size: 8.6pt;
  line-height: 1.65;
  text-align: left;
  overflow-wrap: anywhere;
}

nav#TOC {
  break-after: page;
  page-break-after: always;
  padding-top: 3mm;
}

nav#TOC h2 {
  margin: 0 0 7mm;
  padding-bottom: 3mm;
  border-bottom: 2pt solid var(--blue);
  color: var(--navy);
  font-size: 20pt;
}

nav#TOC ul {
  margin: 0;
  padding-left: 0;
  list-style: none;
}

nav#TOC > ul > li {
  margin: 2.4mm 0;
  padding-bottom: 1.2mm;
  border-bottom: 0.45pt dotted #aebbc6;
  font-weight: 650;
}

nav#TOC ul ul {
  margin: 1.2mm 0 0 5mm;
  columns: 2;
  column-gap: 9mm;
}

nav#TOC ul ul li {
  margin: 0.8mm 0;
  color: var(--muted);
  font-size: 8.4pt;
  font-weight: 400;
  break-inside: avoid;
}

section.level2 {
  break-before: page;
  page-break-before: always;
}

section.level2.no-page-break {
  break-before: auto;
  page-break-before: auto;
}

h2 {
  margin: 0 0 6mm;
  padding: 0 0 2.6mm;
  border-bottom: 1.8pt solid var(--blue);
  color: var(--navy);
  font-size: 18pt;
  line-height: 1.3;
  font-weight: 750;
}

h3 {
  margin: 6.5mm 0 2.5mm;
  color: #17547e;
  font-size: 13pt;
  line-height: 1.35;
  font-weight: 720;
  break-after: avoid;
  page-break-after: avoid;
}

h4 {
  margin: 4mm 0 1.8mm;
  color: #26647f;
  font-size: 11pt;
  break-after: avoid;
}

hr {
  margin: 7mm 0;
  border: 0;
  border-top: 0.7pt solid var(--line);
}

ul, ol {
  margin: 0.7em 0 1em;
  padding-left: 1.65em;
}

li {
  margin: 0.2em 0;
  padding-left: 0.12em;
  orphans: 2;
  widows: 2;
}

blockquote {
  margin: 4mm 0;
  padding: 3mm 4.5mm;
  border-left: 2.4pt solid var(--cyan);
  background: var(--soft);
  color: #30475a;
}

.table-wrap {
  width: 100%;
  margin: 4mm 0 5mm;
}

table {
  width: 100%;
  border-collapse: collapse;
  border-top: 1.2pt solid var(--navy);
  border-bottom: 1.2pt solid var(--navy);
  color: #213243;
  font-size: 8.25pt;
  line-height: 1.42;
}

table.wide-table {
  font-size: 7.35pt;
  line-height: 1.34;
}

thead {
  display: table-header-group;
  background: #e7f0f5;
}

tr {
  break-inside: avoid;
  page-break-inside: avoid;
}

th, td {
  padding: 1.7mm 1.5mm;
  border-bottom: 0.45pt solid #cbd6df;
  vertical-align: top;
  overflow-wrap: anywhere;
}

th {
  color: var(--navy);
  font-weight: 720;
  text-align: left;
}

tbody tr:nth-child(even) {
  background: #f7fafb;
}

figure {
  margin: 5mm auto 6mm;
  text-align: center;
  break-inside: avoid;
  page-break-inside: avoid;
}

figure img {
  display: block;
  max-width: 100%;
  max-height: 208mm;
  width: auto;
  height: auto;
  margin: 0 auto;
  object-fit: contain;
}

figcaption {
  margin: 2.4mm auto 0;
  max-width: 94%;
  color: #4c5f70;
  font-size: 8.2pt;
  line-height: 1.45;
  text-align: center;
}

.math.display {
  display: block;
  margin: 3.2mm 0;
  overflow: visible;
  text-align: center;
  break-inside: avoid;
}

math {
  font-family: "Cambria Math", "STIX Two Math", serif;
}

code {
  font-family: "Consolas", "Cascadia Mono", monospace;
  font-size: 0.88em;
}

:not(pre) > code {
  padding: 0.08em 0.28em;
  border-radius: 2px;
  background: #edf2f5;
  color: #183e57;
  overflow-wrap: anywhere;
}

pre {
  margin: 3.5mm 0 5mm;
  padding: 3.8mm 4.2mm;
  border-left: 2.2pt solid #4e91b4;
  border-radius: 2px;
  background: #f3f6f8;
  color: #243746;
  font-size: 7.35pt;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  break-inside: auto;
}

.workflow-figure {
  margin: 5mm 0 6mm;
  padding: 4mm;
  border: 0.8pt solid var(--line);
  background: #fbfcfd;
  break-inside: avoid;
}

.workflow-title {
  margin: 0 0 3mm;
  color: var(--navy);
  font-size: 10.5pt;
  font-weight: 720;
}

.workflow-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2.2mm 4mm;
}

.flow-step {
  display: grid;
  grid-template-columns: 8mm 1fr;
  align-items: center;
  min-height: 12mm;
  padding: 2mm 2.6mm;
  border: 0.6pt solid #c6d5df;
  border-radius: 2px;
  background: white;
  color: #2c4152;
  font-size: 8.1pt;
  line-height: 1.35;
}

.flow-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 6mm;
  height: 6mm;
  border-radius: 50%;
  background: var(--blue);
  color: white;
  font-family: "Segoe UI", sans-serif;
  font-size: 7.5pt;
  font-weight: 700;
}

.workflow-caption {
  margin: 3mm 0 0;
  color: var(--muted);
  font-size: 8pt;
  text-align: center;
}

.callout {
  margin: 4mm 0;
  padding: 3mm 4mm;
  border: 0.7pt solid #c5d6df;
  background: var(--warm);
  break-inside: avoid;
}

@media print {
  a { color: inherit; }
  nav#TOC a { color: var(--blue); }
}
"""


WORKFLOW_STEPS = [
    "Wolff MC 生成完整 L=64 构型",
    "链级 train/val/test 划分与质量审计",
    "随机 MASK 的离散扩散训练",
    "按 validation NELBO 选择 EMA checkpoint",
    "修正 ancestral 采样的时间条件",
    "仅在 validation 上校准 24/48/96/128 steps",
    "冻结 128 steps 与 temperature=1.0",
    "3 个 sampling seeds 生成 4,608 张构型",
    "与新 8-chain/10k MC reference 比较",
    "bootstrap 置信区间、G(r)、S(k)与最终报告",
]


def find_pandoc() -> Path:
    candidates = [
        ROOT / "tmp" / "pdf-tools" / "pypandoc" / "files" / "pandoc.exe",
        ROOT / "tmp" / "pdf-tools" / "pypandoc" / "files" / "pandoc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    located = shutil.which("pandoc")
    if located:
        return Path(located)
    raise FileNotFoundError("Pandoc was not found in tmp/pdf-tools or PATH")


def sanitize_markdown(text: str) -> str:
    # Keep PDF punctuation deterministic and replace the Mermaid source with a
    # styled HTML placeholder that will become a native workflow diagram.
    for dash in ("\u2011", "\u2013", "\u2014"):
        text = text.replace(dash, "-")
    text, count = re.subn(
        r"```mermaid\s*.*?```",
        '<div id="workflow-placeholder"></div>',
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected one Mermaid block, replaced {count}")
    return text


def make_tag(soup: BeautifulSoup, name: str, text: str | None = None, **attrs):
    tag = soup.new_tag(name, **attrs)
    if text is not None:
        tag.string = text
    return tag


def add_workflow(soup: BeautifulSoup) -> None:
    placeholder = soup.find(id="workflow-placeholder")
    if placeholder is None:
        raise RuntimeError("Workflow placeholder not found in Pandoc HTML")
    wrapper = make_tag(soup, "div", attrs={"class": "workflow-figure"})
    wrapper.append(make_tag(soup, "div", "完整实验流程", attrs={"class": "workflow-title"}))
    grid = make_tag(soup, "div", attrs={"class": "workflow-grid"})
    for index, label in enumerate(WORKFLOW_STEPS, start=1):
        card = make_tag(soup, "div", attrs={"class": "flow-step"})
        card.append(make_tag(soup, "span", str(index), attrs={"class": "flow-number"}))
        card.append(make_tag(soup, "span", label, attrs={"class": "flow-label"}))
        grid.append(card)
    wrapper.append(grid)
    wrapper.append(
        make_tag(
            soup,
            "div",
            "图 0：从独立 MC 数据生成到冻结采样器和最终物理评估的完整链路。",
            attrs={"class": "workflow-caption"},
        )
    )
    placeholder.replace_with(wrapper)


def add_cover_and_toc(soup: BeautifulSoup) -> None:
    body = soup.body
    if body is None:
        raise RuntimeError("Pandoc HTML has no body")
    title = soup.find("h1")
    if title is None:
        raise RuntimeError("Report title was not found")
    metadata = title.find_next_sibling("p")

    nav = soup.find("nav", id="TOC")
    if nav is not None:
        nav.extract()

    cover = make_tag(soup, "section", attrs={"class": "cover"})
    cover.append(make_tag(soup, "div", "ISM / GENERATIVE PHYSICS", attrs={"class": "cover-kicker"}))
    title.extract()
    title["class"] = ["cover-title"]
    cover.append(title)
    cover.append(
        make_tag(
            soup,
            "p",
            "正式训练 · 采样器校准 · 独立 MC 物理评估",
            attrs={"class": "cover-subtitle"},
        )
    )

    metrics = make_tag(soup, "div", attrs={"class": "cover-metrics"})
    for value, label in [
        ("64 x 64", "完整周期晶格"),
        ("60,000", "训练优化 steps"),
        ("0.361248", "最佳验证 NELBO"),
        ("4,608 / 10k", "模型样本 / MC reference"),
    ]:
        card = make_tag(soup, "div", attrs={"class": "metric-card"})
        card.append(make_tag(soup, "span", value, attrs={"class": "metric-value"}))
        card.append(make_tag(soup, "span", label, attrs={"class": "metric-label"}))
        metrics.append(card)
    cover.append(metrics)

    if metadata is not None:
        metadata.extract()
        metadata["class"] = ["cover-meta"]
        cover.append(metadata)
    else:
        cover.append(make_tag(soup, "p", "版本：2026-07-31", attrs={"class": "cover-meta"}))

    body.insert(0, cover)
    if nav is not None:
        cover.insert_after(nav)


def postprocess_html() -> None:
    soup = BeautifulSoup(HTML.read_text(encoding="utf-8"), "html.parser")
    soup.html["lang"] = "zh-CN"
    title_node = soup.find("title")
    if title_node is not None:
        title_node.string = "L=64 临界二维 Ising 离散扩散 Pilot 实验报告"
    style = make_tag(soup, "style")
    style.string = CSS
    soup.head.append(style)

    add_workflow(soup)
    add_cover_and_toc(soup)

    # Section 1 can end with a short continuation. Let section 2 follow it on
    # the same page so the report does not create an almost-empty page.
    for section in soup.select("section.level2"):
        heading = section.find("h2", recursive=False)
        if heading is not None and heading.get_text(" ", strip=True).startswith("2. "):
            section["class"] = list(section.get("class", [])) + ["no-page-break"]

    for table in soup.find_all("table"):
        columns = len(table.find("tr").find_all(["th", "td"])) if table.find("tr") else 0
        if columns >= 6:
            table["class"] = list(table.get("class", [])) + ["wide-table"]
        wrapper = make_tag(soup, "div", attrs={"class": "table-wrap"})
        table.wrap(wrapper)

    HTML.write_text(str(soup), encoding="utf-8", newline="\n")


def prepare() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    clean_text = sanitize_markdown(SOURCE.read_text(encoding="utf-8"))
    TEMP_MARKDOWN.write_text(clean_text, encoding="utf-8", newline="\n")
    pandoc = find_pandoc()
    resource_path = os.pathsep.join([str(ROOT / "docs"), str(ROOT)])
    command = [
        str(pandoc),
        str(TEMP_MARKDOWN),
        "--from=markdown+pipe_tables+fenced_code_blocks+tex_math_dollars+tex_math_single_backslash+raw_html",
        "--to=html5",
        "--standalone",
        "--mathml",
        "--toc",
        "--toc-depth=2",
        "--section-divs",
        "--embed-resources",
        f"--resource-path={resource_path}",
        "--metadata=lang:zh-CN",
        "--metadata=pagetitle:L=64 临界二维 Ising 离散扩散 Pilot 实验报告",
        "--metadata=toc-title:目录",
        "--highlight-style=tango",
        f"--output={HTML}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    postprocess_html()
    print(json.dumps({"html": str(HTML), "bytes": HTML.stat().st_size}, ensure_ascii=False))


def finalize() -> None:
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen.canvas import Canvas

    if not RAW_PDF.exists():
        raise FileNotFoundError(f"Raw browser PDF is missing: {RAW_PDF}")
    reader = PdfReader(str(RAW_PDF))
    writer = PdfWriter()
    total_content_pages = max(1, len(reader.pages) - 1)

    for page_index, page in enumerate(reader.pages):
        if page_index > 0:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            packet = BytesIO()
            canvas = Canvas(packet, pagesize=(width, height))
            canvas.setStrokeColor(HexColor("#c7d2dc"))
            canvas.setLineWidth(0.45)
            canvas.line(48, 35, width - 48, 35)
            canvas.setFillColor(HexColor("#627384"))
            canvas.setFont("Helvetica", 7.4)
            canvas.drawString(48, 22, "L=64 Ising Discrete Diffusion Pilot")
            canvas.drawRightString(
                width - 48,
                22,
                f"{page_index} / {total_content_pages}",
            )
            canvas.save()
            packet.seek(0)
            overlay = PdfReader(packet).pages[0]
            page.merge_page(overlay)
        writer.add_page(page)

    writer.add_metadata(
        {
            "/Title": "L=64 Critical 2D Ising Discrete Diffusion Pilot Report",
            "/Author": "ISM experiment archive",
            "/Subject": "Training, sampler calibration, and independent physics evaluation",
            "/Keywords": "Ising, discrete diffusion, criticality, Monte Carlo, correlation function",
        }
    )
    with FINAL_PDF.open("wb") as handle:
        writer.write(handle)
    print(
        json.dumps(
            {"pdf": str(FINAL_PDF), "pages": len(reader.pages), "bytes": FINAL_PDF.stat().st_size},
            ensure_ascii=False,
        )
    )


def audit() -> None:
    import pdfplumber

    if not FINAL_PDF.exists():
        raise FileNotFoundError(FINAL_PDF)
    pages = []
    full_text_parts = []
    with pdfplumber.open(FINAL_PDF) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            full_text_parts.append(text)
            out_of_bounds = 0
            for char in page.chars:
                if (
                    float(char.get("x0", 0)) < -2
                    or float(char.get("x1", 0)) > page.width + 2
                    or float(char.get("top", 0)) < -2
                    or float(char.get("bottom", 0)) > page.height + 2
                ):
                    out_of_bounds += 1
            pages.append(
                {
                    "page": page_number,
                    "characters": len(text),
                    "images": len(page.images),
                    "out_of_bounds_characters": out_of_bounds,
                }
            )
    full_text = "\n".join(full_text_parts)
    required_terms = [
        "正式训练",
        "0.361248",
        "采样器",
        "两点相关函数",
        "最终结论",
        "审稿式自检",
    ]
    result = {
        "pdf": str(FINAL_PDF),
        "bytes": FINAL_PDF.stat().st_size,
        "page_count": len(pages),
        "text_characters": len(full_text),
        "replacement_character_count": full_text.count("\ufffd"),
        "required_terms": {term: term in full_text for term in required_terms},
        "blank_or_nearly_blank_pages": [p["page"] for p in pages if p["characters"] < 20],
        "out_of_bounds_characters": sum(p["out_of_bounds_characters"] for p in pages),
        "pages": pages,
    }
    AUDIT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def contact_sheets() -> None:
    from PIL import Image, ImageDraw, ImageOps

    files = sorted(
        RENDER_DIR.glob("page-*.png"),
        key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No rendered pages found in {RENDER_DIR}")
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    for old in CONTACT_DIR.glob("contact-*.png"):
        old.unlink()

    columns = 4
    rows = 2
    cell_width = 380
    cell_height = 545
    margin = 18
    per_sheet = columns * rows
    for sheet_index in range(0, len(files), per_sheet):
        batch = files[sheet_index : sheet_index + per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * cell_width + 2 * margin, rows * cell_height + 2 * margin),
            "#dfe5e9",
        )
        draw = ImageDraw.Draw(canvas)
        for offset, path in enumerate(batch):
            with Image.open(path) as page:
                page_rgb = page.convert("RGB")
                thumb = ImageOps.contain(page_rgb, (cell_width - 28, cell_height - 42))
            x = margin + (offset % columns) * cell_width + (cell_width - thumb.width) // 2
            y = margin + (offset // columns) * cell_height + 24
            canvas.paste(thumb, (x, y))
            draw.rectangle((x - 1, y - 1, x + thumb.width, y + thumb.height), outline="#8897a3")
            page_number = int(re.search(r"(\d+)$", path.stem).group(1))
            draw.text((x, y - 17), f"page {page_number}", fill="#253847")
        output = CONTACT_DIR / f"contact-{sheet_index // per_sheet + 1:02d}.png"
        canvas.save(output, optimize=True)
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "finalize", "audit", "contact"])
    args = parser.parse_args()
    {
        "prepare": prepare,
        "finalize": finalize,
        "audit": audit,
        "contact": contact_sheets,
    }[args.command]()


if __name__ == "__main__":
    main()
