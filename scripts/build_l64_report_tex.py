"""Generate a self-contained Chinese XeLaTeX source tree for the L=64 report."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.md"
TMP = ROOT / "tmp" / "pdfs"
TEMP_MARKDOWN = TMP / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.tex-source.md"
TEMP_HEADER = TMP / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.header.tex"
OUTPUT_DIR = ROOT / "output" / "tex"
FIGURES_DIR = OUTPUT_DIR / "figures"
TEX = OUTPUT_DIR / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.tex"
PDF = OUTPUT_DIR / "L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.pdf"


FIGURE_NAMES = [
    "01_training_curves.png",
    "02_sampler_convergence.png",
    "03_random_samples.png",
    "04_magnetization_matched_samples.png",
    "05_ensemble_distributions.png",
    "07_scalar_observables.png",
    "06_correlation_and_structure.png",
    "08_seed_stability.png",
]


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


LATEX_HEADER = r"""
% Chinese report styling added by scripts/build_l64_report_tex.py.
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{float}
\usepackage{placeins}
\usepackage{xurl}
\usepackage{etoolbox}
\usepackage{setspace}
\usepackage{array}
\usepackage{fvextra}

\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}[Scale=MatchLowercase]
\setCJKmainfont{SimSun}[AutoFakeBold=2.2]
\setCJKsansfont{Microsoft YaHei}
\setCJKmonofont{Microsoft YaHei}

\definecolor{ISMBlue}{HTML}{174F78}
\definecolor{ISMCyan}{HTML}{278EB5}
\definecolor{ISMDark}{HTML}{182A3A}
\definecolor{ISMMuted}{HTML}{607180}
\definecolor{ISMLight}{HTML}{EAF2F6}

\setstretch{1.28}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.28em}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.22}
\fvset{breaklines=true,breakanywhere=true,fontsize=\small}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{3pt}}

\titleformat{\section}
  {\Large\sffamily\bfseries\color{ISMBlue}}
  {}{0pt}{}
  [\vspace{0.2em}\color{ISMCyan}\titlerule]
\titleformat{\subsection}
  {\large\sffamily\bfseries\color{ISMBlue}}
  {}{0pt}{}
\titleformat{\subsubsection}
  {\normalsize\sffamily\bfseries\color{ISMCyan}}
  {}{0pt}{}
\titlespacing*{\section}{0pt}{1.8em}{0.9em}
\titlespacing*{\subsection}{0pt}{1.25em}{0.45em}
\titlespacing*{\subsubsection}{0pt}{1em}{0.35em}

\captionsetup{
  font=small,
  labelfont={bf,color=ISMBlue},
  textfont={color=ISMMuted},
  justification=centering,
  singlelinecheck=true,
  skip=6pt
}
\floatplacement{figure}{H}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily\color{ISMMuted}L=64 Ising 离散扩散 Pilot}
\fancyhead[R]{\small\sffamily\color{ISMMuted}实验技术报告}
\fancyfoot[C]{\small\sffamily\color{ISMMuted}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\setlength{\headheight}{15pt}

\AtBeginDocument{%
  \hypersetup{
    linkcolor=ISMBlue,
    urlcolor=ISMCyan,
    citecolor=ISMBlue,
    pdfauthor={ISM experiment archive},
    pdftitle={L=64 Critical 2D Ising Discrete Diffusion Pilot Report},
    pdfsubject={Training, sampler calibration, and independent physics evaluation}
  }%
}

\makeatletter
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \thispagestyle{empty}
    \vspace*{13mm}
    {\color{ISMBlue}\rule{8pt}{218mm}}\hspace{10mm}%
    \begin{minipage}[b][218mm][c]{0.80\textwidth}
      {\sffamily\bfseries\color{ISMCyan}\large ISM / GENERATIVE PHYSICS\par}
      \vspace{15mm}
      {\sffamily\bfseries\color{ISMBlue}\fontsize{26}{34}\selectfont\@title\par}
      \vspace{6mm}
      {\sffamily\color{ISMMuted}\large 正式训练 · 采样器校准 · 独立 MC 物理评估\par}
      \vspace{14mm}
      \renewcommand{\arraystretch}{1.35}
      \begin{tabular}{>{\centering\arraybackslash}p{0.21\linewidth}
                      >{\centering\arraybackslash}p{0.21\linewidth}
                      >{\centering\arraybackslash}p{0.24\linewidth}
                      >{\centering\arraybackslash}p{0.24\linewidth}}
        \hline
        \textbf{\large 64 x 64} & \textbf{\large 60,000} &
        \textbf{\large 0.361248} & \textbf{\large 4,608 / 10k} \\
        完整周期晶格 & 训练优化 steps & 最佳验证 NELBO & 模型样本 / MC reference \\
        \hline
      \end{tabular}
      \vspace{16mm}
      {\color{ISMCyan}\rule{\linewidth}{1.2pt}\par}
      \vspace{4mm}
      {\small\color{ISMMuted}
        版本：2026-07-31\par
        实验状态：正式训练、采样器校准与独立最终评估均已完成\par
        工程归档：\texttt{C:/Users/zhangxiangfei/Desktop/ISM}\par
      }
      \vfill
      {\sffamily\color{ISMMuted}\small 中文 XeLaTeX 可复现版本\par}
    \end{minipage}
  \end{titlepage}
  \pagenumbering{arabic}
}
\makeatother
"""


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
    raise FileNotFoundError("Pandoc was not found")


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    chosen_font: ImageFont.FreeTypeFont,
    fill: str,
    width: int,
) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_long_words=True))
    bounds = draw.multiline_textbbox((0, 0), wrapped, font=chosen_font, spacing=10, align="center")
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - text_width) / 2
    y = box[1] + (box[3] - box[1] - text_height) / 2 - bounds[1]
    draw.multiline_text((x, y), wrapped, font=chosen_font, fill=fill, spacing=10, align="center")


def build_workflow() -> Path:
    width, height = 2400, 1520
    image = Image.new("RGB", (width, height), "#F8FBFC")
    draw = ImageDraw.Draw(image)
    regular = r"C:\Windows\Fonts\msyh.ttc"
    bold = r"C:\Windows\Fonts\msyhbd.ttc"
    title_font = font(bold, 66)
    card_font = font(regular, 36)
    number_font = font(bold, 34)
    caption_font = font(regular, 29)

    draw.rectangle((0, 0, 34, height), fill="#174F78")
    draw.text((105, 70), "完整实验流程", font=title_font, fill="#174F78")
    draw.line((105, 160, width - 100, 160), fill="#278EB5", width=7)

    left = 105
    top = 205
    card_width = 1035
    card_height = 205
    column_gap = 70
    row_gap = 28
    for index, label in enumerate(WORKFLOW_STEPS, start=1):
        column = (index - 1) % 2
        row = (index - 1) // 2
        x0 = left + column * (card_width + column_gap)
        y0 = top + row * (card_height + row_gap)
        x1 = x0 + card_width
        y1 = y0 + card_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill="white", outline="#B8CBD7", width=4)
        circle_center = (x0 + 82, y0 + card_height // 2)
        radius = 42
        draw.ellipse(
            (
                circle_center[0] - radius,
                circle_center[1] - radius,
                circle_center[0] + radius,
                circle_center[1] + radius,
            ),
            fill="#176DA4",
        )
        number_box = (
            circle_center[0] - radius,
            circle_center[1] - radius,
            circle_center[0] + radius,
            circle_center[1] + radius,
        )
        draw_centered_multiline(draw, number_box, str(index), number_font, "white", width=3)
        text_box = (x0 + 145, y0 + 18, x1 - 28, y1 - 18)
        draw_centered_multiline(draw, text_box, label, card_font, "#243B4C", width=25)

    caption = "从独立 MC 数据生成到冻结采样器和最终物理评估的完整链路"
    draw.text((105, height - 78), caption, font=caption_font, fill="#607180")
    output = FIGURES_DIR / "00_experiment_workflow.png"
    image.save(output, dpi=(300, 300), optimize=True)
    return output


def copy_figures() -> None:
    source_dir = ROOT / "artifacts" / "final_l64" / "figures"
    for name in FIGURE_NAMES:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, FIGURES_DIR / name)


def sanitize_markdown(text: str) -> str:
    for dash in ("\u2011", "\u2013", "\u2014"):
        text = text.replace(dash, "-")
    text = text.replace("\u2212", "-")

    first_section = text.find("## 报告结构")
    if first_section < 0:
        raise RuntimeError("The first report section was not found")
    text = text[first_section:]

    workflow_tex = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/00_experiment_workflow.png}
\caption*{实验流程示意图：从独立 MC 数据生成到冻结采样器和最终物理评估。}
\end{figure}
"""
    text, count = re.subn(
        r"```mermaid\s*.*?```",
        lambda _: workflow_tex,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected one Mermaid block, replaced {count}")

    text = re.sub(
        r"!\[图\d+[：:]\s*([^]]+)\]\(\.\./artifacts/final_l64/figures/([^)]+)\)",
        r"![\1](figures/\2)",
        text,
    )
    if "../artifacts/final_l64/figures/" in text:
        raise RuntimeError("At least one report figure path was not rewritten")

    metadata = """---
title: '$L=64$ 临界二维 Ising 离散扩散 Pilot 实验报告'
date: '2026-07-31'
lang: zh-CN
---

"""
    return metadata + text


def postprocess_tex() -> None:
    text = TEX.read_text(encoding="utf-8")
    text = text.replace("\\tableofcontents", "\\tableofcontents\n\\clearpage", 1)
    # Keep figures and tables from crossing into the next major section.
    text = text.replace("\\section{", "\\FloatBarrier\n\\section{")
    # Pandoc may emit Unicode dashes in automatically generated text.
    for dash in ("\u2011", "\u2013", "\u2014"):
        text = text.replace(dash, "-")
    TEX.write_text(text, encoding="utf-8", newline="\n")


def prepare() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    build_workflow()
    copy_figures()

    markdown = sanitize_markdown(SOURCE.read_text(encoding="utf-8"))
    TEMP_MARKDOWN.write_text(markdown, encoding="utf-8", newline="\n")
    TEMP_HEADER.write_text(LATEX_HEADER, encoding="utf-8", newline="\n")

    pandoc = find_pandoc()
    command = [
        str(pandoc),
        str(TEMP_MARKDOWN),
        "--from=markdown+pipe_tables+fenced_code_blocks+tex_math_dollars+tex_math_single_backslash+raw_tex",
        "--to=latex",
        "--standalone",
        "--shift-heading-level-by=-1",
        "--toc",
        "--toc-depth=1",
        "--top-level-division=section",
        "--metadata=toc-title:目录",
        "--variable=documentclass:ctexart",
        "--variable=classoption:UTF8",
        "--variable=classoption:zihao=5",
        "--variable=papersize:a4",
        "--variable=geometry:top=22mm,bottom=23mm,left=21mm,right=21mm",
        f"--include-in-header={TEMP_HEADER}",
        f"--output={TEX}",
    ]
    subprocess.run(command, cwd=OUTPUT_DIR, check=True)
    postprocess_tex()
    print(
        json.dumps(
            {
                "tex": str(TEX),
                "tex_bytes": TEX.stat().st_size,
                "figures": len(list(FIGURES_DIR.glob("*.png"))),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()


if __name__ == "__main__":
    main()
