"""Build LungLens_OnePager.pdf: a strictly one-page project description.

The prose comes from docs/one_pager.md; every number comes from
reports/metrics.json (and reports/splits.csv), so the PDF can never quote a
number that a real run did not produce. The script FAILS if the result is not
exactly one page.

Usage:
    python scripts/make_one_pager.py
"""

import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils import load_json  # noqa: E402

OUT = ROOT / "LungLens_OnePager.pdf"
INK, INK_2, BLUE, RULE = "#0b0b0b", "#52514e", "#2a78d6", "#e1e0d9"


def register_fonts() -> None:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("Sans", "DejaVuSans.ttf"), ("SansBold", "DejaVuSans-Bold.ttf"),
                       ("SansOblique", "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="SansOblique")


def results_rows(metrics: dict) -> list[list[str]]:
    op = metrics["test"]["at_operating_threshold"]
    m, ci = op["metrics"], op["ci95"]
    rows = [["Metric", "Value", "95% CI"]]
    for key, label in (("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("precision", "Precision"),
                       ("f1", "F1"), ("accuracy", "Accuracy")):
        rows.append([label, f"{m[key]:.3f}", f"{ci[key][0]:.3f}-{ci[key][1]:.3f}"])
    rows.append(["AUROC", f"{metrics['test']['auroc']:.3f}", f"{ci['auroc'][0]:.3f}-{ci['auroc'][1]:.3f}"])
    return rows


def results_sentences(metrics: dict) -> str:
    cal, d, op = metrics["calibration"], metrics["decision"], metrics["operating_point"]
    c = metrics["test"]["at_operating_threshold"]["confusion"]
    text = (
        f"Threshold {op['threshold']:.2f}, chosen on validation data. It missed {c['fn']} of {c['fn'] + c['tp']} "
        f"pneumonia X-rays but flagged {c['fp']} of {c['fp'] + c['tn']} normal ones: specificity is the weak spot "
        f"(validation: {op['val_metrics']['specificity']:.3f}). Calibration error (ECE): validation "
        f"{cal['val_ece_before']:.3f} to {cal['val_ece_after']:.3f}, test {cal['test_ece_before']:.3f} to "
        f"{cal['test_ece_after']:.3f} after temperature scaling. {100 * d['test_inconclusive_rate']:.1f}% of test "
        "X-rays fall in the Inconclusive zone."
    )
    ood = ROOT / "reports" / "ood_eval.json"
    if ood.exists():
        parts = [f"{name.split(' (')[0]} {100 * r['flagged_rate']:.1f}%" for name, r in load_json(ood).items()]
        text += " Input check, share of images flagged: " + ", ".join(parts) + "."
    return text


def markdown_to_flowables(md: str, styles: dict, metrics: dict, n_train: int) -> list:
    """A tiny Markdown subset: '#', '##', paragraphs, '- ' bullets, **bold**."""
    md = md.replace("{{N_TRAIN}}", f"{n_train:,}").replace("{{N_TEST}}", f"{metrics['data']['n_test']:,}")
    flow = []

    def inline(text):
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    for block in md.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("# "):
            continue  # the title is drawn in the header band
        if block.startswith("## "):
            flow.append(Paragraph(inline(block[3:]), styles["h2"]))
        elif block == "{{RESULTS}}":
            table = Table(results_rows(metrics), colWidths=[30 * mm, 18 * mm, 28 * mm], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), "Sans", 8),
                ("FONT", (0, 0), (-1, 0), "SansBold", 8),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(BLUE)),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]))
            # Keep the heading, table and notes in one column so the notes never start a column on their own.
            heading = [flow.pop()] if flow and isinstance(flow[-1], Paragraph) else []
            flow.append(KeepTogether(heading + [table, Spacer(1, 2 * mm),
                                                Paragraph(results_sentences(metrics), styles["small"])]))
        elif all(line.startswith("- ") for line in block.splitlines()):
            for line in block.splitlines():
                flow.append(Paragraph(inline(line[2:]), styles["bullet"], bulletText="•"))
        elif block.startswith("UnivaBio"):  # subtitle line under the title
            flow.append(Paragraph(inline(block), styles["subtitle"]))
        else:
            flow.append(Paragraph(inline(" ".join(block.splitlines())), styles["body"]))
    return flow


def build() -> int:
    register_fonts()
    metrics = load_json(ROOT / "reports" / "metrics.json")
    splits = pd.read_csv(ROOT / "reports" / "splits.csv")
    n_train = int((splits["split"] == "train").sum())

    styles = {
        "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=8.3, leading=11, textColor=colors.HexColor(INK_2),
                                   spaceAfter=3),
        "h2": ParagraphStyle("h2", fontName="SansBold", fontSize=10, leading=13, textColor=colors.HexColor(BLUE),
                             spaceBefore=5, spaceAfter=2),
        "body": ParagraphStyle("body", fontName="Sans", fontSize=8.3, leading=11.2, textColor=colors.HexColor(INK)),
        "small": ParagraphStyle("small", fontName="Sans", fontSize=7.6, leading=10, textColor=colors.HexColor(INK_2)),
        "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=8.3, leading=11, leftIndent=9, bulletIndent=1,
                                 textColor=colors.HexColor(INK), spaceAfter=1),
    }
    width, height = A4
    margin, gap, header_h = 13 * mm, 7 * mm, 22 * mm
    col_w = (width - 2 * margin - gap) / 2
    frame_h = height - 2 * margin - header_h - 4 * mm
    frames = [Frame(margin, margin, col_w, frame_h, id="left", leftPadding=0, rightPadding=0, topPadding=0),
              Frame(margin + col_w + gap, margin, col_w, frame_h, id="right", leftPadding=0, rightPadding=0, topPadding=0)]

    def header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(BLUE))
        canvas.rect(0, height - margin - header_h + 4 * mm, width, header_h + margin, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("SansBold", 21)
        canvas.drawString(margin, height - margin - 6 * mm, "LungLens: See what the AI sees")
        canvas.setFont("Sans", 9.5)
        canvas.drawString(margin, height - margin - 12.5 * mm,
                          "Explainable chest X-ray screening for earlier pneumonia detection")
        canvas.setFillColor(colors.HexColor(INK_2))
        canvas.setFont("Sans", 6.8)
        canvas.drawString(margin, 7 * mm, "Educational screening aid, not a diagnostic device. All numbers are "
                          "generated from reports/metrics.json by scripts/make_one_pager.py.")
        canvas.restoreState()

    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=margin,
                          bottomMargin=margin, title="LungLens one-pager", author="Ribhu")
    doc.addPageTemplates([PageTemplate(id="page", frames=frames, onPage=header)])

    story = markdown_to_flowables((ROOT / "docs" / "one_pager.md").read_text(encoding="utf-8"), styles, metrics, n_train)
    example = ROOT / "reports" / "figures" / "onepager_example.png"
    if example.exists():
        story += [Spacer(1, 3 * mm), Image(str(example), width=col_w, height=col_w * 0.5)]
    doc.build(story)
    return len(PdfReader(OUT).pages)


if __name__ == "__main__":
    pages = build()
    print(f"wrote {OUT.name}: {pages} page(s)")
    if pages != 1:
        sys.exit(f"ERROR: the one-pager must be exactly 1 page, got {pages}. Shorten docs/one_pager.md.")
