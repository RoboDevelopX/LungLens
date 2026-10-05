"""
build_code_pdf.py - Builds docs/LungLens_Code.pdf, a printable copy of every
source file in the project (required by the hackathon submission).

Run from the project folder:  python docs/build_code_pdf.py
Needs: pygments, playwright (with Chromium installed).
"""

import html
import os
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, TextLexer

# The files to include, in the order a reader should go through them.
FILES = [
    ("README.md", "Project overview and how to run it"),
    ("requirements.txt", "Python libraries the project needs"),
    ("prepare_data.py", "Step 1: download and unpack the X-ray dataset"),
    ("dataset.py", "Loading images and the patient-level train/validation split"),
    ("network.py", "The neural network and image preprocessing"),
    ("train.py", "Step 2: fine-tune the network"),
    ("evaluate.py", "Step 3: measure the model on the unseen test set"),
    ("gradcam.py", "Explainability: Grad-CAM heatmaps"),
    ("app.py", "Step 4: the interactive Streamlit web app"),
]

formatter = HtmlFormatter(linenos="table", style="friendly", cssclass="code")


def render(path: str) -> str:
    source = Path(path).read_text()
    try:
        lexer = get_lexer_for_filename(path)
    except Exception:
        lexer = TextLexer()
    return highlight(source, lexer, formatter)


def main():
    metrics = json.loads(Path("results/metrics.json").read_text())
    total_lines = sum(len(Path(f).read_text().splitlines()) for f, _ in FILES)

    toc = "".join(f"<tr><td><code>{f}</code></td><td>{html.escape(d)}</td>"
                  f"<td>{len(Path(f).read_text().splitlines())}</td></tr>" for f, d in FILES)
    sections = "".join(
        f'<section><h2>{i}. {f}</h2><p class="desc">{html.escape(d)}</p>{render(f)}</section>'
        for i, (f, d) in enumerate(FILES, 1))

    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    @page {{ size: A4; margin: 14mm 12mm; }}
    body {{ font-family: 'DejaVu Sans', sans-serif; color: #1b2430; font-size: 10pt; }}
    h1 {{ font-size: 28pt; margin: 0 0 4pt; color: #0b4f6c; }}
    h2 {{ font-size: 14pt; color: #0b4f6c; border-bottom: 2px solid #0b4f6c; padding-bottom: 3pt; }}
    .cover {{ page-break-after: always; padding-top: 40mm; }}
    .tag {{ font-size: 13pt; color: #475569; margin-bottom: 18pt; }}
    table.toc {{ border-collapse: collapse; width: 100%; margin-top: 12pt; }}
    table.toc td, table.toc th {{ border-bottom: 1px solid #d5dde5; padding: 5pt; text-align: left; }}
    section {{ page-break-before: always; }}
    .desc {{ color: #475569; margin-top: -4pt; }}
    .code {{ font-size: 7.6pt; }}
    .code pre {{ margin: 0; font-family: 'DejaVu Sans Mono', monospace; white-space: pre-wrap; word-break: break-all; }}
    .code td.linenos {{ color: #94a3b8; padding-right: 8pt; vertical-align: top; user-select: none; }}
    .code td.linenos pre {{ white-space: pre; }}
    .codetable {{ width: 100%; table-layout: fixed; }}
    .codetable td.linenos {{ width: 26pt; }}
    {formatter.get_style_defs('.code')}
    </style></head><body>
    <div class="cover">
      <h1>LungLens</h1>
      <div class="tag">See what the AI sees. Explainable chest X-ray screening for earlier pneumonia detection.</div>
      <p><b>Source code listing</b> &middot; UnivaBio Hackathon, AI for Human Health</p>
      <p>Language: Python &middot; {len(FILES)} files &middot; {total_lines} lines (including comments)</p>
      <p>Test-set result: {metrics['accuracy']:.1%} accuracy, {metrics['sensitivity']:.1%} sensitivity,
         {metrics['specificity']:.1%} specificity, AUC {metrics['auc']:.3f} on {metrics['test_images']} unseen X-rays.</p>
      <table class="toc"><tr><th>File</th><th>Purpose</th><th>Lines</th></tr>{toc}</table>
    </div>
    {sections}
    </body></html>"""

    out_html = Path("docs/code_listing.html")
    out_html.write_text(page)
    with sync_playwright() as p:
        # CHROMIUM_PATH lets you point at an existing Chrome/Chromium if needed.
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        pg = browser.new_page()
        pg.goto(out_html.resolve().as_uri())
        pg.pdf(path="docs/LungLens_Code.pdf", format="A4", print_background=True,
               display_header_footer=True,
               header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#64748b">'
                               'LungLens source code &middot; page <span class="pageNumber"></span> of '
                               '<span class="totalPages"></span></div>',
               margin={"top": "14mm", "bottom": "16mm", "left": "12mm", "right": "12mm"})
        browser.close()
    out_html.unlink()
    print("Wrote docs/LungLens_Code.pdf")


if __name__ == "__main__":
    main()
