"""Build LungLens_Code.pdf: a title page, the file tree, then every source file
with syntax highlighting and line numbers (hackathon deliverable: "Code PDF").

Uses reportlab (PDF layout) + pygments (syntax colours). Fonts are the DejaVu
fonts that ship with matplotlib, so the script works the same on any OS.

Usage:
    python scripts/make_code_pdf.py            # writes LungLens_Code.pdf in the repo root
"""

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from pygments import lex
from pygments.lexers import get_lexer_for_filename
from pygments.styles import get_style_by_name
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, XPreformatted

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "LungLens_Code.pdf"

# Which files count as "code developed for the project" (in this order).
INCLUDE = ["app.py", "pages/*.py", "src/*.py", "scripts/*.py", "tests/*.py", "config.yaml",
           "requirements.txt", "requirements-dev.txt", ".streamlit/config.toml", "notebooks/train_colab.ipynb"]

FONT_SIZE = 7.2
LINE_HEIGHT = 8.8


def register_fonts() -> None:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("Mono", "DejaVuSansMono.ttf"), ("MonoBold", "DejaVuSansMono-Bold.ttf"),
                       ("Sans", "DejaVuSans.ttf"), ("SansBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))


def source_files() -> list[Path]:
    files = []
    for pattern in INCLUDE:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and path not in files and "__pycache__" not in path.parts:
                files.append(path)
    return files


def notebook_as_text(path: Path) -> str:
    """Show a notebook as its cells (code and markdown), not as raw JSON."""
    import json

    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    parts = []
    for i, cell in enumerate(cells, 1):
        body = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            body = "\n".join("# " + line for line in body.splitlines())
        parts.append(f"# ---- cell {i} ({cell['cell_type']}) ----\n{body}")
    return "\n\n".join(parts)


def safe(text: str) -> str:
    """Replace characters the PDF font cannot draw (e.g. emoji) with a visible placeholder."""
    return "".join(ch if ord(ch) < 0x10000 else "[emoji]" for ch in text)


def highlighted_lines(path: Path, style) -> list[str]:
    """Return one reportlab-markup string per source line, coloured by pygments."""
    if path.suffix == ".ipynb":
        text, lexer = notebook_as_text(path), get_lexer_for_filename("x.py")
    else:
        text = path.read_text(encoding="utf-8")
        try:
            lexer = get_lexer_for_filename(path.name)
        except Exception:
            lexer = get_lexer_for_filename("x.txt")
    lines, current = [], ""
    for token_type, value in lex(safe(text), lexer):
        # Find the colour pygments' style assigns to this token (walk up to parent token types).
        t = token_type
        while t not in style.styles and t.parent is not None:
            t = t.parent
        spec = style.style_for_token(t)
        colour = spec["color"]
        for i, piece in enumerate(value.split("\n")):
            if i > 0:
                lines.append(current)
                current = ""
            if piece:
                chunk = escape(piece)
                if colour:
                    chunk = f'<font color="#{colour}">{chunk}</font>'
                if spec["bold"]:
                    chunk = f'<font name="MonoBold">{chunk}</font>'
                current += chunk
    if current:
        lines.append(current)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def wrap_plain_length(markup: str) -> int:
    """Visible length of a markup line (tags removed, entities counted once)."""
    import re

    plain = re.sub(r"<[^>]+>", "", markup)
    return len(re.sub(r"&(amp|lt|gt);", "x", plain))


def file_tree(files: list[Path]) -> str:
    lines = ["lunglens/"]
    for path in files:
        rel = path.relative_to(ROOT)
        n = len(path.read_text(encoding="utf-8").splitlines())
        lines.append(f"  {str(rel):<45} {n:>5} lines")
    return "\n".join(lines)


def build() -> Path:
    register_fonts()
    style = get_style_by_name("friendly")
    files = source_files()
    width, _ = A4
    usable = width - 30 * mm
    char_w = pdfmetrics.stringWidth("M", "Mono", FONT_SIZE)
    max_chars = int(usable / char_w) - 6  # minus the line-number gutter

    title = ParagraphStyle("t", fontName="SansBold", fontSize=26, leading=32, textColor=colors.HexColor("#0b0b0b"))
    sub = ParagraphStyle("s", fontName="Sans", fontSize=12, leading=17, textColor=colors.HexColor("#52514e"))
    h = ParagraphStyle("h", fontName="SansBold", fontSize=11, leading=14, textColor=colors.HexColor("#0b0b0b"),
                       spaceBefore=4, spaceAfter=6)
    code = ParagraphStyle("c", fontName="Mono", fontSize=FONT_SIZE, leading=LINE_HEIGHT)

    try:
        # "-dirty" is appended when tracked files differ from that commit, so the PDF never overclaims.
        commit = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    total_lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)

    story = [
        Spacer(1, 60 * mm),
        Paragraph("LungLens: See what the AI sees", title),
        Spacer(1, 6 * mm),
        Paragraph("Explainable chest X-ray screening for earlier pneumonia detection", sub),
        Paragraph("Complete source code &middot; UnivaBio student hackathon 2026", sub),
        Spacer(1, 10 * mm),
        Paragraph(f"Author: Ribhu &middot; generated {dt.date.today().isoformat()} from git commit {commit}", sub),
        Paragraph(f"{len(files)} files &middot; {total_lines:,} lines", sub),
        PageBreak(),
        Paragraph("File tree (files included in this document)", h),
        XPreformatted(escape(file_tree(files)), code),
        PageBreak(),
    ]
    gutter = '<font color="#9a9890">{:>4}</font>  '
    for path in files:
        story.append(Paragraph(escape(str(path.relative_to(ROOT))), h))
        rows = []
        for n, line in enumerate(highlighted_lines(path, style), 1):
            if wrap_plain_length(line) <= max_chars:
                rows.append(gutter.format(n) + line)
            else:
                # Too wide for the page: re-wrap the plain text (losing colours for this line only).
                import re

                plain = re.sub(r"<[^>]+>", "", line)
                plain = plain.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                pieces = [plain[i:i + max_chars] for i in range(0, len(plain), max_chars)]
                rows.append(gutter.format(n) + escape(pieces[0]))
                rows += ['<font color="#9a9890">   +</font>  ' + escape(p) for p in pieces[1:]]
        # Split long files into chunks so pages break cleanly.
        for i in range(0, len(rows), 80):
            story.append(XPreformatted("\n".join(rows[i:i + 80]), code))
        story.append(PageBreak())

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(colors.HexColor("#6b6a65"))
        canvas.drawString(15 * mm, 10 * mm, "LungLens source code")
        canvas.drawRightString(width - 15 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title="LungLens source code", author="Ribhu")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    out = build()
    from pypdf import PdfReader

    print(f"wrote {out.relative_to(ROOT)}: {len(PdfReader(out).pages)} pages, {os.path.getsize(out) / 1e6:.2f} MB")
    sys.exit(0)
