"""
build_onepager.py - Builds docs/LungLens_One_Page.pdf, the one-page project
description for the hackathon submission. The numbers come straight from
results/metrics.json, so the page always matches the trained model.

Run from the project folder:  python docs/build_onepager.py
"""

import base64
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def image_tag(path: str, style: str = "") -> str:
    """Embed an image inside the HTML so the PDF is self-contained."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="{style}">'


def main():
    m = json.loads(Path("results/metrics.json").read_text())
    pct = lambda x: f"{x * 100:.1f}%"

    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    @page {{ size: A4; margin: 0; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'DejaVu Sans', sans-serif; color: #1b2430; font-size: 8.6pt; line-height: 1.38; }}
    .page {{ width: 210mm; height: 297mm; padding: 11mm 12mm 9mm; overflow: hidden; }}
    header {{ background: #0b4f6c; color: white; padding: 9pt 12pt; border-radius: 6pt; display: flex; justify-content: space-between; align-items: center; }}
    header h1 {{ margin: 0; font-size: 22pt; letter-spacing: 0.5pt; }}
    header .tag {{ font-size: 10pt; opacity: 0.95; }}
    header .meta {{ text-align: right; font-size: 8pt; opacity: 0.9; white-space: nowrap; margin-left: 10pt; }}
    h2 {{ font-size: 10.5pt; color: #0b4f6c; margin: 8pt 0 3pt; border-bottom: 1.5px solid #b9d3df; padding-bottom: 1.5pt; }}
    p {{ margin: 2pt 0 4pt; }}
    ul {{ margin: 2pt 0 4pt; padding-left: 13pt; }}
    li {{ margin-bottom: 2pt; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; }}
    .flow {{ display: flex; align-items: stretch; gap: 4pt; margin: 4pt 0; }}
    .step {{ flex: 1; background: #eef5f8; border: 1px solid #b9d3df; border-radius: 4pt; padding: 4pt; text-align: center; font-size: 7.6pt; }}
    .step b {{ display: block; color: #0b4f6c; font-size: 8.2pt; }}
    .arrow {{ align-self: center; color: #0b4f6c; font-weight: bold; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 8.2pt; }}
    td, th {{ border-bottom: 1px solid #d5dde5; padding: 2.5pt 4pt; text-align: left; }}
    th {{ color: #475569; font-weight: normal; }}
    td.num {{ text-align: right; font-weight: bold; }}
    .note {{ background: #fff7e6; border-left: 3pt solid #e69500; padding: 4pt 6pt; font-size: 8pt; }}
    .figs {{ display: flex; gap: 6pt; }}
    .figs img {{ width: 50%; }}
    footer {{ margin-top: 6pt; font-size: 7pt; color: #64748b; }}
    </style></head><body><div class="page">

    <header>
      <div><h1>🫁 LungLens</h1><div class="tag">See what the AI sees. Explainable chest X-ray screening for earlier pneumonia detection.</div></div>
      <div class="meta">UnivaBio Hackathon 2026<br>Theme: AI for Human Health</div>
    </header>

    <div class="cols">
    <div>
      <h2>The problem</h2>
      <p>Pneumonia is the single biggest infectious killer of young children: the WHO reports it caused
      14% of all deaths of children under five in 2019 (about 740,000 children). A chest X-ray is the
      standard way to confirm it, but many hospitals and clinics do not have a radiologist available to
      read X-rays quickly, so diagnosis and treatment are delayed.</p>
      <p>AI can read X-rays in seconds, but most AI tools are <b>black boxes</b>: they give an answer
      without a reason. Doctors are right not to trust an answer they cannot check.</p>

      <h2>Our solution</h2>
      <p><b>LungLens</b> is a web app that screens a chest X-ray for pneumonia in a few seconds and
      <b>shows its reasoning</b>: a heatmap highlights the parts of the lungs that drove the decision,
      so a doctor can see at a glance whether the AI is looking at real disease or at something
      irrelevant.</p>

      <h2>How it works</h2>
      <div class="flow">
        <div class="step"><b>1. Upload</b>Chest X-ray (JPG/PNG)</div><div class="arrow">→</div>
        <div class="step"><b>2. Analyse</b>DenseNet121 neural network</div><div class="arrow">→</div>
        <div class="step"><b>3. Score</b>Probability of pneumonia</div><div class="arrow">→</div>
        <div class="step"><b>4. Explain</b>Grad-CAM heatmap</div>
      </div>
      <ul>
        <li><b>Transfer learning:</b> we started from a DenseNet121 already trained by the open-source
        TorchXRayVision project on hundreds of thousands of adult chest X-rays, then fine-tuned it on
        4,694 pediatric X-rays (Kermany et al., 2018).</li>
        <li><b>Fair evaluation:</b> validation images were split <i>by patient</i>, so the model is never
        tested on a child it has already seen. The final test set of {m['test_images']} X-rays was used only once, at the end.</li>
        <li><b>Balanced decisions:</b> the dataset has 3× more pneumonia than normal images, so we weighted
        the training loss and chose the alert threshold that balances sensitivity and specificity.</li>
        <li><b>Grad-CAM</b> (Selvaraju et al., 2017) uses the network's own gradients to measure how much
        each region of the image contributed to the answer.</li>
      </ul>

      <h2>Key features</h2>
      <ul>
        <li>Upload your own X-ray or try built-in samples from the test set.</li>
        <li>Clear result: pneumonia probability, alert threshold and a plain-language verdict.</li>
        <li>Side-by-side original X-ray and adjustable heatmap overlay.</li>
        <li>Built-in guide to reading the heatmap and spotting model shortcuts.</li>
        <li>Sanity check that warns when an upload does not look like an X-ray.</li>
      </ul>
    </div>

    <div>
      <h2>See it in action</h2>
      {image_tag("docs/app_screenshot.png", "width:100%; border:1px solid #cbd5e1; border-radius:4pt")}

      <h2>Results on {m['test_images']} unseen test X-rays</h2>
      <table>
        <tr><th>Accuracy</th><td class="num">{pct(m['accuracy'])}</td>
            <th>AUC (1.0 = perfect)</th><td class="num">{m['auc']:.3f}</td></tr>
        <tr><th>Sensitivity (sick caught)</th><td class="num">{pct(m['sensitivity'])}</td>
            <th>Specificity (healthy cleared)</th><td class="num">{pct(m['specificity'])}</td></tr>
        <tr><th>Precision</th><td class="num">{pct(m['precision'])}</td>
            <th>Missed pneumonia cases</th><td class="num">{m['false_negative']} of {m['false_negative'] + m['true_positive']}</td></tr>
      </table>
      <div class="figs">{image_tag("results/confusion_matrix.png")}{image_tag("results/roc_curve.png")}</div>

      <h2>Responsible AI</h2>
      <div class="note"><b>Screening aid, not a diagnosis.</b> LungLens was trained on children aged 1 to 5
      from one hospital, so it may not work as well on adults or on X-rays from other machines. It flags
      X-rays for a doctor to review and never replaces one. The heatmap exists so that people can
      question the AI rather than blindly trust it.</div>

      <h2>Built with</h2>
      <p>Python · PyTorch · TorchXRayVision · scikit-learn · Streamlit · Matplotlib. Data: Kermany et al.,
      <i>Cell</i> 2018 (CC BY 4.0). Trained on an ordinary 4-core CPU in under 90 minutes.</p>
    </div>
    </div>
    <footer>LungLens is a student research prototype and is not a medical device.</footer>
    </div></body></html>"""

    out_html = Path("docs/onepager.html")
    out_html.write_text(page)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        pg = browser.new_page()
        pg.goto(out_html.resolve().as_uri())
        pg.pdf(path="docs/LungLens_One_Page.pdf", format="A4", print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()
    out_html.unlink()
    print("Wrote docs/LungLens_One_Page.pdf")


if __name__ == "__main__":
    main()
