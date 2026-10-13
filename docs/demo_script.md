# LungLens demo video: script and shot list (2 min 45 s)

**Setup before recording**
- Run `streamlit run app.py` locally (or open the deployed app). Browser at 100% zoom, window about 1400 px wide, light mode, notifications off.
- Click through every sample once before recording, so the results are cached and appear instantly.
- Record the screen with your voice (OBS Studio, or the built-in recorder on Windows/macOS). Speak slowly. The numbers below are the real ones from `reports/metrics.json`; if the screen ever shows a different number, say the one on screen.

| Time | On screen | What you say |
|---|---|---|
| 0:00-0:15 | Home page, title and disclaimer visible | "Pneumonia is the biggest infectious killer of young children, and reading chest X-rays needs experts many clinics don't have. AI can help, but only if people can trust it. This is LungLens: see what the AI sees." |
| 0:15-0:40 | *Try a sample image* is already selected with *X-ray 1: Pneumonia (bacterial)*. Point at the result card, then the bar under the percentage. | "Here is a child's chest X-ray from the test set. LungLens flags signs consistent with pneumonia. The percentage is calibrated: we rescaled the model's scores on validation data so they match how often it is right there. The bar shows the normal zone, an inconclusive zone, and the pneumonia zone." |
| 0:40-1:05 | Drag the opacity slider down to 0 and back up to about 0.6; switch between *Side by side* and *Overlay only*. | "This is Grad-CAM, which I implemented myself with PyTorch hooks. It shows which regions pushed the pneumonia score up; here, the upper and middle chest on both sides. It shows where, not why, and it is not an outline of the infection." |
| 1:05-1:15 | Open *Clinician view*. | "The explanation is written for families. The clinician view adds the exact probability, the decision zones, where the attention sits, and the test-set sensitivity and specificity." |
| 1:15-1:35 | Pick *Not an X-ray: a photo*, then *Not an X-ray: a document*. | "A normal classifier would happily give a confident answer for a photo. LungLens checks the input first and refuses images that don't look like chest X-rays." |
| 1:35-1:55 | Pick *X-ray 7: Pneumonia scored below the threshold*. | "This is the only pneumonia X-ray in the whole test set that the model scored below fifty percent: forty-eight. Because it falls in the Inconclusive zone, LungLens asks for a clinician instead of calling it normal." |
| 1:55-2:20 | *Model performance* page: headline table, then move the threshold slider. | "Every number comes from 624 test X-rays the model never saw, with 95% confidence intervals. It caught 389 of 390 pneumonia cases. But it also flagged 140 of 234 healthy children, so specificity is our weak spot, and we show it instead of hiding it. The slider shows the trade-off." |
| 2:20-2:35 | *Sanity check* page: randomisation table and the border-attention examples. | "We also test the explanations: scramble the model's weights and the heatmaps change completely, so they reflect what the model learned. And we openly show the cases where it looks at the image edges." |
| 2:35-2:45 | *Limitations & ethics* page, scroll slowly. | "LungLens was trained on children from one hospital and hasn't been externally validated. It's an educational screening aid, not a diagnostic device. Thank you." |

**Shot list**
1. Title card (2 s, optional): "LungLens: See what the AI sees".
2. Home page with the pneumonia sample: result card, bar and heatmap.
3. Opacity slider and the view switch.
4. Clinician view expanded.
5. Photo refused; document refused.
6. X-ray 7: Inconclusive at 48%.
7. Model performance: metrics table and threshold slider.
8. Sanity check: randomisation and border attention.
9. Limitations & ethics.

**Tips**
- Keep the cursor still while talking; move it to point, then stop.
- If you retrain the model, re-run `python scripts/make_samples.py` and read the new numbers from the screen.
- The spoken text is about 360 words, which fits 2:45 at a calm pace. If you run long, drop the clinician view (1:05-1:15).
