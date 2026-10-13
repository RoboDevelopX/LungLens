# LungLens demo video: script and shot list (2:56)

The ready-made video, `LungLens_demo.mp4` (1920 x 1080, 2:56, captions burned in, no voice), is a screen
recording of the real app with the trained model, following this script step by step. Its captions are the
narration below, word for word, and `LungLens_demo.srt` holds the same captions as a subtitle file. The numbers are
the real ones from `reports/*.json` for model `b901dd05afd1`.

**To add your voice:** open the video in any editor (Clipchamp on Windows, iMovie on macOS, or CapCut), record a
voice track while it plays, reading each line when its caption appears, and export. Keep the captions: they help
judges who watch without sound. The narration is 329 words, a calm pace for 2:56.

**To record your own take instead:** run `streamlit run app.py` (or open the deployed app), set the browser to 100%
zoom and about 1500 px wide, click through every sample once so the results are cached, then follow the table. If
the screen ever shows a different number, say the one on screen.

| Time | On screen | Narration |
|---|---|---|
| 0:00-0:04 | Title card | *(optional)* "LungLens: see what the AI sees." |
| 0:04-0:09 | *Screen an X-ray* page, top: title and disclaimer | "Pneumonia is the single largest infectious cause of death in children worldwide." |
| 0:09-0:13 | Same view | "Reading chest X-rays takes trained radiologists, and many clinics have too few." |
| 0:13-0:19 | Point at *See what the AI sees* | "AI can help, but only if people can trust it. LungLens shows its work." |
| 0:19-0:28 | Open the *Sample image* list, run down the options, pick *X-ray 1: Pneumonia (bacterial)* | "Pick a sample X-ray or upload your own. This one is from the held-out test set." |
| 0:28-0:32 | Scroll to the result card; point at the percentage | "LungLens flags signs consistent with pneumonia and gives a calibrated chance." |
| 0:32-0:36 | Run the pointer along the three-zone bar | "The bar shows three zones: normal, inconclusive and pneumonia." |
| 0:36-0:41 | Point at *What this means* | "A plain-language explanation says what the result does and does not mean." |
| 0:41-0:47 | Point at the Grad-CAM image | "Grad-CAM, written from scratch with PyTorch hooks, shows which regions pushed the score up." |
| 0:47-1:02 | Drag *Heatmap opacity* to 0, then to 1, then to 0.6 | "Fade it in and out with the opacity slider..." |
| 1:02-1:07 | Click *Overlay only*, then *Side by side* | "...or view the overlay on its own." |
| 1:07-1:14 | Point at the heatmap | "It shows where the model looked, not why, and it is not an outline of the infection." |
| 1:14-1:20 | Open *Clinician view* | "The clinician view adds the exact probability, the decision zones and the test-set sensitivity and specificity." |
| 1:20-1:26 | Scroll up, choose *Upload your own*, click *Upload* and pick a photo | "You can also upload your own image. Here, a photo is uploaded by mistake." |
| 1:26-1:34 | Point at the refusal reason, then switch back to *Try a sample image* | "LungLens checks the input first and refuses, instead of giving a confident wrong answer." |
| 1:34-1:40 | Pick *Not an X-ray: a document* | "Documents and blank images are refused too." |
| 1:40-1:48 | Pick *X-ray 7: Pneumonia scored below the threshold*; point at the percentage | "This is the only pneumonia X-ray in the test set that the model scored below 50%." |
| 1:48-1:55 | Point at *Inconclusive: please consult a clinician* | "At 48% it lands in the Inconclusive zone, so LungLens asks a clinician to decide instead of saying normal." |
| 1:55-2:01 | Open *Model performance*; point at the metrics table | "Every number comes from 624 test X-rays the model never saw, with 95% confidence intervals." |
| 2:01-2:07 | Point at the note under the table (1 missed, 140 false alarms) | "It caught 389 of 390 pneumonia cases, but also flagged 140 of 234 healthy children." |
| 2:07-2:13 | Point at the *Specificity* row | "Specificity is the weak spot, and the app shows it instead of hiding it." |
| 2:13-2:30 | Scroll to *Try the trade-off yourself*; drag the threshold to 0.90, to 0.15, then back to 0.50 | "Move the threshold to see the trade-off between missed cases and false alarms." |
| 2:30-2:35 | Open *Sanity check*: Check 1, model randomisation | "We also test the explanations: with randomised weights, the heatmaps change completely..." |
| 2:35-2:42 | Scroll to Check 3, attention on the image edges | "...so they reflect what the model learned. Heatmaps that sit on the image edges are shown openly." |
| 2:42-2:48 | Open *Limitations & ethics*; scroll slowly | "LungLens was trained on children from one hospital and has not been externally validated." |
| 2:48-2:52 | Keep scrolling through the known limitations | "It is an educational screening aid, not a diagnostic device." |
| 2:52-2:56 | End card: github.com/RoboDevelopX/LungLens | *(optional)* "Thank you." |

**Shot list**
1. Title card: "LungLens: See what the AI sees".
2. Screen page with X-ray 1: result card, three-zone bar, plain-language explanation, Grad-CAM side by side.
3. Opacity slider, then the *Overlay only* and *Side by side* views.
4. Clinician view expanded.
5. Upload a photo: refused. Pick the document sample: refused.
6. X-ray 7: Inconclusive at 48%.
7. Model performance: metrics table with confidence intervals, then the threshold slider.
8. Sanity check: randomisation table, then attention on the image edges.
9. Limitations & ethics, then the end card with the GitHub link.

**Tips**
- Keep the pointer still while talking; move it to point, then stop.
- If you retrain the model, re-run `python scripts/make_samples.py` and read the new numbers from the screen: the
  ready-made video shows this model's numbers.
