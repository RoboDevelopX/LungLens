# LungLens demo video: script and shot list

Target length: about 3 minutes. Record your screen (OBS Studio, or the built-in
recorder: Win+G on Windows, Cmd+Shift+5 on Mac) with your microphone on.
Before you start, run `streamlit run app.py`, open the app in your browser,
zoom the browser to about 110%, and close other tabs.

Speak in your own words. The lines below are a guide, not something to read word for word.

---

## Shot 1: The hook (0:00 to 0:20)
**On screen:** the LungLens app home page, or a title slide with the logo and tagline.

> "Pneumonia kills more young children than any other infection, around 740,000
> children under five every year. A chest X-ray can confirm it, but in many places
> there isn't a radiologist available to read it quickly. I built LungLens: an AI
> that screens chest X-rays for pneumonia in seconds, and, most importantly,
> shows you *why* it made its decision."

## Shot 2: The problem with black-box AI (0:20 to 0:40)
**On screen:** stay on the app, point the cursor at the tagline "See what the AI sees".

> "Most medical AI just gives you an answer. A doctor can't check that, and
> shouldn't trust it blindly. So LungLens doesn't just say 'pneumonia' or
> 'normal'. It draws a heatmap showing which parts of the X-ray drove the decision."

## Shot 3: Demo with a pneumonia X-ray (0:40 to 1:20)
**On screen:** choose "Try a sample", pick **Pneumonia example 3**. Wait for the result.
Slowly move the cursor over the heatmap. Drag the "Heatmap strength" slider down
to 0 and back up.

> "Here's an X-ray from a child the model has never seen. LungLens flags signs of
> pneumonia, with the probability here. On the right is the Grad-CAM heatmap: red
> means 'this is where I found the evidence'. It's pointing inside the lungs,
> where pneumonia shows up as white, cloudy patches. I can fade the heatmap
> in and out to compare it with the original X-ray."

## Shot 4: Demo with a normal X-ray (1:20 to 1:45)
**On screen:** pick **Normal example 1**.

> "Now a healthy child's X-ray. LungLens says no signs of pneumonia, and the
> heatmap now shows the evidence for 'normal' instead: clear lung fields."

## Optional Shot 4b: An honest mistake (adds about 15 seconds)
**On screen:** pick **Normal example 3**. LungLens flags it (about 24%, just above the
13% alert level) even though doctors labelled it normal.

> "It isn't perfect. This healthy X-ray gets flagged as a false alarm. LungLens is
> tuned to rarely miss a sick child, and the price is some false alarms, which a
> doctor then clears. That's exactly why the heatmap matters: a doctor can see what
> the AI reacted to and decide for themselves."

## Shot 5: Uploading your own X-ray (1:45 to 2:05)
**On screen:** switch to "Upload my own", drag in an X-ray file (for example one
from `data/chest_xray/test/`). Optionally upload a colour photo to show the warning.

> "Anyone can upload their own X-ray image. LungLens also checks the upload: if
> it's a colour photo instead of an X-ray, it warns you that the result won't mean
> anything."

## Shot 6: How it works and results (2:05 to 2:40)
**On screen:** the sidebar with the test results, then `results/confusion_matrix.png`
and briefly `gradcam.py` in your code editor.

> "Under the hood, I used transfer learning: a DenseNet121 network already trained
> on hundreds of thousands of chest X-rays, which I fine-tuned on about 4,700
> children's X-rays. I split the data by patient so the model is never tested on a
> child it has seen. On 624 completely unseen test X-rays it reaches
> 90.1% accuracy and an AUC of 0.955. It catches 98.5% of pneumonia cases,
> missing only 6 out of 390, at the cost of some false alarms on healthy X-rays. The heatmap uses a technique called Grad-CAM: it takes the gradients
> flowing back into the last layer of the network to measure how much each region
> of the image mattered."

## Shot 7: Responsible AI and close (2:40 to 3:00)
**On screen:** scroll to the yellow "Screening aid, not a diagnosis" warning, then end on the app.

> "LungLens is a screening aid, not a doctor. It was trained on young children from
> one hospital, so it needs much more testing before real-world use. But I think
> AI in medicine should always show its work, and that's what LungLens does.
> Thanks for watching!"

---

## Tips
- Do one practice run first; it is normal to need two or three takes.
- Keep the mouse still while you talk about something, and move it to point at what you mention.
- If the video runs long, shorten Shot 5 first.
