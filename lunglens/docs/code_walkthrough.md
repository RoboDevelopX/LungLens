# LungLens: code walkthrough and judge prep

Read this before judging. It explains each file in the order you would present
it, then lists questions judges are likely to ask with short answers you can
give in your own words.

## The 30-second version

"LungLens takes a chest X-ray, runs it through a neural network that outputs the
probability of pneumonia, and then uses Grad-CAM to show which parts of the image
drove that answer. I used transfer learning from a network already trained on
chest X-rays, fine-tuned it on about 4,700 children's X-rays, and tested it on
624 X-rays it had never seen."

## Walking through the code (in order)

### 1. `prepare_data.py`: getting the data
- Downloads the Kermany dataset from Google's public copy (no Kaggle login).
- The files are in **TFRecord** format: records stored back to back. The
  `read_tfrecord` function reads each record's length (8 bytes), skips a
  checksum, reads the data, and skips another checksum.
- Each image record is a JPEG; the matching path record (e.g.
  `./PNEUMONIA/person1_bacteria_1.jpeg`) tells us the label from the folder name.
- Result: `data/chest_xray/train|test/NORMAL|PNEUMONIA/*.jpeg`.

### 2. `dataset.py`: loading images fairly
- `patient_id()` reads the child's ID out of the file name.
- `train_val_split()` holds out 10% of **patients** (not images) for validation,
  so one child's X-rays never appear on both sides. Otherwise the model could
  "recognise the child" and the score would be too optimistic.
- `XrayDataset` is a PyTorch Dataset: give it an index and it returns
  (image tensor, label).

### 3. `network.py`: the model
- `LungLensNet` = pretrained DenseNet121 **feature extractor** + a new
  `nn.Linear(1024, 1)` **classifier**.
- `forward()`: image → 1024 feature maps of 7×7 → average each map (global
  average pooling) → 1024 numbers → one **logit**. `sigmoid(logit)` gives a
  probability from 0 to 1.
- `freeze_early_layers()`: only the last dense block and the classifier learn.
  Early layers detect general things like edges, which are already good.
- `to_model_range()`: the pretrained network expects pixels from -1024 to 1024.
- `train_transform` adds random crops, small rotations and brightness changes
  (data augmentation) so the model generalises.

### 4. `train.py`: teaching the model
- The training loop: forward pass → loss → `loss.backward()` (back-propagation
  computes gradients) → `optimizer.step()` (adjusts weights).
- **Loss**: `BCEWithLogitsLoss` (binary cross-entropy) with `pos_weight` to
  correct for 3× more pneumonia images than normal ones.
- **Optimiser**: AdamW, with a bigger learning rate for the brand-new layer.
- After each epoch it computes validation **AUC** and keeps the best model.
- `best_threshold()` chooses the probability cut-off that maximises
  sensitivity + specificity on validation data (Youden's J).

### 5. `evaluate.py`: honest testing
- Uses the 624 test X-rays exactly once, with the saved threshold.
- Computes accuracy, sensitivity, specificity, precision, F1 and AUC, and saves
  the confusion matrix, ROC curve and Grad-CAM examples to `results/`.

### 6. `gradcam.py`: explaining the decision
1. Forward pass, keeping the 7×7 feature maps.
2. `torch.autograd.grad` gives the gradient of the score with respect to each
   feature map: "how much would the answer change if this map got stronger?"
3. Average each map's gradient into a weight, multiply and add up the maps,
   keep the positive part (ReLU).
4. Upscale to the image size and colour it (red = important).
- For a "normal" answer we flip the sign of the score, so the map shows evidence
  for "normal".
- `overlay()` blends colour into the X-ray in proportion to the heat, so
  unimportant regions stay clear.

### 7. `app.py`: the user interface
- `@st.cache_resource` loads the model only once.
- `looks_like_xray()` warns if the image is colourful (X-rays are grayscale).
- Three columns: original, heatmap, result plus explanation. The slider changes
  the heatmap strength live.

## Likely judge questions

**Why DenseNet121?** It is a well-tested architecture for chest X-rays (CheXNet
used it), and pretrained chest X-ray weights exist for it, which matters when you
only have about 5,000 images.

**What is transfer learning?** Reusing a network trained on a big dataset as the
starting point for a smaller task. Our network had already learned what lungs,
ribs and opacities look like, so it only needed to adapt to children and to the
normal-versus-pneumonia question.

**Why not just use accuracy?** In screening, missing a sick child (false
negative) is worse than a false alarm. Sensitivity tells us how many sick
children we catch; specificity tells us how many false alarms we raise. AUC
measures how well the scores separate the two groups at every threshold.

**Why is the threshold not 0.5?** The model's probabilities are not perfectly
calibrated, and the classes are imbalanced. We picked the threshold on
validation data (never test data) to balance sensitivity and specificity.

**Why is specificity (76%) lower than sensitivity (98.5%)?** On purpose and by
circumstance. The threshold (13%) was chosen to balance the two on validation
data, and on the test set the model ends up favouring catching pneumonia: it
missed only 6 of 390 sick children but raised 56 false alarms on 234 healthy
ones. For screening, a false alarm costs a doctor's second look, while a miss
can cost a child's health. This test set is also known to differ a little from
the training images, so in a real hospital we would re-tune the threshold on
that hospital's own X-rays.

**How do you know the model isn't cheating?** Patient-level split, a test set
used once, and the heatmaps: we can see whether it looks inside the lungs or at
labels and image edges. When it looks at the wrong place, that is a warning sign
and the app tells users so.

**What are the limitations?** One hospital, young children only, two classes,
and coarse heatmaps. It would need testing on data from other hospitals and
approval as a medical device before any clinical use.

**What would you do next?** Test on other datasets (e.g. RSNA), add more
classes (bacterial versus viral pneumonia, COVID-19), calibrate the
probabilities, and get feedback from radiologists on the heatmaps.

**Did you use AI to build it?** Yes, as the hackathon encourages. An AI assistant
helped write the code; I made sure I understand every part and can explain it.
