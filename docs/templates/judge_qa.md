### 1. "How well does it work?"

On the official test set ({{n_test}} X-rays: {{test_pneumonia}} pneumonia, {{test_normal}} normal), which was
never used for training, calibration or choosing the threshold, LungLens reaches **sensitivity {{sens}}**
(95% CI {{sens_ci}}), **specificity {{spec}}** ({{spec_ci}}) and **AUROC {{auroc}}** ({{auroc_ci}}). In
plain numbers it missed {{fn}} of {{test_pneumonia}} pneumonia X-rays and raised {{fp}} false alarms on
{{test_normal}} normal ones. The intervals come from {{n_bootstrap}} bootstrap resamples. It is very good at
not missing pneumonia and weak at clearing healthy children (next question). These numbers describe this one
pediatric dataset; we have no external validation.

### 2. "Why is specificity so much lower on the test set than on validation?"

At the same threshold, specificity was {{val_spec}} on validation but {{spec}} on the test set: {{fp}} of
{{test_normal}} normal test X-rays were flagged, {{fp_conf90}} of them with at least 90% confidence. We report
this instead of re-tuning, because tuning on the test set would make every number on it meaningless. We cannot
prove the cause. Our validation images come from the official training folder, so they share whatever the
training images and labels have in common, and the test folder evidently differs. The dataset description
says that "the evaluation set was also checked by a third expert"; if that is the test folder, its labels may
follow a stricter standard than the training labels the model learned from. In practice it means the
threshold should be re-set on local data before LungLens is used anywhere, and that a flag is a reason for a
clinician to look, never a diagnosis.

### 3. "How did you choose the threshold?"

The rule was fixed *before* testing and applied to validation data only: take the highest threshold that still
catches {{target_sensitivity}} of pneumonia cases (sensitivity-first, because a screening tool should rarely miss
a sick child), but never go above {{max_threshold}}. The cap exists because the model is very confident: the
{{target_sensitivity}} rule alone allowed a threshold of {{threshold_for_target}}, which would have called an X-ray
"normal" at any calibrated chance of pneumonia below that. The operating threshold is {{threshold}}
(validation sensitivity {{val_sens}}, specificity {{val_spec}}). A false alarm costs a doctor's review; a miss can
delay treatment. The *Model Performance* page has a slider that shows the trade-off live; we did not change the
threshold after seeing the test results.

### 4. "How do I know the percentage it shows is honest?"

The raw network outputs are rescaled with **temperature scaling** (Guo et al., 2017): one number, T =
{{temperature}}, fitted on validation data only. The expected calibration error fell from {{ece_val_before}} to
{{ece_val_after}} on validation, but only from {{ece_test_before}} to {{ece_test_after}} on the untouched test
set: calibration carries over only to similar images, and on the test folder the model stays over-confident,
mostly on normal X-rays it calls pneumonia. So the app calls the number a "calibrated chance" but never shows 0%
or 100%, and the *How it works* page tells users to read it as a rough guide. Prevalence matters too: the
training data is about three-quarters pneumonia, far more than a real screening population, so in practice the
true chance behind a given score would be lower.

### 5. "What does the heatmap actually show? Can I trust it?"

Grad-CAM (Selvaraju et al., 2017), which I implemented with two PyTorch hooks: one saves the last feature map,
the other its gradient with respect to the pneumonia score; the gradients are averaged into one weight per
channel, the weighted maps are summed, negative values dropped, and the 7 x 7 result is stretched over the
image. A test checks it against the `pytorch-grad-cam` library. It shows *where* the evidence was, not *why*.

To check it is not just an edge detector we ran the model-randomisation test (Adebayo et al., 2018) on
{{rand_n}} test X-rays: {{rand_verdict}} (Mean rank correlation with the trained model's heatmap: {{rand_classifier}}
after randomising the final layer, {{rand_top}} after also randomising the last dense block.)
We also measured attention on the image edges, where text markers sit: {{border_verdict}} Normal X-rays lean on
the edges more than pneumonia X-rays (median border share {{border_tn}} for correctly cleared normal X-rays,
{{border_tp}} for correctly flagged pneumonia): with little pneumonia evidence in the lungs, the map shows
whatever weak evidence exists, often at the borders.

### 6. "What happens if I upload something that is not a chest X-ray?"

Two layers. First, simple image checks with thresholds set from the statistics of the training X-rays only
(colour, contrast, blank or mostly white/black images, size, shape). Second, the model's **energy score**
(Liu et al., 2020): inputs the model finds unfamiliar get a warning. Measured on images the model never saw
(the share flagged, i.e. refused or warned):

{{ood_list}}

The guard is not perfect: other grayscale medical images are the hardest case, which is why the energy
check only warns and the disclaimer stays on screen. A small classifier trained to recognise chest X-rays
would be the next step.

### 7. "Your data is children from one hospital. Would it work on adults or other hospitals?"

We do not know, and we expect it to be worse. Every image comes from Guangzhou Women and Children's Medical
Center (ages 1-5). Pneumonia models are known to pick up hospital-specific cues and lose accuracy at new sites
(Zech et al., 2018). The next step would be an external test on an adult dataset such as RSNA Pneumonia
Detection or a different children's hospital, before any claim beyond this dataset.

### 8. "How did you avoid data leakage?"

Three rules. (1) The validation set is split **by patient** from the official training folder, so no child
appears in both ({{n_train}} training / {{n_val}} validation X-rays). (2) The threshold, the temperature and
the energy threshold are all fitted on validation data. (3) The {{n_test}} test images were scored once, at the
end. One honest caveat: the file names show {{shared_ids}} patient numbers in both the official train and test
folders ({{shared_test_images}} test images), always with a different infection type, which suggests separate
numbering rather than the same child, but we cannot prove it from the file names alone.

### 9. "Why DenseNet121 and transfer learning?"

With about four thousand training images, learning from scratch would overfit. DenseNet121 pretrained on
ImageNet already detects edges and textures; we first trained only a new 2-class head with the backbone
frozen, then fine-tuned everything at a 10x lower learning rate, with early stopping on validation AUROC (best
checkpoint: {{best_stage}} stage, epoch {{best_epoch}}; {{n_epochs}} epochs in {{train_minutes}} minutes on a
4-core CPU). DenseNet121 is also the backbone of CheXNet (Rajpurkar et al., 2017), is small (about 7 million
parameters) and runs on a CPU: the full screening pipeline including Grad-CAM takes {{infer_median}} per image
(median of {{infer_runs}} runs on {{infer_threads}}). ResNet50 is one command-line flag away.

### 10. "What does 'Inconclusive' mean, and how often does it happen?"

Scores between {{low}} and {{high}} are the least reliable, so the app says "Inconclusive: please consult a
clinician" instead of forcing a label. The zone holds every probability whose odds are within a factor of
{{odds_factor}} of the threshold's odds; working on the odds means the zone can never run past 0% or 100%. On
the test set {{inconclusive_rate}} of X-rays fall in that zone, and the accuracy on the rest is {{acc_decided}}.
{{missed_in_band}} Both edges of the zone can be set in `config.yaml`.

### 11. "The dataset is imbalanced. Did that bias the model?"

About three pneumonia images for every normal one. We used class-weighted cross-entropy so each class counts
equally during training, report metrics that do not reward always saying "pneumonia" (specificity, AUROC,
precision), and set the threshold with an explicit rule on validation data, fixed before testing.

### 12. "Is this ready for clinical use?"

No. It would need external and prospective validation, a reader study against radiologists, subgroup
analysis (the dataset has no age, sex or ethnicity data), regulatory review as software as a medical device,
and monitoring after deployment. LungLens is an educational prototype that shows how such a tool can be built
to be honest about its uncertainty.

### 13. "What would you do with more time?"

External validation on adult and multi-hospital data, with the threshold re-set on local validation data;
better input checking (a small classifier trained to recognise chest X-rays); lung segmentation so attention
outside the lungs can be measured directly; and a radiologist review of the false alarms and the missed case.
