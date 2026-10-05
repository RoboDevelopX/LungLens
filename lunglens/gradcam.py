"""
gradcam.py - "See what the AI sees": Grad-CAM heatmaps.

The problem
-----------
A neural network gives an answer but not a reason. A doctor should never
trust a "black box", so LungLens shows WHERE in the X-ray the model found
the evidence for its answer.

Grad-CAM in four steps (Selvaraju et al., 2017)
-----------------------------------------------
1. Run the X-ray through the network and keep the last 7x7 grid of
   feature maps (1024 maps, each 7x7). Each cell covers one patch of the
   X-ray, roughly 32x32 pixels.
2. Ask: "if this feature map got a bit stronger, how much would the
   answer change?" That is the gradient of the score with respect to
   the feature map, computed by back-propagation.
3. Average each map's gradient into one importance weight, then add up
   all 1024 maps multiplied by their weights. Keep only positive values
   (ReLU): we want regions that SUPPORT the answer.
4. Stretch the 7x7 result up to the X-ray's size and colour it:
   red = strong evidence, blue = little evidence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


def gradcam(model, image_tensor: torch.Tensor, threshold: float = 0.5):
    """
    Returns (probability_of_pneumonia, heatmap) where heatmap is a 224x224
    numpy array with values from 0 (ignored) to 1 (most important).
    """
    model.eval()
    # Step 1: forward pass, keeping the feature maps.
    logit, feature_maps = model(image_tensor, return_features=True)
    probability = torch.sigmoid(logit).item()

    # Explain the answer the model actually gives. For a "pneumonia" answer
    # we explain the pneumonia score; for a "normal" answer we flip the
    # sign, so the map shows the evidence for "normal" instead.
    target = logit if probability >= threshold else -logit

    # Step 2: gradient of the target score with respect to the feature maps.
    gradients = torch.autograd.grad(target.sum(), feature_maps)[0]   # (1, 1024, 7, 7)

    # Step 3: one weight per feature map, then a weighted sum of the maps.
    weights = gradients.mean(dim=(2, 3), keepdim=True)               # (1, 1024, 1, 1)
    cam = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)

    # Step 4: upscale to 224x224 and rescale to 0..1.
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return probability, cam


def overlay(image: Image.Image, heatmap: np.ndarray, opacity: float = 0.6) -> Image.Image:
    """
    Colour the heatmap (blue -> red) and blend it on top of the X-ray.
    Each pixel's colour strength follows its heat, so unimportant areas
    stay a clear X-ray and only the important regions light up.
    """
    xray = np.asarray(image.convert("RGB")).astype(float) / 255.0
    # Resize the 224x224 heatmap to the original X-ray size.
    heat = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.BILINEAR)
    heat = np.asarray(heat).astype(float) / 255.0
    # The "jet" colour map turns 0..1 into blue..green..yellow..red.
    coloured = colormaps["jet"](heat)[:, :, :3]
    alpha = (heat * opacity)[:, :, None]            # per-pixel blend strength
    blended = xray * (1 - alpha) + coloured * alpha
    return Image.fromarray(np.uint8(blended * 255))
