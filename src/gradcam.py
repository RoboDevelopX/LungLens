"""Grad-CAM, written from scratch with PyTorch hooks (Selvaraju et al., 2017).

Idea in four steps, for one image and one class (e.g. PNEUMONIA):
  1. Forward pass. A *forward hook* saves the final feature map A
     (C channels x 7 x 7 for a 224 x 224 input).
  2. Backward pass from that class's score. A *gradient hook* saves dScore/dA,
     which says how much each feature-map cell pushed the score up or down.
  3. Average each channel's gradient over the 7 x 7 grid -> one importance
     weight per channel (alpha_k).
  4. Heatmap = ReLU(sum_k alpha_k * A_k): keep only regions that INCREASE the
     score, upsample 7 x 7 -> 224 x 224 and rescale to [0, 1].

The heatmap is coarse (each of the 7 x 7 cells covers about 32 x 32 pixels),
so it shows *where* the evidence is, not fine outlines.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        # Register the forward hook once; it fires on every forward pass.
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()
        # Ask autograd to call us back with the gradient of this tensor during backward().
        # Under torch.no_grad() there is no gradient to wait for, so plain inference still works.
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """x: 1 x 3 x H x W tensor. Returns (heatmap H x W in [0, 1], logits 1 x 2).

        class_idx=None explains the predicted class; pass 1 to always explain PNEUMONIA.
        """
        self.model.eval()
        with torch.enable_grad():  # works even if the caller is inside torch.no_grad()
            self.model.zero_grad()
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)                  # 1 x C x 1 x 1   (alpha_k)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))      # 1 x 1 x 7 x 7
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]

        cam = cam - cam.min()
        if cam.max() > 0:  # an all-zero map (no positive evidence) stays all zeros
            cam = cam / cam.max()
        return cam.cpu().numpy(), logits.detach()

    def remove(self) -> None:
        """Detach the hook (good practice once you are done with this explainer)."""
        self._handle.remove()


def target_layer_for(model: nn.Module) -> nn.Module:
    """The layer Grad-CAM watches: the ReLU after the backbone's last block (see src/model.py)."""
    return model.final_relu


# ---------------------------------------------------------------------------
# Drawing the heatmap on top of the X-ray
# ---------------------------------------------------------------------------
def overlay(gray_image: Image.Image, cam: np.ndarray, opacity: float = 0.5) -> Image.Image:
    """Blend a heatmap onto a grayscale X-ray.

    Low-attention pixels keep the original X-ray; high-attention pixels turn
    yellow -> orange -> red. The colour strength is cam * opacity, so the
    opacity slider in the app fades the whole heatmap in and out.
    """
    import matplotlib

    gray = np.asarray(gray_image.convert("L").resize(cam.shape[::-1]), dtype=np.float32) / 255.0
    base = np.stack([gray] * 3, axis=-1)                                  # H x W x 3
    colour = matplotlib.colormaps["YlOrRd"](cam)[..., :3]                 # H x W x 3 in [0, 1]
    strength = (opacity * cam)[..., None]                                 # H x W x 1
    blended = (1 - strength) * base + strength * colour
    return Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Measurements used by the Sanity Check page
# ---------------------------------------------------------------------------
def border_fraction(cam: np.ndarray, band: float = 0.10) -> float:
    """Share of the heatmap's total weight in the outer `band` of the image.

    The outer 10% band on each side is about 36% of the pixels. Real lung
    findings sit inside the image, while text markers ("R", "L"), rulers and
    borders sit at the edges, so a high value is a warning sign of shortcut
    learning.
    """
    h, w = cam.shape
    bh, bw = int(h * band), int(w * band)
    inner = cam[bh : h - bh, bw : w - bw].sum()
    total = cam.sum()
    return float((total - inner) / total) if total > 0 else 0.0


def randomized_copy(model: nn.Module, seed: int = 0, layers: str = "classifier") -> nn.Module:
    """Model-randomisation test (Adebayo et al., 2018).

    Copy the model and re-initialise some of its weights with random values.
    If Grad-CAM is truly explaining what the model LEARNED, the heatmaps of the
    randomised model should look very different. If they look the same, the
    "explanation" is just an edge detector and should not be trusted.

    layers="classifier": only the final layer; "top": also the last dense block
    (+ its batch-norm) for DenseNet, or the last stage for ResNet.

    Make the copy BEFORE attaching a GradCAM to `model` (deepcopy would copy the hook too).
    """
    torch.manual_seed(seed)
    rand = copy.deepcopy(model)
    modules = [rand.classifier]
    if layers == "top":
        if rand.backbone_name == "densenet121":
            modules += [rand.features.denseblock4, rand.features.norm5]
        else:
            modules.append(rand.features[-1])  # ResNet layer4
    for block in modules:
        for m in block.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    return rand.eval()


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two heatmaps (1 = same ordering of pixels)."""
    from scipy.stats import spearmanr

    return float(spearmanr(a.ravel(), b.ravel()).statistic)
