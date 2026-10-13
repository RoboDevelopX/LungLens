"""The classifier: an ImageNet-pretrained backbone with a new 2-class head.

Transfer learning in one sentence: the backbone already knows how to see
edges, textures and shapes from 1.2 million ImageNet photos, so we only need
to teach it what pneumonia looks like on an X-ray.

We wrap torchvision's model in our own small class so that the three steps
are explicit and easy to point at:

    image -> features (backbone) -> final_relu -> average pool -> classifier -> 2 logits
                                      ^
                                      Grad-CAM looks here (last spatial feature map, 7x7)
"""

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

BACKBONES = ("densenet121", "resnet50")


class LungLensNet(nn.Module):
    def __init__(self, backbone: str = "densenet121", pretrained: bool = True, n_classes: int = 2):
        super().__init__()
        if backbone == "densenet121":
            # DenseNet121 is the backbone used by CheXNet (Rajpurkar et al., 2017).
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.densenet121(weights=weights)
            self.features = net.features                 # ends with BatchNorm "norm5"
            n_features = net.classifier.in_features      # 1024
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
            # Everything except ResNet's own pooling and fc layer.
            self.features = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
            )
            n_features = net.fc.in_features              # 2048
        else:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")

        self.backbone_name = backbone
        # A separate (not in-place) ReLU module gives Grad-CAM a clean place to
        # hook: its output is the final feature map, shape B x C x 7 x 7.
        self.final_relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)              # B x C x 7 x 7 -> B x C x 1 x 1
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.final_relu(self.features(x))
        pooled = torch.flatten(self.pool(fmap), 1)       # B x C
        return self.classifier(pooled)                   # B x 2 raw scores (logits)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze (False) or unfreeze (True) the pretrained backbone.

        Stage 1 of training freezes it so the new random head cannot wreck the
        pretrained features with large early gradients; stage 2 unfreezes it.
        """
        for p in self.features.parameters():
            p.requires_grad = trainable


def save_checkpoint(model: LungLensNet, path: Path, extra: dict | None = None) -> None:
    """Save weights plus the facts needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.backbone_name, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[LungLensNet, dict]:
    """Rebuild the model from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = LungLensNet(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def file_fingerprint(path: Path) -> str:
    """Short SHA-256 of a file. metrics.json stores it so the app can warn
    when the weights file does not match the reported numbers."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]
