"""Model: output shapes, freezing, and checkpoint save/load."""

import pytest
import torch

from src.model import LungLensNet, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("backbone,channels", [("densenet121", 1024), ("resnet50", 2048)])
def test_forward_shapes(backbone, channels):
    model = LungLensNet(backbone, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        fmap = model.final_relu(model.features(x))
        logits = model(x)
    assert fmap.shape == (2, channels, 7, 7)   # the map Grad-CAM uses
    assert logits.shape == (2, 2)              # Normal, Pneumonia
    assert (fmap >= 0).all()                   # after the final ReLU


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        LungLensNet("vgg16", pretrained=False)


def test_freeze_and_unfreeze_backbone(tiny_model):
    tiny_model.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in tiny_model.features.parameters())
    assert all(p.requires_grad for p in tiny_model.classifier.parameters())
    tiny_model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in tiny_model.features.parameters())


def test_checkpoint_round_trip(tiny_model, tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(tiny_model, path, extra={"epoch": 3})
    loaded, ckpt = load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(tiny_model(x), loaded(x), atol=1e-6)
    assert ckpt["epoch"] == 3 and ckpt["backbone"] == "densenet121"
