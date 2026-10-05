"""
network.py - The LungLens neural network and image preprocessing.

Big idea: transfer learning
---------------------------
Training a good image model from scratch needs hundreds of thousands of
images. We only have about 5,000. So we start from a DenseNet121 network
that the open-source TorchXRayVision project already trained on hundreds of
thousands of (adult) chest X-rays from public hospital datasets
(NIH, CheXpert, MIMIC, PadChest and others). That network has already learned
what ribs, lungs, the heart and cloudy patches look like.

We keep its "feature extractor" (the convolutional layers that turn an image
into a 7x7 grid of 1024 features) and replace its final layer with our own
single output: "how likely is pneumonia?". Training then only has to adapt
the last part of the network to our task (see train.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224                 # the pretrained network expects 224x224 images
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class LungLensNet(nn.Module):
    """DenseNet121 feature extractor + one output neuron for pneumonia."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load the chest-X-ray-pretrained DenseNet121. The weights file
        # (about 28 MB) is downloaded from GitHub the first time.
        base = xrv.models.DenseNet(weights="densenet121-res224-all" if pretrained else None)

        # The convolutional part: image (1x224x224) -> features (1024x7x7).
        # Each of the 7x7 cells "looks at" one region of the X-ray.
        self.features = base.features

        # Our new final layer: 1024 numbers -> 1 number (the pneumonia score).
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # 1. Extract a 7x7 grid of features from the X-ray.
        feature_maps = F.relu(self.features(x))               # shape: (batch, 1024, 7, 7)
        # 2. Average each feature over the whole image ("global average pooling").
        pooled = F.adaptive_avg_pool2d(feature_maps, 1).flatten(1)   # (batch, 1024)
        # 3. Combine the 1024 averages into one score. This is a "logit":
        #    any real number; sigmoid(logit) turns it into a probability 0..1.
        logit = self.classifier(pooled).squeeze(1)            # (batch,)
        if return_features:
            # Grad-CAM (gradcam.py) needs the 7x7 feature maps as well.
            return logit, feature_maps
        return logit

    def freeze_early_layers(self) -> None:
        """
        Freeze everything except the last dense block, so training only
        updates the deepest layers and our new classifier. The early layers
        detect general things like edges and textures, which are already
        good. This also makes training fast enough to run on a laptop CPU.
        """
        for name, layer in self.features.named_children():
            trainable = name in ("denseblock4", "norm5")
            for p in layer.parameters():
                p.requires_grad = trainable


def to_model_range(img: torch.Tensor) -> torch.Tensor:
    """
    The pretrained network expects pixel values from -1024 to +1024
    (that mimics the range of real X-ray scanner values).
    ToTensor() gives us 0..1, so we stretch it to -1024..+1024.
    """
    return img * 2048.0 - 1024.0


# Preprocessing used for prediction (and for validation/testing):
# grayscale -> resize to 224x224 -> tensor -> rescale to the model's range.
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(to_model_range),
])

# Preprocessing used during training adds small random changes
# ("data augmentation") so the model sees a slightly different version of
# each X-ray every time. This helps it generalise instead of memorising.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(10),                          # tilt up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # vary exposure
    transforms.ToTensor(),
    transforms.Lambda(to_model_range),
])


def load_model(checkpoint_path: str = "models/lunglens.pt"):
    """Load the trained model and the decision threshold saved by train.py."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = LungLensNet(pretrained=False)   # weights come from our checkpoint instead
    model.load_state_dict(checkpoint["model_state"])
    model.eval()                            # switch off training-only behaviour
    return model, checkpoint["threshold"]


def preprocess(image: Image.Image) -> torch.Tensor:
    """Turn a PIL image into a batch of one tensor ready for the model."""
    return eval_transform(image.convert("RGB")).unsqueeze(0)
