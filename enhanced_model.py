# -*- coding: utf-8 -*-
"""Enhanced transfer-learning model for high-resolution gesture images."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ENHANCED_MODEL_PATH = Path(__file__).resolve().parent / "models" / "mobilenetv3_gesture.pth"
IMAGE_SIZE = 224
CLASS_COUNT = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_enhanced_model(weights=None):
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, CLASS_COUNT)
    return model


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.72, 1.0), ratio=(0.88, 1.12)),
            transforms.RandomRotation(16),
            transforms.ColorJitter(brightness=0.24, contrast=0.24, saturation=0.18, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_enhanced_model(device: torch.device):
    checkpoint = torch.load(ENHANCED_MODEL_PATH, map_location=device, weights_only=False)
    model = build_enhanced_model(weights=None)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def crop_high_res_hand_region(image: Image.Image) -> Image.Image:
    """Crop a plausible hand region from high-resolution photos.

    Without an external hand detector this uses a conservative center crop:
    it removes peripheral background while preserving the central hand area.
    The frontend can still upload full-resolution photos; this function avoids
    shrinking the whole scene directly into the classifier.
    """
    image = image.convert("RGB")
    width, height = image.size
    side = int(min(width, height) * 0.88)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    return image.crop((left, top, left + side, top + side))
