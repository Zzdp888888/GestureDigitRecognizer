# -*- coding: utf-8 -*-
"""GPU-cached enhanced training with low CPU usage.

This script is designed for this local project size: it loads all 4320 training
images once, moves tensors to CUDA, and then trains from GPU memory. That avoids
heavy per-batch PIL/torchvision CPU augmentation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from enhanced_model import ENHANCED_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD, build_enhanced_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def read_annotations(path: Path):
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        samples.append((ROOT / image_path.replace("./", ""), int(label)))
    return samples


def load_tensor_dataset(annotation_file: Path, device: torch.device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    images = []
    labels = []
    for image_path, label in read_annotations(annotation_file):
        image = Image.open(image_path).convert("RGB")
        images.append(transform(image))
        labels.append(label)
    x = torch.stack(images).to(device, non_blocking=True)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    # Cheap GPU-side augmentation: brightness/contrast jitter and small noise.
    brightness = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.88, 1.12)
    shift = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-0.06, 0.06)
    noise = torch.randn_like(x) * 0.015
    return x * brightness + shift + noise


@torch.no_grad()
def evaluate(model, x, y, batch_size=256):
    model.eval()
    correct = 0
    total_loss = 0.0
    for start in range(0, x.size(0), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        pred = model(xb)
        total_loss += F.cross_entropy(pred, yb).item() * yb.size(0)
        correct += (pred.argmax(1) == yb).sum().item()
    return correct / y.size(0), total_loss / y.size(0), correct, y.size(0)


def main():
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu-cached training.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    print("Caching tensors on GPU...")
    x_train, y_train = load_tensor_dataset(ROOT / "images" / "train.txt", device)
    x_test, y_test = load_tensor_dataset(ROOT / "images" / "test.txt", device)
    print(f"Train tensor: {tuple(x_train.shape)}")
    print(f"Test tensor: {tuple(x_test.shape)}")

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    epochs = 16
    batch_size = 192

    for epoch in range(1, epochs + 1):
        if epoch == 4:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 3)

        model.train()
        order = torch.randperm(x_train.size(0), device=device)
        total_loss = 0.0
        correct = 0
        for start in range(0, x_train.size(0), batch_size):
            idx = order[start : start + batch_size]
            xb = gpu_augment(x_train[idx])
            yb = y_train[idx]
            with torch.amp.autocast("cuda"):
                pred = model(xb)
                loss = F.cross_entropy(pred, yb, label_smoothing=0.04)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * yb.size(0)
            correct += (pred.argmax(1) == yb).sum().item()

        scheduler.step()
        train_acc = correct / y_train.size(0)
        test_acc, test_loss, test_correct, test_total = evaluate(model, x_test, y_test)
        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss {total_loss / y_train.size(0):.4f} train_acc {train_acc:.2%} "
            f"test_loss {test_loss:.4f} test_acc {test_acc:.2%}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ENHANCED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "accuracy": best_acc,
                    "epoch": epoch,
                    "input_size": 224,
                    "model": "mobilenet_v3_small_gpu_cached",
                    "test_correct": test_correct,
                    "test_total": test_total,
                },
                ENHANCED_MODEL_PATH,
            )
            print(f"Saved best enhanced model: {ENHANCED_MODEL_PATH} ({best_acc:.2%})")

    metrics = {
        "model": "mobilenet_v3_small_gpu_cached",
        "input_size": 224,
        "best_accuracy": round(best_acc, 4),
        "model_path": str(ENHANCED_MODEL_PATH),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "enhanced_gpu_cached_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
