# -*- coding: utf-8 -*-
"""Train a 224px MobileNetV3 Chinese gesture model on grouped train/test txt.

This is the practical high-resolution route for webcam/photo recognition.
It keeps the guide CNN models untouched and saves a separate checkpoint.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from enhanced_model import IMAGENET_MEAN, IMAGENET_STD, build_enhanced_model


ROOT = Path(__file__).resolve().parent
TRAIN_TXT = ROOT / "images" / "chinese" / "train.txt"
TEST_TXT = ROOT / "images" / "chinese" / "test.txt"
MODEL_PATH = ROOT / "models" / "chinese_mobilenetv3_grouped_0_9.pth"
METRICS_PATH = ROOT / "outputs" / "chinese_mobilenetv3_grouped_metrics.json"
IMAGE_SIZE = 224


class TxtGestureDataset(Dataset):
    def __init__(self, txt_path: Path, transform):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path, label = line.split()
            self.samples.append((ROOT / image_path.removeprefix("./"), int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "label": label}


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.62, 1.0), ratio=(0.78, 1.24)),
            transforms.RandomAffine(degrees=18, translate=(0.12, 0.12), scale=(0.82, 1.15), shear=(-6, 6)),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.22, hue=0.025),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4))], p=0.22),
            transforms.RandomHorizontalFlip(p=0.10),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.20, scale=(0.015, 0.08), ratio=(0.45, 2.2), value="random"),
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    for sample in loader:
        x = sample["image"].to(device, non_blocking=True)
        y = sample["label"].to(device, non_blocking=True)
        pred = model(x)
        total_loss += F.cross_entropy(pred, y).item() * y.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.argmax(1).detach().cpu().tolist())

    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    per_total = Counter(y_true)
    per_correct = Counter(a for a, b in zip(y_true, y_pred) if a == b)
    per_class = {
        str(label): {
            "correct": per_correct[label],
            "total": per_total[label],
            "accuracy": round(per_correct[label] / per_total[label], 4),
        }
        for label in sorted(per_total)
    }
    return correct / total, total_loss / total, correct, total, per_class


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    started = time.time()
    random.seed(20260630)
    torch.manual_seed(20260630)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    device = torch.device("cuda")

    train_dataset = TxtGestureDataset(TRAIN_TXT, train_transform())
    test_dataset = TxtGestureDataset(TEST_TXT, eval_transform())
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)} {dict(sorted(Counter(label for _, label in train_dataset.samples).items()))}")
    print(f"Test samples: {len(test_dataset)} {dict(sorted(Counter(label for _, label in test_dataset.samples).items()))}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=160, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.2e-3, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cuda")
    epochs = 32
    best_acc = 0.0
    best_report = {}
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1.5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 4, eta_min=8e-6)

        model.train()
        correct = 0
        total_seen = 0
        total_loss = 0.0
        for batch, sample in enumerate(train_loader):
            x = sample["image"].to(device, non_blocking=True)
            y = sample["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = F.cross_entropy(pred, y, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (pred.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)
            total_seen += y.size(0)
            if batch % 60 == 0:
                print(f"Epoch {epoch:02d}/{epochs} batch {batch:03d} loss {loss.item():.4f}", flush=True)

        scheduler.step()
        train_acc = correct / total_seen
        test_acc, test_loss, test_correct, test_total, per_class = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{epochs} train_loss {total_loss / total_seen:.4f} "
            f"train_acc {train_acc:.2%} test_loss {test_loss:.4f} test_acc {test_acc:.2%}",
            flush=True,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            stale_epochs = 0
            best_report = {
                "model": "mobilenet_v3_small_chinese_grouped_0_9",
                "accuracy": round(best_acc, 4),
                "epoch": epoch,
                "test_correct": test_correct,
                "test_total": test_total,
                "per_class": per_class,
                "input_size": IMAGE_SIZE,
                "split": "source_grouped",
                "model_path": str(MODEL_PATH),
            }
            torch.save(
                {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    **best_report,
                },
                MODEL_PATH,
            )
            print(f"Saved best grouped MobileNetV3: {MODEL_PATH} ({best_acc:.2%})", flush=True)
        else:
            stale_epochs += 1

        if epoch >= 14 and stale_epochs >= 8:
            print(f"Early stop: no test improvement for {stale_epochs} epochs.", flush=True)
            break

    best_report["elapsed_seconds"] = round(time.time() - started, 2)
    METRICS_PATH.write_text(json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
