# -*- coding: utf-8 -*-
"""Train a Chinese number gesture extension model from the Gitee dataset.

Source:
    extra_data/Chinese-number-gestures-recognition

The guide dataset only covers 0-5. This script is the safer route for 6-9
because the dataset uses Chinese number gestures instead of ASL digit signs.
"""

from __future__ import annotations

import json
import random
import time
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from enhanced_model import ENHANCED_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD, build_enhanced_model


ROOT = Path(__file__).resolve().parent
CHINESE_ROOT = (
    ROOT
    / "extra_data"
    / "Chinese-number-gestures-recognition"
    / "digital_gesture_recognition"
    / "resized_img_split"
)
OUTPUT_DIR = ROOT / "outputs"
MANIFEST_PATH = ROOT / "extra_data" / "chinese_gesture_manifest_0_9.csv"
IMAGE_SIZE = 224
CLASS_COUNT = 10


def chinese_samples(max_label: int = 9) -> list[tuple[Path, int]]:
    if MANIFEST_PATH.exists():
        samples = []
        with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                label = int(row["label"])
                if 0 <= label <= max_label:
                    samples.append((ROOT / row["path"], label))
        return samples

    samples = []
    for path in CHINESE_ROOT.rglob("*.jpg"):
        try:
            label = int(path.name.split("_", 1)[0])
        except ValueError:
            continue
        if 0 <= label <= max_label:
            samples.append((path, label))
    return samples


def stratified_split(samples: list[tuple[Path, int]], val_ratio: float = 0.15, seed: int = 20260629):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for path, label in samples:
        by_label[label].append((path, label))
    train, val = [], []
    for label, rows in by_label.items():
        rows = rows[:]
        rng.shuffle(rows)
        val_count = max(100, int(len(rows) * val_ratio))
        val.extend(rows[:val_count])
        train.extend(rows[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class ChineseGestureDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], train: bool):
        self.samples = samples
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.76, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.16, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    brightness = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.82, 1.18)
    contrast = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.86, 1.18)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x * brightness + torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-0.045, 0.045)
    return x + torch.randn_like(x) * 0.012


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
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
    return correct / total, correct, total, per_class


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not CHINESE_ROOT.exists():
        raise FileNotFoundError(CHINESE_ROOT)

    started = time.time()
    random.seed(20260629)
    torch.manual_seed(20260629)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    all_samples = chinese_samples(max_label=9)
    train_samples, val_samples = stratified_split(all_samples)
    print(f"All samples: {len(all_samples)} {dict(Counter(label for _, label in all_samples))}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    train_loader = DataLoader(
        ChineseGestureDataset(train_samples, train=True),
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ChineseGestureDataset(val_samples, train=False),
        batch_size=192,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.0
    best_per_class = {}
    epochs = 18
    batch_size = 128

    for epoch in range(1, epochs + 1):
        if epoch == 4:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 3)

        model.train()
        correct = 0
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = gpu_augment(xb.to(device, non_blocking=True))
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred = model(xb)
                loss = F.cross_entropy(pred, yb, label_smoothing=0.03)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (pred.argmax(1) == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
            total_seen += yb.size(0)

        scheduler.step()
        train_acc = correct / total_seen
        val_acc, val_correct, val_total, per_class = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{epochs} train_loss {total_loss / total_seen:.4f} "
            f"train_acc {train_acc:.2%} val_acc {val_acc:.2%}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_per_class = per_class
            torch.save(
                {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "accuracy": best_acc,
                    "epoch": epoch,
                    "input_size": IMAGE_SIZE,
                    "model": "mobilenet_v3_small_chinese_0_9",
                    "val_correct": val_correct,
                    "val_total": val_total,
                    "per_class": per_class,
                    "dataset": "Chinese-number-gestures-recognition resized_img_split",
                },
                ENHANCED_MODEL_PATH,
            )
            print(f"Saved best Chinese 0-9 model: {ENHANCED_MODEL_PATH} ({best_acc:.2%})")

    metrics = {
        "model": "mobilenet_v3_small_chinese_0_9",
        "best_val_accuracy": round(best_acc, 4),
        "per_class": best_per_class,
        "model_path": str(ENHANCED_MODEL_PATH),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "chinese_0_9_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
