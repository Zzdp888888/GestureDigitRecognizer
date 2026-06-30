# -*- coding: utf-8 -*-
"""Train MobileNetV3 with original data plus external sign digit dataset."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from enhanced_model import ENHANCED_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD, build_enhanced_model


ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "extra_data" / "Sign-Language-Digits-Dataset-master" / "Dataset"
OUTPUT_DIR = ROOT / "outputs"


class MixedGestureDataset(Dataset):
    def __init__(self, original_annotation: Path, external_root: Path, transform):
        self.samples: list[tuple[Path, int, str]] = []
        self.transform = transform
        for line in original_annotation.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path, label = line.split()
            self.samples.append((ROOT / image_path.replace("./", ""), int(label), "original"))
        for label_dir in sorted(external_root.iterdir()):
            if not label_dir.is_dir() or not label_dir.name.isdigit():
                continue
            label = int(label_dir.name)
            for path in sorted(label_dir.glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((path, label, "external"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, source = self.samples[index]
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "label": label, "source": source}


class AnnotationDataset(Dataset):
    def __init__(self, annotation_file: Path, transform):
        self.samples = []
        self.transform = transform
        for line in annotation_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path, label = line.split()
            self.samples.append((ROOT / image_path.replace("./", ""), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "label": label}


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.68, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.22, hue=0.03),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    per_class = {i: [0, 0] for i in range(10)}
    for sample in loader:
        x = sample["image"].to(device)
        y = sample["label"].to(device)
        pred = model(x)
        total_loss += F.cross_entropy(pred, y).item() * y.size(0)
        predicted = pred.argmax(1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
        for gt, pd in zip(y.tolist(), predicted.tolist()):
            per_class[gt][1] += 1
            per_class[gt][0] += int(gt == pd)
    return correct / total, total_loss / total, correct, total, per_class


def main():
    if not EXTERNAL_ROOT.exists():
        raise FileNotFoundError(f"External dataset not found: {EXTERNAL_ROOT}")
    started = time.time()
    random.seed(2026)
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    torch.set_num_threads(4)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = MixedGestureDataset(ROOT / "images" / "train.txt", EXTERNAL_ROOT, train_transform())
    test_dataset = AnnotationDataset(ROOT / "images" / "test.txt", eval_transform())
    print(f"Mixed train samples: {len(train_dataset)}")
    print(f"Original test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_acc = 0.0
    epochs = 18
    for epoch in range(1, epochs + 1):
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 4)

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch, sample in enumerate(train_loader):
            x = sample["image"].to(device, non_blocking=True)
            y = sample["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(x)
                loss = F.cross_entropy(pred, y, label_smoothing=0.04)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
            if batch % 40 == 0:
                print(f"Epoch {epoch:02d}/{epochs} batch {batch:03d} loss {loss.item():.4f}")

        scheduler.step()
        train_acc = correct / total
        test_acc, test_loss, test_correct, test_total, per_class = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{epochs} train_loss {total_loss / total:.4f} "
            f"train_acc {train_acc:.2%} test_loss {test_loss:.4f} test_acc {test_acc:.2%}"
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
                    "model": "mobilenet_v3_small_external_mix",
                    "test_correct": test_correct,
                    "test_total": test_total,
                    "external_dataset": "ardamavi/Sign-Language-Digits-Dataset",
                    "per_class": per_class,
                },
                ENHANCED_MODEL_PATH,
            )
            print(f"Saved best external-mix model: {ENHANCED_MODEL_PATH} ({best_acc:.2%})")

    metrics = {
        "model": "mobilenet_v3_small_external_mix",
        "best_accuracy": round(best_acc, 4),
        "model_path": str(ENHANCED_MODEL_PATH),
        "external_dataset": "https://github.com/ardamavi/Sign-Language-Digits-Dataset",
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "external_mix_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
