# -*- coding: utf-8 -*-
"""Train a dedicated 0-9 extension model on the external digit dataset.

The guide dataset only contains labels 0-5. This script keeps the guide CNN as
the stable 0-5 baseline and trains MobileNetV3 for the extended 0-9 scenario.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from enhanced_model import ENHANCED_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD, build_enhanced_model


ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "extra_data" / "Sign-Language-Digits-Dataset-master" / "Dataset"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_SIZE = 224


def external_samples(root: Path) -> list[tuple[Path, int]]:
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue
        label = int(label_dir.name)
        for path in sorted(label_dir.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, label))
    return samples


def stratified_split(samples: list[tuple[Path, int]], val_ratio: float = 0.18, seed: int = 20260629):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for path, label in samples:
        by_label[label].append((path, label))
    train, val = [], []
    for label, rows in by_label.items():
        rows = rows[:]
        rng.shuffle(rows)
        val_count = max(24, int(len(rows) * val_ratio))
        val.extend(rows[:val_count])
        train.extend(rows[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_tensors(samples: list[tuple[Path, int]], device: torch.device, image_size: int = 256):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    xs, ys = [], []
    for path, label in samples:
        image = Image.open(path).convert("RGB")
        xs.append(transform(image))
        ys.append(label)
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def random_crop_224(x: torch.Tensor) -> torch.Tensor:
    if x.size(-1) == IMAGE_SIZE:
        return x
    max_offset = x.size(-1) - IMAGE_SIZE
    top = torch.randint(0, max_offset + 1, (1,), device=x.device).item()
    left = torch.randint(0, max_offset + 1, (1,), device=x.device).item()
    return x[:, :, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]


def center_crop_224(x: torch.Tensor) -> torch.Tensor:
    top = (x.size(-2) - IMAGE_SIZE) // 2
    left = (x.size(-1) - IMAGE_SIZE) // 2
    return x[:, :, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    x = random_crop_224(x)
    brightness = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.82, 1.18)
    contrast = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.88, 1.16)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x * brightness + torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-0.04, 0.04)
    x = x + torch.randn_like(x) * 0.012
    return x


@torch.no_grad()
def evaluate(model, x, y, batch_size=256):
    model.eval()
    y_true, y_pred = [], []
    for start in range(0, x.size(0), batch_size):
        xb = center_crop_224(x[start : start + batch_size])
        yb = y[start : start + batch_size]
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not EXTERNAL_ROOT.exists():
        raise FileNotFoundError(EXTERNAL_ROOT)

    started = time.time()
    random.seed(20260629)
    torch.manual_seed(20260629)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    train_samples, val_samples = stratified_split(external_samples(EXTERNAL_ROOT))
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print("Caching tensors on GPU...")
    x_train, y_train = load_tensors(train_samples, device)
    x_val, y_val = load_tensors(val_samples, device)

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.0
    best_per_class = {}
    epochs = 24
    batch_size = 160

    for epoch in range(1, epochs + 1):
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-4, weight_decay=1.2e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 4)

        model.train()
        order = torch.randperm(x_train.size(0), device=device)
        correct = 0
        total_loss = 0.0
        for start in range(0, x_train.size(0), batch_size):
            idx = order[start : start + batch_size]
            xb = gpu_augment(x_train[idx])
            yb = y_train[idx]
            with torch.amp.autocast("cuda"):
                pred = model(xb)
                loss = F.cross_entropy(pred, yb, label_smoothing=0.03)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (pred.argmax(1) == yb).sum().item()
            total_loss += loss.item() * yb.size(0)

        scheduler.step()
        train_acc = correct / y_train.size(0)
        val_acc, val_correct, val_total, per_class = evaluate(model, x_val, y_val)
        print(
            f"Epoch {epoch:02d}/{epochs} train_loss {total_loss / y_train.size(0):.4f} "
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
                    "model": "mobilenet_v3_small_external10_split",
                    "val_correct": val_correct,
                    "val_total": val_total,
                    "per_class": per_class,
                    "external_dataset": "ardamavi/Sign-Language-Digits-Dataset",
                },
                ENHANCED_MODEL_PATH,
            )
            print(f"Saved best 0-9 extension model: {ENHANCED_MODEL_PATH} ({best_acc:.2%})")

    metrics = {
        "model": "mobilenet_v3_small_external10_split",
        "best_val_accuracy": round(best_acc, 4),
        "per_class": best_per_class,
        "model_path": str(ENHANCED_MODEL_PATH),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "external10_split_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
