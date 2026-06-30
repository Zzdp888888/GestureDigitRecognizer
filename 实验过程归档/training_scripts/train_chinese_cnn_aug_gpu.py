# -*- coding: utf-8 -*-
"""Train guide-style CNN on Chinese gestures with stronger augmentation.

This script keeps the practical guide's 64x64 CustomNet route, but fixes the
main evaluation issue by using images/chinese/train.txt and test.txt generated
with source-group splitting. Training data is cached on GPU and augmented
online to better match webcam scenes.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from model import CustomNet


ROOT = Path(__file__).resolve().parent
TRAIN_TXT = ROOT / "images" / "chinese" / "train.txt"
TEST_TXT = ROOT / "images" / "chinese" / "test.txt"
MODEL_PATH = ROOT / "models" / "chinese_cnn_0_9_aug.pkl"
OUTPUT_METRICS = ROOT / "outputs" / "chinese_cnn_aug_metrics.json"


def read_txt(path: Path) -> list[tuple[Path, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        rows.append((ROOT / image_path.removeprefix("./"), int(label)))
    return rows


def load_tensors(rows: list[tuple[Path, int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    transform = ToTensor()
    xs, ys = [], []
    for path, label in rows:
        image = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        xs.append(transform(image))
        ys.append(label)
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def affine_grid_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    angle = torch.empty(batch_size, device=device).uniform_(-18.0, 18.0) * math.pi / 180.0
    scale = torch.empty(batch_size, device=device).uniform_(0.82, 1.12)
    tx = torch.empty(batch_size, device=device).uniform_(-0.16, 0.16)
    ty = torch.empty(batch_size, device=device).uniform_(-0.16, 0.16)
    cos_a = torch.cos(angle) * scale
    sin_a = torch.sin(angle) * scale
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    return theta


def random_erasing(x: torch.Tensor, p: float = 0.28) -> torch.Tensor:
    batch, _, height, width = x.shape
    for i in range(batch):
        if torch.rand((), device=x.device).item() >= p:
            continue
        erase_w = int(torch.randint(max(3, width // 10), max(4, width // 4), (), device=x.device).item())
        erase_h = int(torch.randint(max(3, height // 10), max(4, height // 4), (), device=x.device).item())
        left = int(torch.randint(0, max(1, width - erase_w), (), device=x.device).item())
        top = int(torch.randint(0, max(1, height - erase_h), (), device=x.device).item())
        fill = x[i].mean(dim=(1, 2), keepdim=True)
        x[i, :, top : top + erase_h, left : left + erase_w] = fill
    return x


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    batch = x.size(0)
    theta = affine_grid_batch(batch, x.device)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # Horizontal flip is intentionally low probability; some digit gestures are
    # asymmetric, but webcam mirroring can still happen.
    flip_mask = torch.rand(batch, device=x.device) < 0.12
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    brightness = torch.empty((batch, 1, 1, 1), device=x.device).uniform_(0.72, 1.28)
    contrast = torch.empty((batch, 1, 1, 1), device=x.device).uniform_(0.72, 1.30)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x * brightness
    x = x + torch.empty((batch, 1, 1, 1), device=x.device).uniform_(-0.055, 0.055)
    x = torch.clamp(x + torch.randn_like(x) * 0.018, 0.0, 1.0)

    if torch.rand((), device=x.device).item() < 0.35:
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    return random_erasing(torch.clamp(x, 0.0, 1.0))


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 768):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    for start in range(0, x.size(0), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        pred = model(xb)
        total_loss += F.cross_entropy(pred, yb).item() * yb.size(0)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(pred.argmax(1).detach().cpu().tolist())
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    per_total = Counter(y_true)
    per_correct = Counter(a for a, b in zip(y_true, y_pred) if a == b)
    confusion = defaultdict(Counter)
    for a, b in zip(y_true, y_pred):
        confusion[a][b] += 1
    per_class = {
        str(label): {
            "correct": per_correct[label],
            "total": per_total[label],
            "accuracy": round(per_correct[label] / per_total[label], 4),
            "top_predictions": dict(confusion[label].most_common(5)),
        }
        for label in sorted(per_total)
    }
    return correct / total, total_loss / total, correct, total, per_class


def class_weights(y_train: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y_train, minlength=10).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * counts.numel())
    return (weights / weights.mean()).to(y_train.device)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    started = time.time()
    torch.manual_seed(20260630)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_rows = read_txt(TRAIN_TXT)
    test_rows = read_txt(TEST_TXT)
    print(f"Train samples: {len(train_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Train labels: {dict(sorted(Counter(label for _, label in train_rows).items()))}")
    print(f"Test labels: {dict(sorted(Counter(label for _, label in test_rows).items()))}", flush=True)

    print("Caching 64x64 tensors on GPU...", flush=True)
    x_train, y_train = load_tensors(train_rows, device)
    x_test, y_test = load_tensors(test_rows, device)
    weights = class_weights(y_train)

    model = CustomNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    batch_size = 256
    epochs = 120
    best_acc = 0.0
    best_report: dict = {}
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
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
                loss = F.cross_entropy(pred, yb, weight=weights, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            scaler.step(optimizer)
            scaler.update()
            correct += (pred.argmax(1) == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
        scheduler.step()

        train_acc = correct / x_train.size(0)
        test_acc, test_loss, test_correct, test_total, per_class = evaluate(model, x_test, y_test)
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{epochs} lr {lr:.6f} train_loss {total_loss / x_train.size(0):.4f} "
            f"train_acc {train_acc:.2%} test_loss {test_loss:.4f} test_acc {test_acc:.2%}",
            flush=True,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            stale_epochs = 0
            best_report = {
                "model": "custom_cnn_chinese_0_9_aug",
                "accuracy": round(test_acc, 4),
                "epoch": epoch,
                "test_correct": test_correct,
                "test_total": test_total,
                "per_class": per_class,
                "batch_size": batch_size,
                "epochs": epochs,
                "split": "source_grouped",
                "augmentation": [
                    "affine_rotate_translate_scale",
                    "brightness_contrast_noise",
                    "light_blur",
                    "random_erasing",
                    "low_probability_horizontal_flip",
                ],
                "model_path": str(MODEL_PATH),
            }
            torch.save(model, MODEL_PATH)
            print(f"Saved best augmented Chinese CNN: {MODEL_PATH} ({test_acc:.2%})", flush=True)
        else:
            stale_epochs += 1

        if epoch >= 45 and stale_epochs >= 24:
            print(f"Early stop: no test improvement for {stale_epochs} epochs.", flush=True)
            break

    best_report["elapsed_seconds"] = round(time.time() - started, 2)
    OUTPUT_METRICS.write_text(json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
