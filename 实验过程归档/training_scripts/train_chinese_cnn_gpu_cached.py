# -*- coding: utf-8 -*-
"""Fast guide-style CNN training on Chinese number gestures.

This follows the original project's txt dataset format, but caches 64x64
tensors on GPU for much higher throughput. Batch size defaults to 256.
"""

from __future__ import annotations

import json
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
MODEL_PATH = ROOT / "models" / "chinese_cnn_0_9_test.pkl"
OUTPUT_METRICS = ROOT / "outputs" / "chinese_cnn_gpu_cached_metrics.json"


def read_txt(path: Path) -> list[tuple[Path, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        rows.append((ROOT / image_path.replace("./", ""), int(label)))
    return rows


def load_tensors(rows: list[tuple[Path, int]], device: torch.device):
    transform = ToTensor()
    xs, ys = [], []
    for path, label in rows:
        image = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        xs.append(transform(image))
        ys.append(label)
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand((), device=x.device).item() < 0.5:
        x = torch.flip(x, dims=[3])
    brightness = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.86, 1.14)
    contrast = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.88, 1.14)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x * brightness + torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-0.025, 0.025)
    return torch.clamp(x + torch.randn_like(x) * 0.01, 0.0, 1.0)


@torch.no_grad()
def evaluate(model, x, y, batch_size=512):
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    started = time.time()
    torch.manual_seed(20260629)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_rows = read_txt(TRAIN_TXT)
    test_rows = read_txt(TEST_TXT)
    print(f"Train samples: {len(train_rows)}", flush=True)
    print(f"Test samples: {len(test_rows)}", flush=True)
    print("Caching 64x64 tensors on GPU...", flush=True)
    x_train, y_train = load_tensors(train_rows, device)
    x_test, y_test = load_tensors(test_rows, device)

    model = CustomNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    scaler = torch.amp.GradScaler("cuda")
    batch_size = 256
    epochs = 80
    best_acc = 0.0
    best_report = {}

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
                loss = F.cross_entropy(pred, yb, label_smoothing=0.03)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (pred.argmax(1) == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
        scheduler.step()

        train_acc = correct / x_train.size(0)
        test_acc, test_loss, test_correct, test_total, per_class = evaluate(model, x_test, y_test)
        print(
            f"Epoch {epoch:03d}/{epochs} train_loss {total_loss / x_train.size(0):.4f} "
            f"train_acc {train_acc:.2%} test_loss {test_loss:.4f} test_acc {test_acc:.2%}",
            flush=True,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_report = {
                "model": "custom_cnn_chinese_0_9",
                "accuracy": round(test_acc, 4),
                "epoch": epoch,
                "test_correct": test_correct,
                "test_total": test_total,
                "per_class": per_class,
                "batch_size": batch_size,
                "model_path": str(MODEL_PATH),
            }
            torch.save(model, MODEL_PATH)
            print(f"Saved best Chinese CNN: {MODEL_PATH} ({test_acc:.2%})", flush=True)

    best_report["elapsed_seconds"] = round(time.time() - started, 2)
    OUTPUT_METRICS.write_text(json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
