# -*- coding: utf-8 -*-
"""GPU-cached training with original + external sign digit dataset."""

from __future__ import annotations

import json
import random
import time
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


def original_samples(annotation_file: Path):
    samples = []
    for line in annotation_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        samples.append((ROOT / image_path.replace("./", ""), int(label)))
    return samples


def external_samples(root: Path):
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue
        label = int(label_dir.name)
        for path in sorted(label_dir.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, label))
    return samples


def load_tensors(samples, device, image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    xs = []
    ys = []
    for path, label in samples:
        image = Image.open(path).convert("RGB")
        xs.append(transform(image))
        ys.append(label)
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def gpu_augment(x):
    brightness = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.9, 1.1)
    shift = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-0.05, 0.05)
    noise = torch.randn_like(x) * 0.012
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not EXTERNAL_ROOT.exists():
        raise FileNotFoundError(f"External dataset not found: {EXTERNAL_ROOT}")
    started = time.time()
    random.seed(2026)
    torch.manual_seed(2026)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    print(f"Device: {device}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    train_samples = original_samples(ROOT / "images" / "train.txt") + external_samples(EXTERNAL_ROOT)
    test_samples = original_samples(ROOT / "images" / "test.txt")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print("Caching tensors on GPU...")
    x_train, y_train = load_tensors(train_samples, device)
    x_test, y_test = load_tensors(test_samples, device)

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.0
    epochs = 14
    batch_size = 192

    for epoch in range(1, epochs + 1):
        if epoch == 4:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 3)

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
                loss = F.cross_entropy(pred, yb, label_smoothing=0.04)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (pred.argmax(1) == yb).sum().item()
            total_loss += loss.item() * yb.size(0)

        scheduler.step()
        train_acc = correct / y_train.size(0)
        test_acc, test_loss, test_correct, test_total = evaluate(model, x_test, y_test)
        print(
            f"Epoch {epoch:02d}/{epochs} train_loss {total_loss / y_train.size(0):.4f} "
            f"train_acc {train_acc:.2%} test_loss {test_loss:.4f} test_acc {test_acc:.2%}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "accuracy": best_acc,
                    "epoch": epoch,
                    "input_size": 224,
                    "model": "mobilenet_v3_small_external_gpu_cached",
                    "test_correct": test_correct,
                    "test_total": test_total,
                    "external_dataset": "ardamavi/Sign-Language-Digits-Dataset",
                },
                ENHANCED_MODEL_PATH,
            )
            print(f"Saved best external model: {ENHANCED_MODEL_PATH} ({best_acc:.2%})")

    metrics = {
        "model": "mobilenet_v3_small_external_gpu_cached",
        "best_accuracy": round(best_acc, 4),
        "model_path": str(ENHANCED_MODEL_PATH),
        "train_samples": len(train_samples),
        "external_samples": len(external_samples(EXTERNAL_ROOT)),
        "external_dataset": "https://github.com/ardamavi/Sign-Language-Digits-Dataset",
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "external_gpu_cached_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
