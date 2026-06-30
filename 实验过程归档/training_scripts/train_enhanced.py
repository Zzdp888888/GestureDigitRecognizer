# -*- coding: utf-8 -*-
"""Train an enhanced MobileNetV3 model for high-resolution gesture photos."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights

from dataset import CustomDataset
from enhanced_model import ENHANCED_MODEL_PATH, build_enhanced_model, eval_transform, train_transform


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "enhanced_metrics.json"


def accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for sample in dataloader:
            x = sample["image"].to(device)
            y = sample["label"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
            loss_total += loss.item() * y.size(0)
    return correct / total, loss_total / total, correct, total


def main():
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    torch.set_num_threads(4)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = CustomDataset("./images/train.txt", "./images/train", train_transform)
    test_dataset = CustomDataset("./images/test.txt", "./images/test", eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = build_enhanced_model(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.to(device)

    # First train only the classifier head, then fine-tune the full network.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_acc = 0.0
    best_state = None
    epochs = 14
    unfreeze_epoch = 4
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - unfreeze_epoch)

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch, sample in enumerate(train_loader):
            x = sample["image"].to(device)
            y = sample["label"].to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(x)
                loss = loss_fn(pred, y)

            optimizer.zero_grad()
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
        test_acc, test_loss, test_correct, test_total = accuracy(model, test_loader, device)
        print(
            f"Epoch {epoch:02d} train_loss {total_loss / total:.4f} "
            f"train_acc {train_acc:.2%} test_loss {test_loss:.4f} test_acc {test_acc:.2%}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ENHANCED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": best_state,
                    "accuracy": best_acc,
                    "epoch": epoch,
                    "input_size": 224,
                    "model": "mobilenet_v3_small",
                    "test_correct": test_correct,
                    "test_total": test_total,
                },
                ENHANCED_MODEL_PATH,
            )
            print(f"Saved best enhanced model: {ENHANCED_MODEL_PATH} ({best_acc:.2%})")

    metrics = {
        "model": "mobilenet_v3_small",
        "input_size": 224,
        "best_accuracy": round(best_acc, 4),
        "model_path": str(ENHANCED_MODEL_PATH),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
