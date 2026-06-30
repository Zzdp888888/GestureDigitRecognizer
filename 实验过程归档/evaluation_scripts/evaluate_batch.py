# -*- coding: utf-8 -*-
"""Fast batch evaluation for direct-input and ROI-input model diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from enhanced_model import eval_transform, load_enhanced_model
from hand_detector import detect_hand_region


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_original(split: str) -> list[tuple[Path, int]]:
    rows = []
    for line in (ROOT / "images" / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        rows.append((ROOT / image_path.replace("./", ""), int(label)))
    return rows


def load_external() -> list[tuple[Path, int]]:
    root = ROOT / "extra_data" / "Sign-Language-Digits-Dataset-master" / "Dataset"
    rows = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue
        label = int(label_dir.name)
        for path in sorted(label_dir.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((path, label))
    return rows


def load_synthetic() -> list[tuple[Path, int]]:
    rows = []
    for line in (ROOT / "outputs" / "synthetic_clutter" / "labels.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        rows.append((ROOT / image_path, int(label)))
    return rows


def summarize(y_true: list[int], y_pred: list[int]) -> dict:
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    per_total = Counter(y_true)
    per_correct = Counter(a for a, b in zip(y_true, y_pred) if a == b)
    confusion = defaultdict(Counter)
    for a, b in zip(y_true, y_pred):
        confusion[a][b] += 1
    per_class = {}
    for label in sorted(per_total):
        per_class[str(label)] = {
            "correct": per_correct[label],
            "total": per_total[label],
            "accuracy": round(per_correct[label] / per_total[label], 4),
            "top_predictions": dict(confusion[label].most_common(5)),
        }
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_class": per_class,
    }


def eval_legacy(samples: list[tuple[Path, int]], roi: bool, batch_size: int) -> dict:
    model = torch.load(ROOT / "models" / "model.pkl", map_location=DEVICE, weights_only=False)
    model.to(DEVICE).eval()
    y_true, y_pred = [], []
    transform = ToTensor()
    xs, ys = [], []
    with torch.no_grad():
        for path, label in samples:
            image = Image.open(path).convert("RGB")
            if roi:
                image = detect_hand_region(image).image
            xs.append(transform(image.resize((64, 64))))
            ys.append(label)
            if len(xs) >= batch_size:
                xb = torch.stack(xs).to(DEVICE)
                pred = model(xb).argmax(1).detach().cpu().tolist()
                y_pred.extend(pred)
                y_true.extend(ys)
                xs, ys = [], []
        if xs:
            xb = torch.stack(xs).to(DEVICE)
            pred = model(xb).argmax(1).detach().cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(ys)
    result = summarize(y_true, y_pred)
    result["model"] = "legacy"
    result["roi"] = roi
    return result


def eval_enhanced(samples: list[tuple[Path, int]], roi: bool, batch_size: int) -> dict:
    model, checkpoint = load_enhanced_model(DEVICE)
    transform = eval_transform()
    y_true, y_pred = [], []
    xs, ys = [], []
    with torch.no_grad():
        for path, label in samples:
            image = Image.open(path).convert("RGB")
            if roi:
                image = detect_hand_region(image).image
            xs.append(transform(image))
            ys.append(label)
            if len(xs) >= batch_size:
                xb = torch.stack(xs).to(DEVICE)
                pred = model(xb).argmax(1).detach().cpu().tolist()
                y_pred.extend(pred)
                y_true.extend(ys)
                xs, ys = [], []
        if xs:
            xb = torch.stack(xs).to(DEVICE)
            pred = model(xb).argmax(1).detach().cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(ys)
    result = summarize(y_true, y_pred)
    result["model"] = checkpoint.get("model", "enhanced")
    result["roi"] = roi
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original-test", "external", "synthetic"], default="original-test")
    parser.add_argument("--model", choices=["legacy", "enhanced"], default="legacy")
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.dataset == "original-test":
        samples = load_original("test")
    elif args.dataset == "external":
        samples = load_external()
    else:
        samples = load_synthetic()
    if args.limit:
        samples = samples[: args.limit]

    result = eval_legacy(samples, args.roi, args.batch_size) if args.model == "legacy" else eval_enhanced(samples, args.roi, args.batch_size)
    result["dataset"] = args.dataset
    result["sample_count"] = len(samples)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
