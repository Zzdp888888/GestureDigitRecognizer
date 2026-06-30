# -*- coding: utf-8 -*-
"""Evaluate legacy/enhanced models with per-class accuracy.

This script is intentionally separate from the guide's `test.py` so the
original practical-project file stays simple while we can diagnose real
deployment issues such as missing 6-9 samples and class confusion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from app_server import enhanced_model, enhanced_probability, legacy_probability
from hand_detector import detect_hand_region


ROOT = Path(__file__).resolve().parent


def load_annotation(path: Path) -> list[tuple[Path, int]]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        samples.append((ROOT / image_path.replace("./", ""), int(label)))
    return samples


def load_external(root: Path) -> list[tuple[Path, int]]:
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue
        label = int(label_dir.name)
        for path in sorted(label_dir.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, label))
    return samples


def predict(image: Image.Image, mode: str) -> tuple[int, float, dict]:
    roi = detect_hand_region(image)
    with torch.no_grad():
        if mode == "legacy":
            probability = legacy_probability(roi.image)
        elif mode == "enhanced":
            if enhanced_model is None:
                raise RuntimeError("Enhanced model is not available.")
            probability = enhanced_probability(roi.image)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    confidence, predicted = torch.max(probability, 0)
    meta = {
        "detector": roi.detector,
        "hand_detected": roi.hand_detected,
        "segmentation_applied": roi.segmentation_applied,
        "mask_coverage": roi.mask_coverage,
    }
    return int(predicted.item()), float(confidence.item()), meta


def evaluate(samples: list[tuple[Path, int]], mode: str, limit: int | None = None) -> dict:
    if limit:
        samples = samples[:limit]
    total = len(samples)
    correct = 0
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)
    segmentation_count = 0
    detector_count = Counter()
    low_examples = []

    for path, label in samples:
        image = Image.open(path).convert("RGB")
        pred, confidence, meta = predict(image, mode)
        per_class_total[label] += 1
        detector_count[meta["detector"]] += 1
        segmentation_count += int(meta["segmentation_applied"])
        confusion[label][pred] += 1
        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        elif len(low_examples) < 20:
            low_examples.append(
                {
                    "path": str(path),
                    "label": label,
                    "pred": pred,
                    "confidence": round(confidence, 4),
                    "detector": meta["detector"],
                    "segmented": meta["segmentation_applied"],
                }
            )

    per_class = {}
    for label in sorted(per_class_total):
        per_class[str(label)] = {
            "correct": per_class_correct[label],
            "total": per_class_total[label],
            "accuracy": round(per_class_correct[label] / per_class_total[label], 4),
            "top_predictions": dict(confusion[label].most_common(5)),
        }

    return {
        "mode": mode,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_class": per_class,
        "detectors": dict(detector_count),
        "segmentation_rate": round(segmentation_count / total, 4) if total else 0.0,
        "wrong_examples": low_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original-test", "external", "synthetic"], default="original-test")
    parser.add_argument("--mode", choices=["legacy", "enhanced"], default="enhanced")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.dataset == "original-test":
        samples = load_annotation(ROOT / "images" / "test.txt")
    elif args.dataset == "external":
        samples = load_external(ROOT / "extra_data" / "Sign-Language-Digits-Dataset-master" / "Dataset")
    else:
        samples = load_annotation(ROOT / "outputs" / "synthetic_clutter" / "labels.txt")

    report = evaluate(samples, args.mode, args.limit)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
