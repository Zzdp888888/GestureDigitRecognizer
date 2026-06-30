# -*- coding: utf-8 -*-
"""Generate a local cluttered-background stress set for gesture recognition."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "synthetic_clutter"


def read_samples(annotation_file: Path) -> list[tuple[Path, int]]:
    samples = []
    for line in annotation_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path, label = line.split()
        samples.append((ROOT / path.replace("./", ""), int(label)))
    return samples


def make_background(width: int, height: int, rng: random.Random) -> Image.Image:
    base = np.zeros((height, width, 3), dtype=np.uint8)
    color_a = np.array([rng.randint(35, 230), rng.randint(35, 230), rng.randint(35, 230)], dtype=np.float32)
    color_b = np.array([rng.randint(35, 230), rng.randint(35, 230), rng.randint(35, 230)], dtype=np.float32)
    for y in range(height):
        t = y / max(height - 1, 1)
        base[y, :, :] = (color_a * (1 - t) + color_b * t).astype(np.uint8)
    for _ in range(rng.randint(8, 18)):
        x0, y0 = rng.randint(0, width), rng.randint(0, height)
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        cv2.line(base, (x0, y0), (x1, y1), color, rng.randint(1, 5))
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, rng.uniform(4, 16), base.shape)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.2)))


def foreground_mask(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mask2 = cv2.inRange(hsv, np.array([0, 18, 45]), np.array([28, 190, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=1.0))


def make_composite(path: Path, label: int, index: int, rng: random.Random, size: int = 640) -> tuple[Image.Image, str]:
    hand = Image.open(path).convert("RGB")
    hand = ImageEnhance.Brightness(hand).enhance(rng.uniform(0.72, 1.28))
    hand = ImageEnhance.Contrast(hand).enhance(rng.uniform(0.82, 1.22))
    scale = rng.uniform(0.34, 0.72)
    target = int(size * scale)
    hand = hand.resize((target, target), Image.Resampling.BICUBIC)
    angle = rng.uniform(-18, 18)
    hand = hand.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    mask = foreground_mask(hand)
    background = make_background(size, size, rng)
    max_x = max(size - hand.width, 0)
    max_y = max(size - hand.height, 0)
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0
    background.paste(hand, (x, y), mask)
    name = f"synthetic_{index:05d}_label_{label}.jpg"
    return background, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = read_samples(ROOT / "images" / "test.txt")
    labels = []
    for index in range(args.count):
        path, label = samples[index % len(samples)]
        image, name = make_composite(path, label, index, rng)
        out_path = OUT_DIR / name
        image.save(out_path, quality=90)
        labels.append(f"{out_path.relative_to(ROOT).as_posix()} {label}")
    (OUT_DIR / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"Generated {len(labels)} images at {OUT_DIR}")


if __name__ == "__main__":
    main()
