# -*- coding: utf-8 -*-
"""Create guide-style train/test txt files for Chinese number gestures.

Each line follows the original practical guide format:
    ./path/to/image.jpg label

The label is parsed from the filename prefix. For example `0_0_102.jpg`
has label 0. Folder names are not labels.

Source:
    extra_data/Chinese-number-gestures-recognition/digital_gesture_recognition/resized_img
"""

from __future__ import annotations

import random
import shutil
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = (
    ROOT
    / "extra_data"
    / "Chinese-number-gestures-recognition"
    / "digital_gesture_recognition"
    / "resized_img"
)
OUT_DIR = ROOT / "images" / "chinese"

# Relative prefix from ROOT to SRC_DIR
REL_PREFIX = SRC_DIR.relative_to(ROOT)


def source_group_key(rel_path: str, label: int) -> str:
    """Return a stable source-group key for leakage-free splitting.

    The Chinese dataset stores about 100 augmented variants per source gesture.
    Filenames look like ``6_1_0_167.jpg`` or ``0_0_4578.jpg``; the last numeric
    segment is the generated variant id. Splitting by image leaks near-duplicate
    variants into both train and test, so we split by filename without the last
    segment.
    """
    stem = Path(rel_path).stem
    parts = stem.split("_")
    if len(parts) <= 1:
        return f"{label}:{stem}"
    return f"{label}:{'_'.join(parts[:-1])}"


def scan_images() -> list[tuple[str, int]]:
    """Scan resized_img directory and return (relative_path, label) pairs.

    Filename format: ``label_xxxx_xxxx.jpg`` where label is the first segment
    before the first underscore.
    """
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    samples: list[tuple[str, int]] = []
    for img_path in sorted(SRC_DIR.glob("*.jpg")):
        try:
            label = int(img_path.name.split("_", 1)[0])
        except ValueError:
            print(f"  [SKIP] Cannot parse label from: {img_path.name}")
            continue
        # Relative path from ROOT, with ./ prefix (use POSIX slashes)
        rel_path = f"./{REL_PREFIX.as_posix()}/{img_path.name}"
        samples.append((rel_path, label))

    if not samples:
        raise RuntimeError(f"No valid .jpg files found in: {SRC_DIR}")

    return samples


def stratified_split(
    samples: list[tuple[str, int]],
    test_ratio: float = 0.15,
    seed: int = 20260629,
    group_split: bool = True,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Stratified train/test split.

    By default, split by source group instead of individual image to prevent
    augmented variants of the same original gesture from leaking into test.
    """
    rng = random.Random(seed)
    if group_split:
        by_label_group: dict[int, dict[str, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
        for path, label in samples:
            by_label_group[label][source_group_key(path, label)].append((path, label))

        train: list[tuple[str, int]] = []
        test: list[tuple[str, int]] = []
        for label, groups in sorted(by_label_group.items()):
            group_items = list(groups.items())
            rng.shuffle(group_items)
            test_group_count = max(1, round(len(group_items) * test_ratio))
            test_keys = {key for key, _ in group_items[:test_group_count]}
            for key, rows in group_items:
                if key in test_keys:
                    test.extend(rows)
                else:
                    train.extend(rows)

        rng.shuffle(train)
        rng.shuffle(test)
        return train, test

    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for path, label in samples:
        by_label[label].append((path, label))

    train: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []
    for label, items in sorted(by_label.items()):
        items = items[:]
        rng.shuffle(items)
        test_count = max(100, int(len(items) * test_ratio))
        test.extend(items[:test_count])
        train.extend(items[test_count:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_txt(path: Path, rows: list[tuple[str, int]]) -> None:
    """Write dataset lines to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{rel_path} {label}" for rel_path, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backup_existing_txt() -> None:
    if not OUT_DIR.exists():
        return
    existing = [OUT_DIR / "train.txt", OUT_DIR / "test.txt"]
    if not any(path.exists() for path in existing):
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = OUT_DIR / "bac" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    print(f"  Existing txt files backed up to: {backup_dir}")


def print_distribution(name: str, rows: list[tuple[str, int]]) -> None:
    """Print per-class distribution of a split."""
    counts = Counter(label for _, label in rows)
    print(f"  [{name}] total={len(rows)}  per-class: {dict(sorted(counts.items()))}")


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found:\n  {SRC_DIR}")

    print(f"Scanning: {SRC_DIR}")
    all_samples = scan_images()
    print(f"  Found {len(all_samples)} images")
    print_distribution("all", all_samples)

    # Handle labels 0-9 only (class 10 is excluded to match 10-class model)
    samples_0_9 = [(p, lbl) for p, lbl in all_samples if 0 <= lbl <= 9]
    skipped = len(all_samples) - len(samples_0_9)
    if skipped:
        print(f"  Filtered to labels 0-9, skipped {skipped} samples with label 10")

    train, test = stratified_split(samples_0_9, test_ratio=0.20, group_split=True)

    print(f"\nSplit result:")
    print_distribution("train", train)
    print_distribution("test", test)

    backup_existing_txt()

    train_txt = OUT_DIR / "train.txt"
    test_txt = OUT_DIR / "test.txt"
    write_txt(train_txt, train)
    write_txt(test_txt, test)

    print(f"\nDataset txt files written:")
    print(f"  train → {train_txt}  ({len(train)} lines)")
    print(f"  test  → {test_txt}  ({len(test)} lines)")


if __name__ == "__main__":
    main()
