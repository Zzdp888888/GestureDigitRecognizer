# -*- coding: utf-8 -*-
"""Build a manifest for the Chinese number gesture dataset.

Important: the real label is the first number in the filename, e.g.
`6_1_0_167.jpg` has label 6. Folder names such as `resized_img7` are only
split buckets and must not be used as labels.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT
    / "extra_data"
    / "Chinese-number-gestures-recognition"
    / "digital_gesture_recognition"
    / "resized_img_split"
)
MANIFEST_PATH = ROOT / "extra_data" / "chinese_gesture_manifest_0_9.csv"
SUMMARY_PATH = ROOT / "outputs" / "chinese_gesture_manifest_summary.json"


def parse_label(path: Path) -> int | None:
    try:
        return int(path.name.split("_", 1)[0])
    except ValueError:
        return None


def main() -> None:
    rows = []
    folder_counts = defaultdict(Counter)
    for path in sorted(DATA_ROOT.rglob("*.jpg")):
        label = parse_label(path)
        if label is None or not 0 <= label <= 9:
            continue
        relative_path = path.relative_to(ROOT).as_posix()
        rows.append({"path": relative_path, "label": label, "folder": path.parent.name})
        folder_counts[path.parent.name][label] += 1

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "folder"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(MANIFEST_PATH),
        "total": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "folder_counts": {folder: dict(sorted(counter.items())) for folder, counter in sorted(folder_counts.items())},
        "label_rule": "label = int(filename.split('_', 1)[0])",
    }
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
