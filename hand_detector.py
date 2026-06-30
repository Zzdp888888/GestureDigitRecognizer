# -*- coding: utf-8 -*-
"""Hand region detector used before digit classification.

The classifier in the training guide expects a cropped hand image. Webcam
frames are full scenes, so this module first proposes square hand ROI
candidates. MediaPipe is used when the environment supports it; otherwise
OpenCV skin masking plus face masking provides a dependency-light fallback.
"""

from __future__ import annotations

import base64
import io
from importlib import metadata
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    image: Image.Image
    hand_detected: bool
    bbox: dict | None
    detector: str
    score: float
    crop_preview: str | None
    message: str
    segmentation_applied: bool
    mask_coverage: float


_MP_HANDS = None
_MP_IMPORT_ERROR: str | None = None
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _try_mediapipe_hands():
    global _MP_HANDS, _MP_IMPORT_ERROR
    if _MP_HANDS is not None or _MP_IMPORT_ERROR is not None:
        return _MP_HANDS
    try:
        numpy_version = metadata.version("numpy")
        tensorflow_version = metadata.version("tensorflow")
        numpy_major = int(numpy_version.split(".", 1)[0])
        tensorflow_major_minor = tuple(int(part) for part in tensorflow_version.split(".")[:2])
        if numpy_major >= 2 and tensorflow_major_minor < (2, 16):
            _MP_IMPORT_ERROR = f"TensorFlow {tensorflow_version} 与 NumPy {numpy_version} 不兼容"
            return None
    except metadata.PackageNotFoundError:
        pass
    except Exception:
        pass
    try:
        import mediapipe as mp  # type: ignore

        _MP_HANDS = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.45,
        )
    except Exception as exc:
        _MP_IMPORT_ERROR = str(exc).splitlines()[-1] if str(exc) else exc.__class__.__name__
        _MP_HANDS = None
    return _MP_HANDS


def _crop_with_margin(image: Image.Image, x: int, y: int, w: int, h: int, margin_ratio: float = 0.32) -> tuple[Image.Image, dict]:
    width, height = image.size
    side = int(max(w, h) * (1.0 + margin_ratio * 2.0))
    side = max(side, 1)
    side = min(side, max(width, height))
    center_x = x + w * 0.5
    center_y = y + h * 0.5
    left = int(round(center_x - side * 0.5))
    top = int(round(center_y - side * 0.5))
    left = max(min(left, width - side), 0)
    top = max(min(top, height - side), 0)
    right = min(left + side, width)
    bottom = min(top + side, height)
    crop = image.crop((left, top, right, bottom))
    bbox = {
        "x": int(left),
        "y": int(top),
        "width": int(right - left),
        "height": int(bottom - top),
    }
    return crop, bbox


def _center_fallback(image: Image.Image) -> tuple[Image.Image, dict]:
    width, height = image.size
    side = int(min(width, height) * 0.82)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    crop = image.crop((left, top, left + side, top + side))
    bbox = {"x": int(left), "y": int(top), "width": int(side), "height": int(side)}
    return crop, bbox


def _bbox_iou(a: dict, b: dict) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = ax0 + a["width"], ay0 + a["height"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = bx0 + b["width"], by0 + b["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _preview_data_url(image: Image.Image, max_side: int = 220) -> str:
    preview = image.convert("RGB").copy()
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mask_hsv = cv2.inRange(hsv, np.array([0, 18, 45]), np.array([28, 190, 255]))
    return cv2.bitwise_or(mask_ycrcb, mask_hsv)


def _segment_hand_foreground(crop: Image.Image, source_size: tuple[int, int] | None = None) -> tuple[Image.Image, bool, float]:
    """Softly de-emphasize background inside a hand ROI.

    This is deliberately not hard instance segmentation. The mask only controls
    a soft blend: probable skin stays sharp, the rest is blurred/desaturated.
    If the mask quality is poor, the original crop is returned.
    """
    rgb = np.asarray(crop.convert("RGB"))
    height, width = rgb.shape[:2]
    if min(width, height) < 80:
        return crop.convert("RGB"), False, 0.0

    skin = _skin_mask(rgb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foreground = np.zeros_like(skin)
    min_area = max(width * height * 0.018, 80)
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(contour) < min_area:
            continue
        cv2.drawContours(foreground, [contour], -1, 255, thickness=-1)
        kept += 1
    if kept == 0:
        return crop.convert("RGB"), False, 0.0

    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    coverage = float((foreground > 0).mean())
    if coverage < 0.045 or coverage > 0.78:
        return crop.convert("RGB"), False, round(coverage, 4)

    foreground = cv2.GaussianBlur(foreground, (9, 9), 0)
    alpha = np.clip(foreground.astype(np.float32) / 255.0, 0.0, 1.0)
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=4.5, sigmaY=4.5)
    gray = cv2.cvtColor(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    background = (blurred.astype(np.float32) * 0.55 + gray.astype(np.float32) * 0.45).astype(np.uint8)
    composited = (rgb.astype(np.float32) * alpha[..., None] + background.astype(np.float32) * (1.0 - alpha[..., None])).astype(np.uint8)
    return Image.fromarray(composited), True, round(coverage, 4)


def _mediapipe_detect(image: Image.Image) -> DetectionResult | None:
    hands = _try_mediapipe_hands()
    if hands is None:
        return None

    rgb = np.asarray(image.convert("RGB"))
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    height, width = rgb.shape[:2]
    landmarks = result.multi_hand_landmarks[0].landmark
    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]
    x0, x1 = max(min(xs), 0), min(max(xs), width - 1)
    y0, y1 = max(min(ys), 0), min(max(ys), height - 1)
    crop, bbox = _crop_with_margin(image, x0, y0, x1 - x0, y1 - y0, margin_ratio=0.42)
    segmented, segmentation_applied, mask_coverage = _segment_hand_foreground(crop, image.size)
    return DetectionResult(
        image=segmented,
        hand_detected=True,
        bbox=bbox,
        detector="mediapipe_hands",
        score=1.0,
        crop_preview=_preview_data_url(segmented),
        message="已通过 MediaPipe Hands 定位手部区域，并使用方形 ROI 裁剪",
        segmentation_applied=segmentation_applied,
        mask_coverage=mask_coverage,
    )


def _mediapipe_candidates(image: Image.Image) -> list[DetectionResult]:
    detected = _mediapipe_detect(image)
    return [detected] if detected is not None else []


def _mask_faces(rgb: np.ndarray, mask: np.ndarray) -> None:
    if _FACE_CASCADE.empty():
        return
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(48, 48))
    for x, y, w, h in faces:
        pad = int(max(w, h) * 0.18)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, mask.shape[1])
        y1 = min(y + h + pad, mask.shape[0])
        mask[y0:y1, x0:x1] = 0


def _opencv_candidates(image: Image.Image, max_candidates: int = 5) -> list[DetectionResult]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if width < 32 or height < 32:
        return []

    mask = _skin_mask(rgb)
    _mask_faces(rgb, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(width * height * 0.012, 520)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 24 or h < 24:
            continue
        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area
        aspect = w / max(h, 1)
        if aspect < 0.28 or aspect > 3.8:
            continue
        center_y = (y + h * 0.5) / height
        center_x = (x + w * 0.5) / width
        centrality = 1.0 - min(abs(center_x - 0.5) * 1.15, 0.75)
        lower_half_bonus = 1.08 if center_y > 0.42 else 0.88
        shape_score = 1.0 - min(abs(fill_ratio - 0.48), 0.42)
        score = (area / (width * height)) * centrality * lower_half_bonus * max(shape_score, 0.25)
        candidates.append((score, x, y, w, h, area))

    if not candidates:
        return []

    candidates.sort(reverse=True, key=lambda item: item[0])
    results: list[DetectionResult] = []
    seen: list[dict] = []
    for rank, (score, x, y, w, h, _) in enumerate(candidates):
        crop, bbox = _crop_with_margin(image, x, y, w, h, margin_ratio=0.42 if rank == 0 else 0.34)
        if any(_bbox_iou(bbox, old) > 0.72 for old in seen):
            continue
        seen.append(bbox)
        segmented, segmentation_applied, mask_coverage = _segment_hand_foreground(crop, image.size)
        normalized_score = min(float(score) * 18.0, 0.98)
        results.append(
            DetectionResult(
                image=segmented,
                hand_detected=True,
                bbox=bbox,
                detector=f"opencv_skin_candidate_{len(results) + 1}",
                score=round(normalized_score, 4),
                crop_preview=_preview_data_url(segmented),
                message="已通过 OpenCV 肤色轮廓生成手部候选 ROI，并进行软背景虚化",
                segmentation_applied=segmentation_applied,
                mask_coverage=mask_coverage,
            )
        )
        if len(results) >= max_candidates:
            break
    return results


def _opencv_detect(image: Image.Image) -> DetectionResult | None:
    candidates = _opencv_candidates(image, max_candidates=1)
    return candidates[0] if candidates else None


def _fallback_candidates(image: Image.Image) -> list[DetectionResult]:
    width, height = image.size
    base: list[tuple[str, tuple[int, int, int, int]]] = []
    center_crop, center_bbox = _center_fallback(image)
    base.append(("center_fallback", (center_bbox["x"], center_bbox["y"], center_bbox["width"], center_bbox["height"])))

    side = int(min(width, height) * 0.58)
    for name, cx, cy in [
        ("lower_center_fallback", width * 0.5, height * 0.62),
        ("left_center_fallback", width * 0.34, height * 0.55),
        ("right_center_fallback", width * 0.66, height * 0.55),
    ]:
        left = int(max(min(cx - side / 2, width - side), 0))
        top = int(max(min(cy - side / 2, height - side), 0))
        base.append((name, (left, top, side, side)))

    results = []
    seen: list[dict] = []
    for name, (x, y, w, h) in base:
        crop = image.crop((x, y, x + w, y + h))
        bbox = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        if any(_bbox_iou(bbox, old) > 0.82 for old in seen):
            continue
        seen.append(bbox)
        results.append(
            DetectionResult(
                image=crop,
                hand_detected=False,
                bbox=bbox,
                detector=name,
                score=0.0,
                crop_preview=_preview_data_url(crop),
                message="未检测到稳定手部轮廓；已加入兜底候选 ROI",
                segmentation_applied=False,
                mask_coverage=0.0,
            )
        )
    return results


def detect_hand_candidates(image: Image.Image, max_candidates: int = 7) -> list[DetectionResult]:
    image = image.convert("RGB")
    candidates: list[DetectionResult] = []
    candidates.extend(_mediapipe_candidates(image))
    candidates.extend(_opencv_candidates(image, max_candidates=max_candidates))
    candidates.extend(_fallback_candidates(image))

    unique: list[DetectionResult] = []
    seen: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: (item.hand_detected, item.score), reverse=True):
        if candidate.bbox is not None and any(_bbox_iou(candidate.bbox, old) > 0.86 for old in seen):
            continue
        if candidate.bbox is not None:
            seen.append(candidate.bbox)
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique or _fallback_candidates(image)[:1]


def detect_hand_region(image: Image.Image) -> DetectionResult:
    image = image.convert("RGB")
    return detect_hand_candidates(image, max_candidates=1)[0]
