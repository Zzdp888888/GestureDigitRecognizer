# -*- coding: utf-8 -*-
"""Web server for gesture digit recognition.

Run with:
    python app_server.py
Then open:
    http://127.0.0.1:8000/web/
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet34

from hand_detector import detect_hand_candidates
from enhanced_model import build_enhanced_model


ROOT = Path(__file__).resolve().parent
# GUIDE_MODEL_PATH = ROOT / "models" / "model.pkl"
# CHINESE_MODEL_PATH = ROOT / "models" / "chinese_cnn_0_9.pkl"
GUIDE_MODEL_PATH = ROOT / "models" / "model_0-5test_own.pkl"
CHINESE_MODEL_PATH = ROOT / "models" / "model_chinesetest_own.pkl"
CHINESE_METRICS_PATH = ROOT / "outputs" / "chinese_cnn_gpu_cached_metrics.json"
CHINESE_AUG_CNN_PATH = ROOT / "models" / "chinese_cnn_0_9_aug.pkl"
CHINESE_AUG_METRICS_PATH = ROOT / "outputs" / "chinese_cnn_aug_metrics.json"
EFFICIENTNET_PATH = ROOT / "models" / "chinese_efficientnet_b0_grouped_0_9.pth"
EFFICIENTNET_METRICS_PATH = ROOT / "outputs" / "chinese_efficientnet_b0_grouped_metrics.json"
RESNET34_PATH = ROOT / "models" / "chinese_resnet34_grouped_0_9.pth"
RESNET34_METRICS_PATH = ROOT / "outputs" / "chinese_resnet34_grouped_metrics.json"
MOBILENET_PATH = ROOT / "models" / "chinese_mobilenetv3_grouped_0_9.pth"
MOBILENET_METRICS_PATH = ROOT / "outputs" / "chinese_mobilenetv3_grouped_metrics.json"
ENSEMBLE_CONFIG_PATH = ROOT / "models" / "GestureDigitRecognizer_model.json"
PACKAGED_MODEL_PATH = ROOT / "models" / "GestureDigitRecognizer_model.pt"
HOST = "127.0.0.1"
PORT = 8000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TO_TENSOR = transforms.ToTensor()


@dataclass
class ModelEntry:
    key: str
    label: str
    model: torch.nn.Module
    input_size: int
    accuracy: float | None
    scope: str
    family: str
    note: str = ""
    per_class_accuracy: dict[int, float] | None = None


model_entries: dict[str, ModelEntry] = {}


def relative_resource_key(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_packaged_model() -> dict:
    if not PACKAGED_MODEL_PATH.exists():
        return {}
    try:
        return torch.load(PACKAGED_MODEL_PATH, map_location="cpu", weights_only=False)
    except Exception:
        return {}


PACKAGED_MODEL = load_packaged_model()


def bundled_file_bytes(path: Path) -> bytes | None:
    files = PACKAGED_MODEL.get("files") if isinstance(PACKAGED_MODEL, dict) else None
    if not isinstance(files, dict):
        return None
    return files.get(relative_resource_key(path))


def resource_exists(path: Path) -> bool:
    return bundled_file_bytes(path) is not None or path.exists()


def read_text_resource(path: Path, encoding: str = "utf-8") -> str | None:
    data = bundled_file_bytes(path)
    if data is not None:
        return data.decode(encoding)
    if path.exists():
        return path.read_text(encoding=encoding)
    return None


def load_torch_resource(path: Path):
    data = bundled_file_bytes(path)
    if data is not None:
        return torch.load(io.BytesIO(data), map_location=device, weights_only=False)
    return torch.load(path, map_location=device, weights_only=False)


def load_ensemble_config() -> dict:
    if isinstance(PACKAGED_MODEL, dict) and isinstance(PACKAGED_MODEL.get("ensemble_config"), dict):
        return PACKAGED_MODEL["ensemble_config"]
    try:
        text = read_text_resource(ENSEMBLE_CONFIG_PATH)
        return json.loads(text) if text else {}
    except Exception:
        return {}


def parse_model_digit_map(data: dict | None, fallback: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    if not isinstance(data, dict):
        return fallback
    parsed: dict[tuple[str, int], float] = {}
    for model_key, digit_values in data.items():
        if not isinstance(digit_values, dict):
            continue
        for digit, value in digit_values.items():
            parsed[(str(model_key), int(digit))] = float(value)
    return parsed or fallback


def parse_digit_map(data: dict | None, fallback: dict[int, float]) -> dict[int, float]:
    if not isinstance(data, dict):
        return fallback
    parsed = {int(digit): float(value) for digit, value in data.items()}
    return parsed or fallback


ENSEMBLE_CONFIG = load_ensemble_config()
UPDATED_AT = ENSEMBLE_CONFIG.get("updated_at", "2026-06-30 14:32")
RELIABLE_CHINESE_ORDER = tuple(ENSEMBLE_CONFIG.get("reliable_chinese_order", ["resnet34", "efficientnet_b0", "chinese_aug"]))
REFERENCE_ONLY_MODELS = set(ENSEMBLE_CONFIG.get("reference_only_models", ["chinese"]))
FALLBACK_CHINESE_ORDER = (*RELIABLE_CHINESE_ORDER, "mobilenetv3", "chinese")
SPECIALIST_WEIGHT_BOOSTS = parse_model_digit_map(ENSEMBLE_CONFIG.get("specialist_weight_boosts"), {
    ("resnet34", 0): 1.06,
    ("resnet34", 1): 1.28,
    ("resnet34", 2): 1.24,
    ("resnet34", 3): 0.92,
    ("resnet34", 4): 1.22,
    ("resnet34", 8): 1.06,
    ("chinese_aug", 1): 2.85,
    ("chinese_aug", 3): 0.95,
    ("efficientnet_b0", 6): 1.48,
    ("efficientnet_b0", 7): 1.24,
    ("efficientnet_b0", 9): 1.05,
})
GUIDE_DIGIT_WEIGHTS = parse_digit_map(ENSEMBLE_CONFIG.get("guide_digit_weights"), {
    0: 0.56,
    3: 0.10,
})
MODEL_PROBABILITY_CALIBRATION = parse_model_digit_map(ENSEMBLE_CONFIG.get("model_probability_calibration"), {
    ("resnet34", 8): 0.72,
})
ADJUSTMENT_CONFIG = ENSEMBLE_CONFIG.get("adjustments", {})


def read_metric(path: Path, key: str = "accuracy") -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get(key)
        return float(value) if value is not None else None
    except Exception:
        return None


def read_metrics(path: Path) -> dict:
    try:
        text = read_text_resource(path)
        return json.loads(text) if text else {}
    except Exception:
        return {}


def read_per_class_accuracy(path: Path) -> dict[int, float] | None:
    data = read_metrics(path)
    per_class = data.get("per_class")
    if not isinstance(per_class, dict):
        return None
    result: dict[int, float] = {}
    for key, value in per_class.items():
        if isinstance(value, dict) and value.get("accuracy") is not None:
            result[int(key)] = float(value["accuracy"])
    return result or None


def register_model(entry: ModelEntry) -> None:
    entry.model.to(device)
    entry.model.eval()
    model_entries[entry.key] = entry


def load_full_model(path: Path) -> torch.nn.Module:
    return load_torch_resource(path)


def build_efficientnet_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = load_torch_resource(path)
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 10)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def build_resnet34_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = load_torch_resource(path)
    model = resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def build_mobilenet_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = load_torch_resource(path)
    model = build_enhanced_model(weights=None)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def load_models() -> None:
    register_model(
        ModelEntry(
            key="guide",
            label="指导书 CNN 0-5",
            model=load_full_model(GUIDE_MODEL_PATH),
            input_size=64,
            accuracy=0.9083,
            scope="0-5",
            family="custom_cnn",
            per_class_accuracy={0: 1.0, 1: 0.95, 2: 0.8, 3: 0.8375, 4: 0.85, 5: 1.0},
        )
    )
    if resource_exists(CHINESE_MODEL_PATH):
        register_model(
            ModelEntry(
            key="chinese",
            label="中国 CNN 0-9（旧参考）",
            model=load_full_model(CHINESE_MODEL_PATH),
            input_size=64,
            accuracy=None,
            scope="0-9",
            family="custom_cnn",
            note="旧随机切分模型，置信度偏高，只作为对比参考，不参与自动主结果。",
        )
        )
    if resource_exists(CHINESE_AUG_CNN_PATH):
        register_model(
            ModelEntry(
            key="chinese_aug",
            label="增强 CNN 0-9（分组测试）",
            model=load_full_model(CHINESE_AUG_CNN_PATH),
            input_size=64,
            accuracy=read_metric(CHINESE_AUG_METRICS_PATH),
            scope="0-9",
            family="custom_cnn_aug",
            note="使用新增中国手势数据集和数据增强，按来源分组测试。",
            per_class_accuracy=read_per_class_accuracy(CHINESE_AUG_METRICS_PATH),
        )
        )
    if resource_exists(EFFICIENTNET_PATH):
        model, checkpoint = build_efficientnet_checkpoint(EFFICIENTNET_PATH)
        register_model(
            ModelEntry(
            key="efficientnet_b0",
            label="EfficientNet-B0 0-9（分组测试）",
            model=model,
            input_size=int(checkpoint.get("input_size", 224)),
            accuracy=read_metric(EFFICIENTNET_METRICS_PATH) or checkpoint.get("accuracy"),
            scope="0-9",
            family="efficientnet_b0",
            note="按来源分组测试的高分辨率迁移模型。",
            per_class_accuracy=read_per_class_accuracy(EFFICIENTNET_METRICS_PATH),
        )
        )
    if resource_exists(RESNET34_PATH):
        model, checkpoint = build_resnet34_checkpoint(RESNET34_PATH)
        register_model(
            ModelEntry(
            key="resnet34",
            label="ResNet34 0-9（推荐）",
            model=model,
            input_size=int(checkpoint.get("input_size", 224)),
            accuracy=read_metric(RESNET34_METRICS_PATH) or checkpoint.get("accuracy"),
            scope="0-9",
            family="resnet34",
            note="当前分组测试最高，作为 auto/compare 的主推荐模型。",
            per_class_accuracy=read_per_class_accuracy(RESNET34_METRICS_PATH),
        )
        )
    if resource_exists(MOBILENET_PATH):
        model, checkpoint = build_mobilenet_checkpoint(MOBILENET_PATH)
        register_model(
            ModelEntry(
                key="mobilenetv3",
                label="MobileNetV3 0-9",
                model=model,
                input_size=int(checkpoint.get("input_size", 224)),
                accuracy=read_metric(MOBILENET_METRICS_PATH) or checkpoint.get("accuracy"),
                scope="0-9",
                family="mobilenet_v3_small",
            )
        )


load_models()
available_modes = list(model_entries.keys())
if any(key != "guide" for key in model_entries):
    available_modes += ["auto", "compare"]


def model_meta(entry: ModelEntry) -> dict:
    return {
        "model": entry.family,
        "label": entry.label,
        "input_size": entry.input_size,
        "accuracy": entry.accuracy,
        "scope": entry.scope,
        "note": entry.note,
        "reference_only": entry.key in REFERENCE_ONLY_MODELS,
        "per_class_accuracy": entry.per_class_accuracy,
    }


def default_chinese_key() -> str | None:
    for key in FALLBACK_CHINESE_ORDER:
        if key in model_entries:
            return key
    return None


def comparable_model_keys(predictions: dict) -> list[str]:
    reliable = [key for key in RELIABLE_CHINESE_ORDER if key in predictions]
    if reliable:
        keys = reliable[:]
        if "guide" in predictions:
            keys.append("guide")
        return keys
    fallback = [
        key
        for key in predictions
        if model_entries[key].scope == "0-9" and key not in REFERENCE_ONLY_MODELS
    ]
    return fallback or [key for key in predictions if model_entries[key].scope == "0-9"] or list(predictions)


def model_digit_weight(entry: ModelEntry, digit: int) -> float:
    if entry.key in REFERENCE_ONLY_MODELS:
        return 0.0
    if entry.key == "guide":
        return GUIDE_DIGIT_WEIGHTS.get(digit, 0.0)
    if entry.per_class_accuracy and digit in entry.per_class_accuracy:
        base_weight = max(float(entry.per_class_accuracy[digit]), 0.05)
    else:
        base_weight = max(float(entry.accuracy or 0.30), 0.05)
    return base_weight * SPECIALIST_WEIGHT_BOOSTS.get((entry.key, digit), 1.0)


def aligned_probability_for_entry(entry: ModelEntry, probability: torch.Tensor) -> torch.Tensor:
    if probability.numel() == 10:
        return probability
    aligned = torch.zeros(10, dtype=probability.dtype, device=probability.device)
    limit = min(probability.numel(), 10)
    aligned[:limit] = probability[:limit]
    return aligned


def apply_confusion_adjustments(score: torch.Tensor, predictions: dict) -> tuple[torch.Tensor, list[str]]:
    adjusted = score.clone()
    notes: list[str] = []
    resnet = predictions.get("resnet34")
    efficient = predictions.get("efficientnet_b0")
    chinese_aug = predictions.get("chinese_aug")
    three_config = ADJUSTMENT_CONFIG.get("three_hit_lock", {})
    eight_two_config = ADJUSTMENT_CONFIG.get("eight_two_confusion", {})
    seven_config = ADJUSTMENT_CONFIG.get("seven_specialist", {})

    strong_non3_agreement = (
        resnet
        and efficient
        and resnet["digit"] == efficient["digit"]
        and resnet["digit"] != 3
        and resnet["confidence"] >= float(three_config.get("strong_non3_agreement_min_confidence", 0.82))
        and efficient["confidence"] >= float(three_config.get("strong_non3_agreement_min_confidence", 0.82))
    )
    three_hit = False
    if resnet and resnet["digit"] == 3 and resnet["confidence"] >= float(three_config.get("resnet_min_confidence", 0.35)):
        three_hit = True
    if (
        chinese_aug
        and chinese_aug["digit"] == 3
        and chinese_aug["confidence"] >= float(three_config.get("chinese_aug_min_confidence", 0.35))
        and not strong_non3_agreement
    ):
        three_hit = True
    if three_config.get("enabled", True) and three_hit:
        adjusted[3] = torch.maximum(
            adjusted[3] * float(three_config.get("score_multiplier", 1.28)),
            adjusted.max() * float(three_config.get("min_over_current_max", 1.015)),
        )
        notes.append("3 命中锁定：ResNet34 命中 3，或增强 CNN 命中 3 且无强非 3 共识")

    if eight_two_config.get("enabled", True) and resnet:
        resnet_prob = aligned_probability_for_entry(model_entries["resnet34"], resnet["probability"])
        top_digit = int(torch.argmax(adjusted).item())
        near_8_confusion = top_digit in {2, 8} and adjusted[8] >= adjusted.max() * float(eight_two_config.get("near_top_ratio", 0.35))
        if (
            near_8_confusion
            and adjusted[2] > adjusted[8]
            and resnet_prob[8] >= resnet_prob[2] * float(eight_two_config.get("resnet_8_vs_2_ratio", 0.72))
            and adjusted[8] >= adjusted[2] * float(eight_two_config.get("ensemble_8_vs_2_ratio", 0.62))
        ):
            adjusted[8] *= float(eight_two_config.get("boost_8", 1.32))
            adjusted[2] *= float(eight_two_config.get("dampen_2", 0.88))
            notes.append("8/2 混淆校正：ResNet 对 8 有足够证据，提升 8、压低 2")

    if seven_config.get("enabled", True) and efficient:
        efficient_prob = aligned_probability_for_entry(model_entries["efficientnet_b0"], efficient["probability"])
        if (
            efficient_prob[7] >= float(seven_config.get("efficientnet_7_min_probability", 0.45))
            and adjusted[7] >= adjusted.max() * float(seven_config.get("ensemble_near_top_ratio", 0.58))
        ):
            adjusted[7] *= float(seven_config.get("boost_7", 1.22))
            notes.append("7 专家校正：EfficientNet 对 7 的分组测试表现最好")

    adjusted = adjusted / max(float(adjusted.sum().item()), 1e-8)
    return adjusted, notes


def ensemble_prediction(predictions: dict) -> dict | None:
    candidates = comparable_model_keys(predictions)
    if not candidates:
        return None

    score = torch.zeros(10, device=device)
    model_weights: dict[str, dict[int, float]] = {}
    used_keys: list[str] = []
    for key in candidates:
        entry = model_entries[key]
        aligned_probability = aligned_probability_for_entry(entry, predictions[key]["probability"])
        weights = torch.tensor(
            [model_digit_weight(entry, digit) for digit in range(10)],
            dtype=aligned_probability.dtype,
            device=aligned_probability.device,
        )
        if float(weights.max().item()) <= 0:
            continue
        model_weights[key] = {digit: round(float(weights[digit].item()), 4) for digit in range(10)}
        score += aligned_probability * weights
        used_keys.append(key)

    if not used_keys:
        return None

    score = score / max(float(score.sum().item()), 1e-8)
    score, adjustment_notes = apply_confusion_adjustments(score, predictions)
    digit, confidence, top3 = topk_result(score)
    return {
        "key": "gesture_digit_recognizer",
        "label": "手势数字识别模型（推荐）",
        "digit": digit,
        "confidence": round(confidence, 4),
        "top3": top3,
        "probability": score,
        "used_models": used_keys,
        "model_weights": model_weights,
        "adjustments": adjustment_notes,
        "model_meta": {
            "model": "gesture_digit_recognizer",
            "label": "手势数字识别模型（推荐）",
            "input_size": "mixed",
            "accuracy": None,
            "scope": "0-9",
            "note": "按各模型在测试集每个数字上的准确率逐类加权，并对 7、8/2 常见混淆做保守校正。",
            "reference_only": False,
            "per_class_accuracy": None,
        },
    }


def model_probability(entry: ModelEntry, image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((entry.input_size, entry.input_size), Image.Resampling.BILINEAR)
    tensor = TO_TENSOR(image)
    if entry.input_size >= 128:
        tensor = IMAGENET_NORMALIZE(tensor)
    tensor = tensor.unsqueeze(0).to(device)
    output = entry.model(tensor)
    probability = torch.softmax(output, dim=1)[0]
    calibration = [
        (digit, factor)
        for (model_key, digit), factor in MODEL_PROBABILITY_CALIBRATION.items()
        if model_key == entry.key and digit < probability.numel()
    ]
    if calibration:
        probability = probability.clone()
        for digit, factor in calibration:
            probability[digit] *= factor
        probability = probability / max(float(probability.sum().item()), 1e-8)
    return probability


def topk_result(probability: torch.Tensor, k: int = 3) -> tuple[int, float, list[dict]]:
    confidence, predicted = torch.max(probability, 0)
    top_values, top_indices = torch.topk(probability, k=min(k, probability.numel()))
    top = [
        {"digit": int(idx.item()), "confidence": round(float(val.item()), 4)}
        for val, idx in zip(top_values, top_indices)
    ]
    return int(predicted.item()), float(confidence.item()), top


def run_mode_probabilities(image: Image.Image, mode: str) -> dict:
    if mode == "compare":
        keys = list(model_entries.keys())
    elif mode == "auto":
        keys = [key for key in RELIABLE_CHINESE_ORDER if key in model_entries]
        if "guide" in model_entries:
            keys = ["guide", *keys]
        if not keys:
            chinese_key = default_chinese_key()
            keys = [chinese_key] if chinese_key else ["guide"]
    else:
        keys = [mode] if mode in model_entries else ["guide"]

    predictions = {}
    for key in keys:
        entry = model_entries[key]
        probability = model_probability(entry, image)
        digit, confidence, top3 = topk_result(probability)
        predictions[key] = {
            "key": key,
            "label": entry.label,
            "digit": digit,
            "confidence": round(confidence, 4),
            "top3": top3,
            "model_meta": model_meta(entry),
            "probability": probability,
        }

    ensemble = ensemble_prediction(predictions) if mode in {"auto", "compare"} else None
    if ensemble:
        selected_prediction = ensemble
        selected_meta = ensemble["model_meta"]
        selected_key = ensemble["key"]
        actual_mode = f"{mode}_gesture_digit_recognizer"
    elif mode == "auto":
        selected_key = default_chinese_key() if default_chinese_key() in predictions else None
        selected_key = selected_key or ("guide" if "guide" in predictions else next(iter(predictions)))
        selected_prediction = predictions[selected_key]
        selected_meta = model_meta(model_entries[selected_key])
        actual_mode = f"auto_{selected_key}"
    elif mode == "compare":
        selected_key = comparable_model_keys(predictions)[0]
        selected_prediction = predictions[selected_key]
        selected_meta = model_meta(model_entries[selected_key])
        actual_mode = "compare"
    else:
        selected_key = keys[0]
        selected_prediction = predictions[selected_key]
        selected_meta = model_meta(model_entries[selected_key])
        actual_mode = selected_key

    model_results = {
        key: {k: v for k, v in pred.items() if k != "probability"}
        for key, pred in predictions.items()
    }
    if ensemble:
        model_results[ensemble["key"]] = {
            key: value
            for key, value in ensemble.items()
            if key not in {"probability", "model_weights"}
        }
    return {
        "probability": selected_prediction["probability"],
        "actual_mode": actual_mode,
        "chosen_key": selected_key,
        "chosen_meta": selected_meta,
        "digit": selected_prediction["digit"],
        "confidence": selected_prediction["confidence"],
        "top3": selected_prediction["top3"],
        "model_results": model_results,
        "ensemble": {k: v for k, v in ensemble.items() if k != "probability"} if ensemble else None,
    }


def roi_selection_score(model_confidence: float, detector_score: float, hand_detected: bool, candidate_index: int) -> float:
    detector_bonus = 0.10 * detector_score if hand_detected else -0.10
    rank_penalty = min(candidate_index * 0.025, 0.10)
    return model_confidence + detector_bonus - rank_penalty


def predict_from_image(image: Image.Image, requested_mode: str = "auto") -> dict:
    source_size = image.size

    mode = requested_mode if requested_mode in available_modes else "auto"
    if mode == "auto" and default_chinese_key() is None:
        mode = "guide"

    candidates = detect_hand_candidates(image, max_candidates=7)
    scored = []
    with torch.no_grad():
        for index, candidate in enumerate(candidates):
            result = run_mode_probabilities(candidate.image, mode)
            score = roi_selection_score(result["confidence"], candidate.score, candidate.hand_detected, index)
            scored.append((score, candidate, result, index))

    scored.sort(reverse=True, key=lambda item: item[0])
    selection_score, hand_region, selected, selected_index = scored[0]
    inference_image = hand_region.image
    confidence_value = selected["confidence"]
    low_quality_roi = (
        not hand_region.hand_detected
        and confidence_value < 0.82
        and source_size[0] * source_size[1] > 64 * 64
    )
    detector_message = hand_region.message
    if len(candidates) > 1:
        detector_message = f"{detector_message}；已在 {len(candidates)} 个 ROI 候选中选择第 {selected_index + 1} 个"
    if low_quality_roi:
        detector_message = f"{detector_message}；ROI 质量偏低，建议让手靠近画面中央并避开人脸/强光"

    return {
        "digit": selected["digit"],
        "confidence": round(confidence_value, 4),
        "device": str(device),
        "mode": selected["actual_mode"],
        "requested_mode": requested_mode,
        "model": selected["chosen_meta"]["model"],
        "model_key": selected["chosen_key"],
        "model_meta": selected["chosen_meta"],
        "source_size": {"width": source_size[0], "height": source_size[1]},
        "roi_size": {"width": inference_image.size[0], "height": inference_image.size[1]},
        "hand_detected": hand_region.hand_detected and not low_quality_roi,
        "hand_bbox": hand_region.bbox,
        "detector": hand_region.detector,
        "detector_score": hand_region.score,
        "detector_message": detector_message,
        "roi_selection_score": round(selection_score, 4),
        "roi_candidates": len(candidates),
        "selected_roi_index": selected_index,
        "segmentation_applied": hand_region.segmentation_applied,
        "mask_coverage": hand_region.mask_coverage,
        "crop_preview": hand_region.crop_preview,
        "guide": selected["model_results"].get("guide"),
        "chinese": selected["model_results"].get(selected["chosen_key"]),
        "model_results": selected["model_results"],
        "ensemble": selected.get("ensemble"),
        "top3": selected["top3"],
    }


def decode_image_from_json(body: bytes) -> tuple[Image.Image, str]:
    payload = json.loads(body.decode("utf-8"))
    image_data = payload.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    binary = base64.b64decode(image_data)
    return Image.open(io.BytesIO(binary)), payload.get("mode", "auto")


class GestureHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in ("/", "/web"):
            path = "/web/"
        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "device": str(device),
                    "updated_at": UPDATED_AT,
                    "default_mode": "auto" if "auto" in available_modes else "guide",
                    "available_modes": available_modes,
                    "models": {key: model_meta(entry) for key, entry in model_entries.items()},
                    "recommended_model": default_chinese_key() or "guide",
                    "ensemble_package": {
                        "name": ENSEMBLE_CONFIG.get("name", "inline_defaults"),
                        "version": ENSEMBLE_CONFIG.get("version", "inline"),
                        "path": str(ENSEMBLE_CONFIG_PATH.relative_to(ROOT)),
                    },
                    "packaged_model": {
                        "enabled": bool(PACKAGED_MODEL),
                        "name": PACKAGED_MODEL.get("name") if isinstance(PACKAGED_MODEL, dict) else None,
                        "display_name": PACKAGED_MODEL.get("display_name") if isinstance(PACKAGED_MODEL, dict) else None,
                        "version": PACKAGED_MODEL.get("version") if isinstance(PACKAGED_MODEL, dict) else None,
                        "path": str(PACKAGED_MODEL_PATH.relative_to(ROOT)),
                    },
                    "reliable_compare_order": [key for key in RELIABLE_CHINESE_ORDER if key in model_entries],
                    "reference_only_models": sorted(key for key in REFERENCE_ONLY_MODELS if key in model_entries),
                    "recognition_scope": {
                        "default": "auto runs reliable Chinese 0-9 models and uses per-digit collaborative scoring",
                        "guide": "0-5 guide dataset",
                        "compare": "show every loaded model, main result follows grouped-test reliability",
                    },
                    "hand_detector": {
                        "pipeline": "multi_roi_candidates -> model_rerank -> soft_background_blur",
                        "enabled": True,
                    },
                },
            )
            return
        if path.startswith("/sample/"):
            sample_path = ROOT / path.removeprefix("/sample/").lstrip("/")
        elif path.startswith("/web/"):
            sample_path = ROOT / path.lstrip("/")
            if sample_path.is_dir():
                sample_path = sample_path / "index.html"
        else:
            self._send_json(404, {"error": "Not found"})
            return

        if not sample_path.exists() or not sample_path.resolve().is_relative_to(ROOT):
            self._send_json(404, {"error": "File not found"})
            return

        content_type = mimetypes.guess_type(str(sample_path))[0] or "application/octet-stream"
        self._send(200, sample_path.read_bytes(), content_type)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            image, requested_mode = decode_image_from_json(self.rfile.read(length))
            self._send_json(200, predict_from_image(image, requested_mode))
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    print(f"Gesture recognition server running on http://{HOST}:{PORT}")
    print(f"Web: http://{HOST}:{PORT}/web/")
    print(f"Model device: {device}")
    print(f"Available modes: {available_modes}")
    print(f"Loaded models: {list(model_entries)}")
    ThreadingHTTPServer((HOST, PORT), GestureHandler).serve_forever()


if __name__ == "__main__":
    main()
