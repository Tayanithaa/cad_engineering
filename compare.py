from __future__ import annotations

import cv2
import numpy as np

from config import OCR_CONFIDENCE_THRESHOLD, SSIM_CHANGE_THRESHOLD


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _crop(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]


def _ink_density(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = _gray(crop)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    return float(np.count_nonzero(binary) / max(binary.size, 1))


def region_ssim_score(diff_map: np.ndarray, bbox: list[int]) -> float:
    x, y, w, h = bbox
    crop = diff_map[y : y + h, x : x + w]
    if crop.size == 0:
        return 1.0
    return float(np.mean(crop))


def derive_geometric_change_type(image_a: np.ndarray, aligned_b: np.ndarray, diff_map: np.ndarray, bbox: list[int]) -> tuple[str, float]:
    ssim = region_ssim_score(diff_map, bbox)
    density_a = _ink_density(_crop(image_a, bbox))
    density_b = _ink_density(_crop(aligned_b, bbox))
    delta = density_b - density_a

    if abs(delta) > 0.018:
        return ("Added" if delta > 0 else "Removed"), ssim
    if ssim < SSIM_CHANGE_THRESHOLD:
        return "Modified", ssim
    return "Unchanged", ssim


def _normalized_string(value: str) -> str:
    return " ".join((value or "").lower().split())


def derive_text_change_type(ocr_result: dict) -> str:
    if not ocr_result.get("likely_text"):
        return "Unchanged"
    confidence = float(ocr_result.get("ocr_confidence") or 0.0)
    if confidence < OCR_CONFIDENCE_THRESHOLD:
        return "Possible change"

    value_a = ocr_result.get("parsed_value_a")
    value_b = ocr_result.get("parsed_value_b")
    unit = ocr_result.get("unit")
    if value_a is not None and value_b is not None:
        if unit and ocr_result.get("unit") != unit:
            return "Modified"
        return "Unchanged" if abs(float(value_a) - float(value_b)) < 1e-6 else "Modified"

    text_a = _normalized_string(ocr_result.get("normalized_text_a") or ocr_result.get("raw_text_a"))
    text_b = _normalized_string(ocr_result.get("normalized_text_b") or ocr_result.get("raw_text_b"))
    if text_a == text_b:
        return "Unchanged"
    if text_a and not text_b:
        return "Removed"
    if text_b and not text_a:
        return "Added"
    return "Modified"
