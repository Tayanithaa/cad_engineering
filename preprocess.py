from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropResult:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    found_border: bool
    metadata: dict


def _safe_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def find_outer_border_bbox(image: np.ndarray) -> tuple[int, int, int, int, bool]:
    try:
        gray = _safe_gray(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as exc:
        raise ValueError("OpenCV failed while detecting the drawing border.") from exc

    h, w = image.shape[:2]
    min_area = 0.10 * w * h
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area:
            continue
        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        border_score = area * (0.5 + fill_ratio)
        candidates.append((border_score, (x, y, bw, bh)))

    if not candidates:
        return 0, 0, w, h, False

    _, bbox = max(candidates, key=lambda item: item[0])
    x, y, bw, bh = bbox
    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    x2 = min(w, x + bw + 2 * pad)
    y2 = min(h, y + bh + 2 * pad)
    return x, y, x2 - x, y2 - y, True


def crop_to_border(image: np.ndarray, label: str = "drawing") -> CropResult:
    x, y, w, h, found = find_outer_border_bbox(image)
    cropped = image[y : y + h, x : x + w].copy()
    metadata = {
        "label": label,
        "border_bbox": [int(x), int(y), int(w), int(h)],
        "found_border": bool(found),
        "cropped_width": int(cropped.shape[1]),
        "cropped_height": int(cropped.shape[0]),
    }
    print(f"[Stage 2] {label}: border {'found' if found else 'not found, using full page'} {metadata['border_bbox']}")
    return CropResult(image=cropped, bbox=(x, y, w, h), found_border=found, metadata=metadata)


def crop_pair_to_borders(image_a: np.ndarray, image_b: np.ndarray) -> tuple[CropResult, CropResult]:
    return crop_to_border(image_a, "Drawing A"), crop_to_border(image_b, "Drawing B")
