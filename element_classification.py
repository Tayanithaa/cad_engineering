from __future__ import annotations

import cv2
import numpy as np

from config import DOOR_BOTTOM_BAND_RATIO, ROOF_TOP_BAND_RATIO, WINDOW_MAX_AREA_RATIO


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _building_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    gray = _gray(image)
    _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return x, y, w, h


def _has_column_lines(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    gray = _gray(crop)
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(12, int(crop.shape[0] * 0.45))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return False
    vertical_x = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > 2 * max(dx, 1):
            vertical_x.append((x1 + x2) / 2)
    if len(vertical_x) < 2:
        return False
    vertical_x.sort()
    gaps = np.diff(vertical_x)
    if len(gaps) == 0:
        return False
    return bool(np.median(gaps) > 3 and np.std(gaps) <= max(8, np.median(gaps) * 0.75))


def _is_roof_like(contour: np.ndarray | None, crop: np.ndarray) -> bool:
    if contour is None:
        gray = _gray(crop)
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    return 3 <= len(approx) <= 4


def classify_region(image: np.ndarray, region: dict, building_box: tuple[int, int, int, int] | None = None) -> dict:
    x, y, w, h = region["bbox"]
    contour = region.get("contour")
    img_h, img_w = image.shape[:2]
    building_box = building_box or _building_bbox(image)
    bx, by, bw, bh = building_box
    area = float(region.get("area", w * h))
    aspect_ratio = w / max(h, 1)
    area_ratio = area / max(img_w * img_h, 1)
    center_y = y + h / 2
    relative_y = (center_y - by) / max(bh, 1)
    crop = image[max(0, y - 4) : min(img_h, y + h + 4), max(0, x - 4) : min(img_w, x + w + 4)]

    category = "Geometric change (unclassified)"
    confidence = 0.35
    reason = "No confident shape rule matched."

    if _has_column_lines(crop):
        category = "Pillar/Column"
        confidence = 0.78
        reason = "Long near-vertical parallel lines detected."
    elif relative_y <= ROOF_TOP_BAND_RATIO and _is_roof_like(contour, crop):
        category = "Roof"
        confidence = 0.72
        reason = "Top-band triangular/trapezoidal contour."
    elif aspect_ratio < 0.75 and relative_y >= 1.0 - DOOR_BOTTOM_BAND_RATIO and h > w * 1.4:
        category = "Door"
        confidence = 0.70
        reason = "Tall rectangle in the lower building band."
    elif 0.45 <= aspect_ratio <= 2.25 and area_ratio <= WINDOW_MAX_AREA_RATIO:
        category = "Window"
        confidence = 0.62
        reason = "Small-to-medium square or moderately rectangular region."

    return {
        "region_id": region["region_id"],
        "bbox": [int(x), int(y), int(w), int(h)],
        "element_type": category,
        "classification_confidence": confidence,
        "classification_reason": reason,
        "aspect_ratio": float(aspect_ratio),
        "area": area,
    }


def classify_regions(image: np.ndarray, regions: list[dict]) -> list[dict]:
    building_box = _building_bbox(image)
    results = [classify_region(image, region, building_box) for region in regions]
    print(f"[Stage 6a] Classified {len(results)} regions by geometry.")
    return results
