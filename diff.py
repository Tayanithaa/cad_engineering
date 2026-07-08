from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cad_engineering.config import MIN_CONTOUR_AREA


@dataclass
class DiffResult:
    ssim_score: float
    diff_map: np.ndarray
    binary_mask: np.ndarray
    regions: list[dict]
    metadata: dict


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def compute_ssim_diff(image_a: np.ndarray, aligned_b: np.ndarray) -> tuple[float, np.ndarray]:
    gray_a = _gray(image_a)
    gray_b = _gray(aligned_b)
    if gray_a.shape != gray_b.shape:
        raise ValueError("SSIM diff requires aligned images with the same dimensions.")
    diff_map = cv2.absdiff(gray_a, gray_b)
    diff_map_norm = diff_map.astype(np.float32) / 255.0
    return float(1.0 - np.mean(diff_map_norm)), 1.0 - diff_map_norm


def run_structural_diff(image_a: np.ndarray, aligned_b: np.ndarray) -> DiffResult:
    gray_a = _gray(image_a)
    gray_b = _gray(aligned_b)
    
    # 1. Segment drawing lines (lines are dark, background is bright/white)
    _, lines_a = cv2.threshold(gray_a, 220, 255, cv2.THRESH_BINARY_INV)
    _, lines_b = cv2.threshold(gray_b, 220, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Increase edge/line distance tolerance (allows up to 7-pixel alignment variance)
    kernel_tolerance = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    lines_a_dilated = cv2.dilate(lines_a, kernel_tolerance)
    lines_b_dilated = cv2.dilate(lines_b, kernel_tolerance)
    
    # 3. Detect changes (Lines in B that aren't near lines in A, and vice-versa)
    added = cv2.subtract(lines_b, lines_a_dilated)
    removed = cv2.subtract(lines_a, lines_b_dilated)
    diff_mask = cv2.bitwise_or(added, removed)
    
    # 4. Clean up isolated noise pixels
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    # 5. Build normalized diff_map (0.0 = difference, 1.0 = identical)
    diff_map = 1.0 - (diff_mask.astype(np.float32) / 255.0)
    
    # 6. Extract regions (contours)
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    # Set general min area lower (400px) to retain door/window/pillar elements
    min_area_threshold = max(MIN_CONTOUR_AREA * 2, 400)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append(
            {
                "region_id": f"R{len(regions) + 1:03d}",
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": area,
                "contour": contour,
            }
        )
        
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, region in enumerate(regions, start=1):
        region["region_id"] = f"R{index:03d}"
        
    changed_area = int(np.count_nonzero(diff_mask))
    total_area = int(diff_mask.shape[0] * diff_mask.shape[1])
    score = float(1.0 - (changed_area / total_area)) if total_area else 1.0
    
    metadata = {
        "global_ssim_score": score,
        "changed_pixel_area": changed_area,
        "total_pixel_area": total_area,
        "changed_area_percent": float((changed_area / total_area) * 100.0) if total_area else 0.0,
        "region_count": len(regions),
    }
    
    print(f"[Stage 4 & 5] Computed edge-tolerant diff: {len(regions)} regions found.")
    return DiffResult(score, diff_map, diff_mask, regions, metadata)
