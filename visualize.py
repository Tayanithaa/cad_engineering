from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


CHANGE_COLORS = {
    "Added": (34, 166, 74),
    "Removed": (220, 53, 69),
    "Modified": (245, 190, 20),
    "Possible change": (142, 95, 196),
    "Unchanged": (170, 170, 170),
}


def image_to_base64_png(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    with BytesIO() as buffer:
        Image.fromarray(image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_image(image: np.ndarray, bbox: list[int], pad: int = 6) -> np.ndarray:
    x, y, w, h = bbox
    ih, iw = image.shape[:2]
    return image[max(0, y - pad) : min(ih, y + h + pad), max(0, x - pad) : min(iw, x + w + pad)].copy()


def side_by_side(image_a: np.ndarray, aligned_b: np.ndarray) -> np.ndarray:
    h = max(image_a.shape[0], aligned_b.shape[0])
    w_a = image_a.shape[1]
    w_b = aligned_b.shape[1]
    canvas = np.full((h, w_a + w_b, 3), 255, dtype=np.uint8)
    canvas[: image_a.shape[0], :w_a] = image_a
    canvas[: aligned_b.shape[0], w_a : w_a + w_b] = aligned_b
    return canvas


def annotated_overlay(aligned_b: np.ndarray, records: list[dict], show_unchanged: bool = False) -> np.ndarray:
    overlay = aligned_b.copy()
    for record in records:
        change_type = record.get("change_type", "Modified")
        if change_type == "Unchanged" and not show_unchanged:
            continue
        color = CHANGE_COLORS.get(change_type, CHANGE_COLORS["Modified"])
        x, y, w, h = record["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            overlay,
            record["region_id"],
            (x, max(16, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def ssim_heatmap(diff_map: np.ndarray) -> np.ndarray:
    changed = ((1.0 - np.clip(diff_map, 0.0, 1.0)) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(changed, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def build_visuals(image_a: np.ndarray, aligned_b: np.ndarray, diff_map: np.ndarray, records: list[dict]) -> dict:
    visuals = {
        "side_by_side": side_by_side(image_a, aligned_b),
        "overlay": annotated_overlay(aligned_b, records),
        "heatmap": ssim_heatmap(diff_map),
    }
    print("[Stage 8] Built side-by-side, overlay, and SSIM heatmap visuals.")
    return visuals
