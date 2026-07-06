"""
Stage 5: Text/Dimension Extraction per Region (OCR — no AI).

For each detected bounding box, crops the same coordinates from v1 and v2,
upscales, tries multiple rotations, runs EasyOCR, and keeps the
highest-confidence result. Also parses dimension strings into normalized
numeric values + units via regex.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (
    OCR_LANGUAGES,
    OCR_ROTATIONS,
    OCR_MIN_CONFIDENCE,
    OCR_CROP_PADDING,
)

# Hard cap on the longest edge fed into EasyOCR after upscaling. Large diff
# regions (e.g. whole-wall geometry changes) would otherwise be upscaled to
# huge images and make each OCR pass take tens of seconds. Small crops with
# tiny dimension text still get upscaled for readability; large ones get
# downscaled back down to this cap instead.
OCR_MAX_EDGE_PX = 700

# If the very first (0-degree) rotation already yields at least this average
# confidence, skip the remaining rotations — most CAD text is upright, and
# this avoids paying for 4x the OCR passes on every region.
OCR_EARLY_EXIT_CONFIDENCE = 0.75

_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr

                _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
    return _reader


@dataclass
class OcrResult:
    raw_text: str
    confidence: float
    parsed_value: float | None = None
    unit: str | None = None


@dataclass
class RegionCrops:
    crop_v1: np.ndarray
    crop_v2: np.ndarray
    ocr_v1: OcrResult
    ocr_v2: OcrResult


# Feet-inches e.g. 12'-6", 12' 6", 12'-6 1/2"
_FT_IN_RE = re.compile(
    r"(?P<feet>\d+)\s*'[\s-]*(?P<inches>\d+(?:\s+\d+/\d+|\.\d+)?)?\s*\"?"
)
# Plain inches with fraction e.g. 6 1/2", 3.25"
_INCH_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s*\"")
# Millimeters e.g. 450mm, 450 mm
_MM_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
# Diameter / plus-minus prefixed numbers
_DIAMETER_RE = re.compile(r"[ØøΦ]\s*(?P<value>\d+(?:\.\d+)?)")
_PLUSMINUS_RE = re.compile(r"±\s*(?P<value>\d+(?:\.\d+)?)")
_PLAIN_NUMBER_RE = re.compile(r"(?P<value>-?\d+(?:\.\d+)?)")


def _parse_fraction(text: str) -> float:
    text = text.strip()
    if "/" in text:
        parts = text.split()
        whole = 0.0
        frac_part = text
        if len(parts) == 2:
            whole = float(parts[0])
            frac_part = parts[1]
        num, den = frac_part.split("/")
        return whole + (float(num) / float(den))
    return float(text)


def parse_dimension(raw_text: str) -> tuple[float | None, str | None]:
    """
    Parse an OCR'd dimension string into a normalized numeric value + unit.
    Returns (None, None) if no recognizable dimension pattern is found.
    """
    text = raw_text.strip()
    if not text:
        return None, None

    m = _FT_IN_RE.search(text)
    if m and (m.group("inches") is not None or "'" in text):
        feet = float(m.group("feet"))
        inches_str = m.group("inches")
        inches = _parse_fraction(inches_str) if inches_str else 0.0
        total_inches = feet * 12 + inches
        return round(total_inches, 4), "in"

    m = _MM_RE.search(text)
    if m:
        return round(float(m.group("value")), 4), "mm"

    m = _DIAMETER_RE.search(text)
    if m:
        return round(float(m.group("value")), 4), "diameter"

    m = _PLUSMINUS_RE.search(text)
    if m:
        return round(float(m.group("value")), 4), "tolerance"

    m = _INCH_RE.search(text)
    if m:
        try:
            return round(_parse_fraction(m.group("value")), 4), "in"
        except (ValueError, ZeroDivisionError):
            pass

    m = _PLAIN_NUMBER_RE.search(text)
    if m:
        try:
            return round(float(m.group("value")), 4), "unitless"
        except ValueError:
            pass

    return None, None


def _upscale(crop: np.ndarray, factor: int) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    target_h, target_w = h * factor, w * factor
    longest_edge = max(target_h, target_w)
    if longest_edge > OCR_MAX_EDGE_PX:
        scale = OCR_MAX_EDGE_PX / longest_edge
        target_h, target_w = max(1, int(h * factor * scale)), max(1, int(w * factor * scale))
    interp = cv2.INTER_CUBIC if (target_h * target_w) >= (h * w) else cv2.INTER_AREA
    return cv2.resize(crop, (target_w, target_h), interpolation=interp)


def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def ocr_crop_best_rotation(crop: np.ndarray, upscale_factor: int = 3) -> OcrResult:
    """
    Run OCR on a crop at 0/90/180/270 degree rotations and keep the
    highest-confidence non-empty result.
    """
    if crop is None or crop.size == 0:
        return OcrResult(raw_text="", confidence=0.0)

    reader = _get_reader()
    upscaled = _upscale(crop, upscale_factor)

    best = OcrResult(raw_text="", confidence=0.0)
    for angle in OCR_ROTATIONS:
        rotated = _rotate(upscaled, angle)
        try:
            results = reader.readtext(rotated, detail=1, paragraph=False)
        except Exception:
            continue

        if not results:
            continue

        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        combined_text = " ".join(texts).strip()
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        if avg_conf > best.confidence:
            best = OcrResult(raw_text=combined_text, confidence=avg_conf)

        if best.confidence >= OCR_EARLY_EXIT_CONFIDENCE:
            break

    if best.confidence < OCR_MIN_CONFIDENCE:
        best = OcrResult(raw_text=best.raw_text if best.confidence > 0 else "", confidence=best.confidence)

    value, unit = parse_dimension(best.raw_text) if best.raw_text else (None, None)
    best.parsed_value = value
    best.unit = unit
    return best


def crop_with_padding(image: np.ndarray, x: int, y: int, w: int, h: int,
                       padding: int = OCR_CROP_PADDING) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w_img, x + w + padding)
    y1 = min(h_img, y + h + padding)
    return image[y0:y1, x0:x1].copy()


def extract_region_ocr(v1_color: np.ndarray, v2_aligned_color: np.ndarray,
                        box) -> RegionCrops:
    """Crop the same coordinates from v1 and v2 and run OCR on both."""
    crop_v1 = crop_with_padding(v1_color, box.x, box.y, box.w, box.h)
    crop_v2 = crop_with_padding(v2_aligned_color, box.x, box.y, box.w, box.h)

    ocr_v1 = ocr_crop_best_rotation(crop_v1)
    ocr_v2 = ocr_crop_best_rotation(crop_v2)

    return RegionCrops(crop_v1=crop_v1, crop_v2=crop_v2, ocr_v1=ocr_v1, ocr_v2=ocr_v2)


@dataclass
class OcrResultWithBBox:
    raw_text: str
    confidence: float
    parsed_value: float | None
    unit: str | None
    bbox: tuple[int, int, int, int]  # x, y, w, h
    
    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "parsed_value": self.parsed_value,
            "unit": self.unit,
            "bbox": list(self.bbox),
        }


def extract_full_image_ocr(color_img: np.ndarray) -> list[OcrResultWithBBox]:
    """Run OCR on the entire image in a single pass, yielding text coordinates."""
    if color_img is None or color_img.size == 0:
        return []

    reader = _get_reader()
    try:
        raw_results = reader.readtext(color_img, paragraph=False)
    except Exception:
        return []

    results = []
    for bbox, text, confidence in raw_results:
        if confidence < OCR_MIN_CONFIDENCE:
            continue

        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        xmin, ymin = int(min(xs)), int(min(ys))
        xmax, ymax = int(max(xs)), int(max(ys))
        w = max(1, xmax - xmin)
        h = max(1, ymax - ymin)

        val, unit = parse_dimension(text)
        results.append(OcrResultWithBBox(
            raw_text=text,
            confidence=confidence,
            parsed_value=val,
            unit=unit,
            bbox=(xmin, ymin, w, h)
        ))

    return results

