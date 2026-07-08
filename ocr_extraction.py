from __future__ import annotations

import re
from functools import lru_cache

import cv2
import numpy as np

from cad_engineering.config import OCR_CONFIDENCE_THRESHOLD, OCR_ENGINE, OCR_UPSCALE_FACTOR


DIMENSION_RE = re.compile(
    r"(?P<diameter>[Øø])?\s*(?P<sign>[±+-])?\s*"
    r"(?:(?P<feet>\d+)\s*['′]\s*[-–]?\s*)?"
    r"(?P<number>\d+\s*/\s*\d+|\d+(?:\.\d+)?)?\s*"
    r"(?P<inch>[\"″])?\s*(?P<unit>mm|cm|m|ft|in|')?",
    re.IGNORECASE,
)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _crop(image: np.ndarray, bbox: list[int], pad: int = 4) -> np.ndarray:
    x, y, w, h = bbox
    ih, iw = image.shape[:2]
    return image[max(0, y - pad) : min(ih, y + h + pad), max(0, x - pad) : min(iw, x + w + pad)]


def likely_text_crop(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    gray = _gray(crop)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = np.count_nonzero(edges) / max(edges.size, 1)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_strokes = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if 4 <= area <= 600 and h <= max(30, crop.shape[0] * 0.8):
            small_strokes += 1
    return bool(edge_density > 0.015 and small_strokes >= 2)


def _prepare_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    scale = max(1, OCR_UPSCALE_FACTOR)
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = _gray(resized)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@lru_cache(maxsize=2)
def _load_engine(engine_name: str):
    if engine_name == "easyocr":
        import easyocr

        return easyocr.Reader(["en"], gpu=False)

    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def _flatten_paddle_result(result) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    if not result:
        return items
    for page in result:
        if not page:
            continue
        for line in page:
            try:
                text = str(line[1][0])
                conf = float(line[1][1])
                items.append((text, conf))
            except Exception:
                continue
    return items


def _run_ocr_once(crop: np.ndarray, engine_name: str = OCR_ENGINE) -> tuple[str, float]:
    prepared = _prepare_crop(crop)
    if prepared.size == 0:
        return "", 0.0
    try:
        engine = _load_engine(engine_name)
        if engine_name == "easyocr":
            result = engine.readtext(prepared)
            items = [(str(item[1]), float(item[2])) for item in result]
        else:
            result = engine.ocr(prepared, cls=True)
            items = _flatten_paddle_result(result)
    except Exception as exc:
        print(f"[Stage 6b] OCR call failed with {engine_name}: {exc}")
        return "", 0.0
    if not items:
        return "", 0.0
    text = " ".join(item[0] for item in items).strip()
    confidence = float(np.mean([item[1] for item in items]))
    return text, confidence


def ocr_best_rotation(crop: np.ndarray, engine_name: str = OCR_ENGINE) -> tuple[str, float, int]:
    best_text = ""
    best_conf = 0.0
    best_angle = 0
    for angle in (0, 90, 180, 270):
        rotated = crop
        if angle == 90:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(crop, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        text, confidence = _run_ocr_once(rotated, engine_name)
        if confidence > best_conf:
            best_text = text
            best_conf = confidence
            best_angle = angle
    return best_text, best_conf, best_angle


def _parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator.strip()) / float(denominator.strip())
    return float(value)


def parse_dimension_text(text: str) -> dict:
    normalized = " ".join((text or "").replace("O/", "Ø").split())
    match = DIMENSION_RE.search(normalized)
    if not match or not match.group("number"):
        return {"raw": text, "normalized_text": normalized, "parsed_value": None, "unit": None}

    try:
        feet = float(match.group("feet") or 0)
        number = _parse_fraction(match.group("number").replace(" ", ""))
        unit = (match.group("unit") or "").lower()
        if match.group("inch") or feet:
            parsed_value = feet * 12.0 + number
            unit = "in"
        else:
            parsed_value = number
            if unit == "'":
                unit = "ft"
        if match.group("sign") == "-":
            parsed_value *= -1
        return {
            "raw": text,
            "normalized_text": normalized,
            "parsed_value": parsed_value,
            "unit": unit or None,
            "diameter": bool(match.group("diameter")),
            "plus_minus": match.group("sign") == "±",
        }
    except Exception:
        return {"raw": text, "normalized_text": normalized, "parsed_value": None, "unit": None}


def extract_ocr_for_region(image_a: np.ndarray, image_b: np.ndarray, region: dict, engine_name: str = OCR_ENGINE) -> dict:
    bbox = region["bbox"]
    crop_a = _crop(image_a, bbox)
    crop_b = _crop(image_b, bbox)
    likely_text = likely_text_crop(crop_a) or likely_text_crop(crop_b)

    text_a = ""
    conf_a = 0.0
    angle_a = 0
    text_b = ""
    conf_b = 0.0
    angle_b = 0

    if likely_text:
        text_a, conf_a, angle_a = ocr_best_rotation(crop_a, engine_name)
        text_b, conf_b, angle_b = ocr_best_rotation(crop_b, engine_name)

    combined_conf = float(max(conf_a, conf_b))
    parsed_a = parse_dimension_text(text_a)
    parsed_b = parse_dimension_text(text_b)
    has_text = bool(text_a.strip() or text_b.strip())

    return {
        "region_id": region["region_id"],
        "bbox": bbox,
        "likely_text": bool(likely_text or has_text),
        "raw_text_a": text_a,
        "raw_text_b": text_b,
        "parsed_value_a": parsed_a.get("parsed_value"),
        "parsed_value_b": parsed_b.get("parsed_value"),
        "unit": parsed_b.get("unit") or parsed_a.get("unit"),
        "normalized_text_a": parsed_a.get("normalized_text", ""),
        "normalized_text_b": parsed_b.get("normalized_text", ""),
        "ocr_confidence": combined_conf,
        "ocr_confidence_a": conf_a,
        "ocr_confidence_b": conf_b,
        "best_rotation_a": angle_a,
        "best_rotation_b": angle_b,
        "low_confidence": bool(has_text and combined_conf < OCR_CONFIDENCE_THRESHOLD),
    }


def extract_ocr_for_regions(image_a: np.ndarray, image_b: np.ndarray, regions: list[dict], engine_name: str = OCR_ENGINE) -> list[dict]:
    results = []
    for index, region in enumerate(regions, start=1):
        print(f"[Stage 6b] OCR region {index}/{len(regions)} ({region['region_id']})")
        results.append(extract_ocr_for_region(image_a, image_b, region, engine_name))
    text_count = sum(1 for item in results if item["likely_text"])
    print(f"[Stage 6b] OCR checked {len(results)} regions; {text_count} looked text-bearing.")
    return results
