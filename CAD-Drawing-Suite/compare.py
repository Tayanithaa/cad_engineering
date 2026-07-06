"""
Stages 6 & 7: Rule-Based Comparison + Merge into Change Records.

Compares v1 vs v2 OCR results for each region using deterministic rules
(no AI), then merges bounding box + comparison outcome into a single
change-record list with region IDs, categories, and quadrant-based
location descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np

from config import NUMERIC_TOLERANCE
from diff import DiffBox, VectorLine, CadObject
from ocr_extraction import RegionCrops


@dataclass
class ChangeRecord:
    region_id: str
    category: str  # Dimension / Text Label / Geometric / Unclassified
    location_description: str
    change_type: str  # Added / Removed / Modified / Unchanged
    v1_value: str
    v2_value: str
    ocr_confidence_v1: float
    ocr_confidence_v2: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    heuristic_note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def location_from_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    """Quadrant + sub-position description using simple image-coordinate logic."""
    cx, cy = x + w / 2, y + h / 2

    vert = "upper" if cy < img_h / 3 else ("lower" if cy > 2 * img_h / 3 else "middle")
    horiz = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")

    if vert == "middle" and horiz == "center":
        return "center"
    return f"{vert}-{horiz}"


def _door_window_heuristic(crop_v1: np.ndarray, crop_v2: np.ndarray) -> str:
    """
    Best-effort symbol heuristic: look for arc-like contours near a
    rectangular gap, which loosely correlates with door/window swing
    symbols in architectural drawings. Clearly labeled as a guess.
    """
    import cv2

    try:
        for crop in (crop_v1, crop_v2):
            if crop is None or crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 5:
                    continue
                area = cv2.contourArea(cnt)
                if area < 40:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # Arcs are elongated/curved: low-to-mid circularity, decent perimeter
                if 0.15 < circularity < 0.65 and perimeter > 60:
                    return "Possible door/window change (heuristic guess, not confirmed)"
    except Exception:
        pass
    return ""


def _analyze_geometric_diff(crop_v1: np.ndarray, crop_v2: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Analyze crops to determine if geometry was added, removed, or shifted/modified."""
    import cv2
    if crop_v1 is None or crop_v2 is None or crop_v1.size == 0 or crop_v2.size == 0:
        return "Geometry modified"

    try:
        # First, try to detect specific door/window size or position changes
        gray1 = cv2.cvtColor(crop_v1, cv2.COLOR_BGR2GRAY) if crop_v1.ndim == 3 else crop_v1
        gray2 = cv2.cvtColor(crop_v2, cv2.COLOR_BGR2GRAY) if crop_v2.ndim == 3 else crop_v2

        # Threshold to get binary ink mask (dark lines on white page)
        _, thresh1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)
        _, thresh2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)

        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours1 and contours2:
            c1 = max(contours1, key=cv2.contourArea)
            c2 = max(contours2, key=cv2.contourArea)

            x1, y1, w1, h1 = cv2.boundingRect(c1)
            x2, y2, w2, h2 = cv2.boundingRect(c2)

            if w1 > 5 and h1 > 5 and w2 > 5 and h2 > 5:
                w_diff_pct = (w2 - w1) / w1
                h_diff_pct = (h2 - h1) / h1

                # Determine structure type based on aspect ratio
                structure_type = "structure"
                aspect_v1 = w1 / h1
                if 0.25 < aspect_v1 < 0.8:
                    structure_type = "door/opening"
                elif 0.8 <= aspect_v1 < 2.2:
                    structure_type = "window/opening"

                if abs(w_diff_pct) > 0.08 or abs(h_diff_pct) > 0.08:
                    change_direction = "increased" if (w_diff_pct + h_diff_pct) > 0 else "decreased"
                    return f"Size of {structure_type} {change_direction} (width by {w_diff_pct*100:+.0f}%, height by {h_diff_pct*100:+.0f}%)"

                shift_x = abs(x2 - x1)
                shift_y = abs(y2 - y1)
                if shift_x > 6 or shift_y > 6:
                    return f"{structure_type.capitalize()} shifted or repositioned"
    except Exception:
        pass

    try:
        # Fallback to general pixel/line density comparison
        gray1 = cv2.cvtColor(crop_v1, cv2.COLOR_BGR2GRAY) if crop_v1.ndim == 3 else crop_v1
        gray2 = cv2.cvtColor(crop_v2, cv2.COLOR_BGR2GRAY) if crop_v2.ndim == 3 else crop_v2
        _, ink1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)
        _, ink2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)
        pixels1 = cv2.countNonZero(ink1)
        pixels2 = cv2.countNonZero(ink2)

        x, y, w, h = bbox
        aspect_ratio = w / h if h != 0 else 1.0

        shape_desc = "compact region"
        if aspect_ratio > 3.0:
            shape_desc = "horizontal line/structure"
        elif aspect_ratio < 0.33:
            shape_desc = "vertical line/structure"

        diff_pct = abs(pixels1 - pixels2) / max(1, min(pixels1, pixels2))

        if pixels2 > pixels1 and diff_pct > 0.15:
            return f"New detail or lines added to a {shape_desc}"
        elif pixels1 > pixels2 and diff_pct > 0.15:
            return f"Lines or detail removed from a {shape_desc}"
        else:
            return f"Geometry or symbol modification in a {shape_desc}"
    except Exception:
        return "Geometry modified"



def classify_architectural_element(box, text_v1: str, text_v2: str, crops: RegionCrops) -> tuple[str, str]:
    """Classify the change region into an architectural element and provide a note."""
    import re
    import cv2
    text_combined = f"{text_v1} {text_v2}".upper().strip()

    # 1. OCR text keyword matching for architectural elements
    if re.search(r"\b(DOOR|DR|D-\d+|D\d+)\b", text_combined):
        return "Door", "Classified as Door via label text"
    if re.search(r"\b(WINDOW|WD|W-\d+|W\d+)\b", text_combined):
        return "Window", "Classified as Window via label text"
    if re.search(r"\b(WALL|PARTITION)\b", text_combined):
        return "Wall", "Classified as Wall via label text"
    if re.search(r"\b(COLUMN|COL|C-\d+)\b", text_combined):
        return "Column", "Classified as Column via label text"
    if re.search(r"\b(STAIR|STAIRCASE|STEPS)\b", text_combined):
        return "Staircase", "Classified as Staircase via label text"
    if re.search(r"\b(ROOM|BEDROOM|BATH|KITCHEN|LIVING|OFFICE|HALL)\b", text_combined):
        return "Room Label", "Classified as Room Label via text"

    # 2. Geometric aspect ratios and sizes
    x, y, w, h = box.x, box.y, box.w, box.h
    aspect_ratio = w / h if h != 0 else 1.0

    # Check for Door swings (using arc heuristic)
    swing_note = _door_window_heuristic(crops.crop_v1, crops.crop_v2)
    if swing_note:
        return "Door/Window Swing", swing_note

    # Long, thin rectangles typically represent walls
    if (w > 150 and h < 25) or (h > 150 and w < 25):
        orientation = "horizontal" if w > h else "vertical"
        return "Wall", f"Classified as Wall ({orientation} structure)"

    # Small square/circular solid blocks typically represent columns
    if 8 < w < 60 and 8 < h < 60 and 0.8 < aspect_ratio < 1.25:
        try:
            gray = cv2.cvtColor(crops.crop_v2, cv2.COLOR_BGR2GRAY) if crops.crop_v2.ndim == 3 else crops.crop_v2
            _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            density = cv2.countNonZero(ink) / (w * h)
            if density > 0.4:
                return "Column", "Classified as structural Column (solid region)"
        except Exception:
            pass

    return "Geometric", ""


def compare_region(box: DiffBox, crops: RegionCrops, img_w: int, img_h: int) -> ChangeRecord:
    """Apply rule-based comparison logic (no AI) to a single region."""
    ocr_v1, ocr_v2 = crops.ocr_v1, crops.ocr_v2
    text_v1 = ocr_v1.raw_text.strip()
    text_v2 = ocr_v2.raw_text.strip()
    location = location_from_bbox(box.x, box.y, box.w, box.h, img_w, img_h)

    has_v1 = bool(text_v1)
    has_v2 = bool(text_v2)

    # Architectural element classification
    arch_cat, arch_note = classify_architectural_element(box, text_v1, text_v2, crops)

    category = arch_cat or "Unclassified"
    change_type = "Modified"
    v1_display = text_v1 or "—"
    v2_display = text_v2 or "—"
    note = arch_note

    if has_v1 and has_v2:
        if ocr_v1.parsed_value is not None and ocr_v2.parsed_value is not None and ocr_v1.unit == ocr_v2.unit:
            category = "Dimension"
            if abs(ocr_v1.parsed_value - ocr_v2.parsed_value) <= NUMERIC_TOLERANCE:
                change_type = "Unchanged"
            else:
                change_type = "Modified"
            v1_display = f"{text_v1} ({ocr_v1.parsed_value} {ocr_v1.unit})"
            v2_display = f"{text_v2} ({ocr_v2.parsed_value} {ocr_v2.unit})"
        else:
            category = arch_cat or "Text Label"
            if _normalize_text(text_v1) == _normalize_text(text_v2):
                change_type = "Unchanged"
            else:
                change_type = "Modified"
    elif has_v2 and not has_v1:
        category = arch_cat or ("Dimension" if ocr_v2.parsed_value is not None else "Text Label")
        change_type = "Added"
    elif has_v1 and not has_v2:
        category = arch_cat or ("Dimension" if ocr_v1.parsed_value is not None else "Text Label")
        change_type = "Removed"
    else:
        category = arch_cat or "Geometric"
        change_type = "Modified"
        geom_desc = _analyze_geometric_diff(crops.crop_v1, crops.crop_v2, (box.x, box.y, box.w, box.h))
        v1_display = "Original geometry"
        v2_display = geom_desc

    return ChangeRecord(
        region_id="",  # assigned by caller
        category=category,
        location_description=location,
        change_type=change_type,
        v1_value=v1_display,
        v2_value=v2_display,
        ocr_confidence_v1=round(ocr_v1.confidence, 3),
        ocr_confidence_v2=round(ocr_v2.confidence, 3),
        bbox=(box.x, box.y, box.w, box.h),
        heuristic_note=note,
    )


def build_change_records(boxes: list[DiffBox], crops_list: list[RegionCrops],
                          img_w: int, img_h: int) -> list[ChangeRecord]:
    """Assign region IDs and merge bbox + comparison results into records,
    filtering out Unchanged regions from the final change log."""
    records: list[ChangeRecord] = []
    for idx, (box, crops) in enumerate(zip(boxes, crops_list), start=1):
        record = compare_region(box, crops, img_w, img_h)
        if record.change_type == "Unchanged":
            continue
        record.region_id = f"R-{idx:03d}"
        records.append(record)

    # Re-number sequentially after filtering so IDs stay contiguous
    for i, record in enumerate(records, start=1):
        record.region_id = f"R-{i:03d}"

    return records


def build_vector_change_records(
    matched: list[VectorLine],
    added: list[VectorLine],
    removed: list[VectorLine],
    modified: list[VectorLine],
    shifted: list[VectorLine],
    img_w: int,
    img_h: int
) -> list[ChangeRecord]:
    """Compile all changed vector lines into structured ChangeRecord entities for the report."""
    records = []

    # Process all changed lines (ignore completely identical Matched lines)
    all_changes = []  # list of (line, change_type)
    for l in added:
        all_changes.append((l, "Added"))
    for l in removed:
        all_changes.append((l, "Removed"))
    for l in modified:
        all_changes.append((l, "Modified"))
    for l in shifted:
        all_changes.append((l, "Shifted"))

    # Sort changes by their Y coordinate (then X) to list them logically (top-to-bottom, left-to-right)
    def get_sort_key(item):
        l = item[0]
        return (min(l.y1, l.y2), min(l.x1, l.x2))

    all_changes.sort(key=get_sort_key)

    for idx, (l, ct) in enumerate(all_changes, start=1):
        x = int(min(l.x1, l.x2))
        y = int(min(l.y1, l.y2))
        w = max(1, int(abs(l.x2 - l.x1)))
        h = max(1, int(abs(l.y2 - l.y1)))

        loc = location_from_bbox(x, y, w, h, img_w, img_h)

        # Standard dimension/wall labeling classification
        if w > 120 and h < 6:
            category = "Wall Segment (Horizontal)"
        elif h > 120 and w < 6:
            category = "Wall Segment (Vertical)"
        elif w < 25 and h < 25:
            category = "Detail Element"
        else:
            category = "Structural Line"

        v1_val = "—"
        v2_val = "—"

        if ct == "Added":
            v2_val = f"Line (ID {l.id}) from ({l.x1:.1f}, {l.y1:.1f}) to ({l.x2:.1f}, {l.y2:.1f}), len: {l.length:.1f}px"
        elif ct == "Removed":
            v1_val = f"Line (ID {l.id}) from ({l.x1:.1f}, {l.y1:.1f}) to ({l.x2:.1f}, {l.y2:.1f}), len: {l.length:.1f}px"
        elif ct in ("Modified", "Shifted"):
            v1_val = f"Line (ID {l.id}) len: {l.length:.1f}px"
            v2_val = l.change_description

        records.append(ChangeRecord(
            region_id=f"L-{idx:03d}",
            category=category,
            location_description=loc,
            change_type=ct,
            v1_value=v1_val,
            v2_value=v2_val,
            ocr_confidence_v1=1.0,
            ocr_confidence_v2=1.0,
            bbox=(x, y, w, h),
            heuristic_note=l.change_description
        ))

    return records


def build_object_change_records(
    matched_results: list[tuple[CadObject | None, CadObject | None, str, str]],
    img_w: int,
    img_h: int
) -> list[ChangeRecord]:
    """Compile matched objects into structured ChangeRecord entities for the report."""
    records = []

    # Filter out UNCHANGED objects from final log
    filtered_results = [(o1, o2, status, desc) for o1, o2, status, desc in matched_results if status != "UNCHANGED"]

    # Sort them by Y coordinate of the object
    def get_sort_key(item):
        o1, o2, _, _ = item
        obj = o2 if o2 is not None else o1
        return (obj.bbox[1], obj.bbox[0])

    filtered_results.sort(key=get_sort_key)

    for idx, (o1, o2, status, desc) in enumerate(filtered_results, start=1):
        obj = o2 if o2 is not None else o1
        x, y, w, h = obj.bbox

        # Format values
        v1_val = "—"
        v2_val = "—"

        if status == "ADDED":
            v2_val = desc
        elif status == "REMOVED":
            v1_val = desc
        elif status in ("MODIFIED", "MOVED"):
            if o1 is not None and o2 is not None:
                if o1.type in ("Dimension", "Text Label"):
                    v1_val = f"'{o1.geometry.get('text')}'"
                    v2_val = f"'{o2.geometry.get('text')}'"
                else:
                    v1_val = f"{o1.type} at ({o1.bbox[0]}, {o1.bbox[1]})"
                    v2_val = desc

        records.append(ChangeRecord(
            region_id=obj.id,
            category=obj.type,
            location_description=obj.location,
            change_type=status.capitalize() if status != "MOVED" else "Moved",
            v1_value=v1_val,
            v2_value=v2_val,
            ocr_confidence_v1=1.0,
            ocr_confidence_v2=1.0,
            bbox=obj.bbox,
            heuristic_note=desc
        ))

    return records


