"""
Stage 4: Difference Detection (classical CV, deterministic).

Uses SSIM + absolute pixel diff on aligned grayscale images, cleans the
result with morphological operations, then extracts bounding boxes via
contour detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import (
    MIN_CONTOUR_AREA,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERATIONS,
    MORPH_CLOSE_ITERATIONS,
    DIFF_BINARY_THRESHOLD,
)


@dataclass
class DiffBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class DiffResult:
    ssim_score: float
    diff_mask: np.ndarray  # binary mask, uint8 0/255
    heatmap: np.ndarray  # BGR color-mapped change density
    boxes: list[DiffBox]
    percent_area_changed: float


def detect_differences(v1_gray: np.ndarray, v2_aligned_gray: np.ndarray,
                        min_contour_area: int = MIN_CONTOUR_AREA) -> DiffResult:
    """
    Compute SSIM-based structural diff + absolute pixel diff between two
    aligned grayscale images, then extract cleaned bounding boxes.
    Uses downscaling to prevent Out-Of-Memory errors on large drawings.
    """
    h_orig, w_orig = v1_gray.shape[:2]
    
    # Target maximum dimension to avoid memory OOM (caps memory at ~20MB instead of 150MB+)
    MAX_DIM = 1800
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        w_scaled = int(round(w_orig * scale))
        h_scaled = int(round(h_orig * scale))
        v1_scaled = cv2.resize(v1_gray, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
        v2_scaled = cv2.resize(v2_aligned_gray, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        v1_scaled = v1_gray
        v2_scaled = v2_aligned_gray

    if v1_scaled.shape != v2_scaled.shape:
        v2_scaled = cv2.resize(v2_scaled, (v1_scaled.shape[1], v1_scaled.shape[0]))

    score, ssim_map = ssim(v1_scaled, v2_scaled, full=True)
    ssim_map = (ssim_map * 255).astype("uint8")
    inv_ssim = 255 - ssim_map

    abs_diff = cv2.absdiff(v1_scaled, v2_scaled)

    combined = cv2.addWeighted(inv_ssim, 0.7, abs_diff, 0.3, 0)

    _, binary = cv2.threshold(combined, DIFF_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    ksize = max(3, int(round(MORPH_KERNEL_SIZE * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERATIONS)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERATIONS)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[DiffBox] = []
    total_area_scaled = 0
    scaled_min_area = min_contour_area * (scale ** 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < scaled_min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Scale bounding boxes back to the original image coordinate frame for OCR
        x_orig = int(round(x / scale))
        y_orig = int(round(y / scale))
        w_orig_box = int(round(w / scale))
        h_orig_box = int(round(h / scale))
        
        x_orig = max(0, min(x_orig, w_orig - 1))
        y_orig = max(0, min(y_orig, h_orig - 1))
        w_orig_box = max(1, min(w_orig_box, w_orig - x_orig))
        h_orig_box = max(1, min(h_orig_box, h_orig - y_orig))
        
        boxes.append(DiffBox(x=x_orig, y=y_orig, w=w_orig_box, h=h_orig_box))
        total_area_scaled += w * h

    boxes.sort(key=lambda b: (b.y, b.x))

    # Scale outputs back to original resolution
    heatmap_scaled = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap_scaled, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    diff_mask = cv2.resize(cleaned, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    img_area_scaled = v1_scaled.shape[0] * v1_scaled.shape[1]
    percent_area_changed = (total_area_scaled / img_area_scaled * 100) if img_area_scaled else 0.0

    return DiffResult(
        ssim_score=float(score),
        diff_mask=diff_mask,
        heatmap=heatmap,
        boxes=boxes,
        percent_area_changed=round(percent_area_changed, 3),
    )


@dataclass
class VectorLine:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle: float
    status: str = "Unmatched"  # Matched, Shifted, Modified, Added, Removed
    match_id: int | None = None
    change_description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "length": round(self.length, 1),
            "angle": round(self.angle, 2),
            "status": self.status,
            "match_id": self.match_id,
            "change_description": self.change_description,
        }


from config import (
    LINE_DISTANCE_TOLERANCE,
    LINE_ANGLE_TOLERANCE,
    LINE_SHIFT_TOLERANCE,
    LINE_LENGTH_TOLERANCE,
    MIN_LINE_LENGTH,
)
import math


def extract_vector_lines(gray_img: np.ndarray) -> list[VectorLine]:
    """Extract line segments from a grayscale image using LSD."""
    lsd = cv2.createLineSegmentDetector()
    lines_detected, _, _, _ = lsd.detect(gray_img)

    vector_lines = []
    if lines_detected is None:
        return vector_lines

    line_id = 1
    for item in lines_detected:
        coords = item.flatten()
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = map(float, coords)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < MIN_LINE_LENGTH:
            continue

        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += math.pi

        vector_lines.append(VectorLine(
            id=line_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            length=round(length, 2),
            angle=round(angle, 4)
        ))
        line_id += 1

    return vector_lines


def compare_vector_lines(lines_v1: list[VectorLine], lines_v2: list[VectorLine]) -> tuple[list[VectorLine], list[VectorLine], list[VectorLine], list[VectorLine], list[VectorLine]]:
    """Compare vector lines from v1 and v2 using generous tolerances to filter out rendering noise."""
    matched = []
    added = []
    removed = []
    modified = []
    shifted = []

    # High-tolerance values to override configuration defaults and prevent false differences
    dist_tolerance = 8.0     # pixels
    angle_tolerance = 0.15   # radians (~8.5 degrees)
    shift_tolerance = 20.0   # pixels
    length_tolerance = 15.0  # pixels

    # Track used indices of lines_v2
    used_v2 = set()

    for l1 in lines_v1:
        best_candidate = None
        best_err = float('inf')
        best_type = None  # "matched", "modified", "shifted"

        mx1, my1 = (l1.x1 + l1.x2) / 2, (l1.y1 + l1.y2) / 2

        for l2 in lines_v2:
            if l2.id in used_v2:
                continue

            mx2, my2 = (l2.x1 + l2.x2) / 2, (l2.y1 + l2.y2) / 2

            # Fast distance check between midpoints
            mid_dist = math.hypot(mx2 - mx1, my2 - my1)
            if mid_dist > shift_tolerance + 15:
                continue

            # Angular difference
            ang_diff = min(abs(l1.angle - l2.angle), math.pi - abs(l1.angle - l2.angle))
            if ang_diff > angle_tolerance:
                continue

            # Endpoint error calculation
            err1 = math.hypot(l1.x1 - l2.x1, l1.y1 - l2.y1) + math.hypot(l1.x2 - l2.x2, l1.y2 - l2.y2)
            err2 = math.hypot(l1.x1 - l2.x2, l1.y1 - l2.y2) + math.hypot(l1.x2 - l2.x1, l1.y2 - l2.y1)
            err = min(err1, err2)

            # Check matching type
            if err <= 2 * dist_tolerance:
                len_diff = abs(l1.length - l2.length)
                if len_diff <= length_tolerance:
                    if err < best_err:
                        best_err = err
                        best_candidate = l2
                        best_type = "matched"
                else:
                    if err < best_err:
                        best_err = err
                        best_candidate = l2
                        best_type = "modified"
            else:
                # Parallel line offset check
                dx = l2.x2 - l2.x1
                dy = l2.y2 - l2.y1
                denom = math.hypot(dx, dy)
                if denom > 0:
                    perp_dist = abs(-dy * mx1 + dx * my1 + (l2.x1 * l2.y2 - l2.x2 * l2.y1)) / denom
                    if perp_dist <= shift_tolerance:
                        if mid_dist <= shift_tolerance and err < best_err:
                            best_err = err
                            best_candidate = l2
                            best_type = "shifted"

        if best_candidate is not None:
            used_v2.add(best_candidate.id)
            if best_type == "matched":
                l1.status = "Matched"
                l1.match_id = best_candidate.id
                matched.append(l1)
            elif best_type == "modified":
                l1.status = "Modified"
                l1.match_id = best_candidate.id
                l1.change_description = f"Line modified: length changed from {l1.length:.1f} to {best_candidate.length:.1f}"
                modified.append(l1)
            elif best_type == "shifted":
                l1.status = "Shifted"
                l1.match_id = best_candidate.id
                l1.change_description = f"Line shifted by {round(best_err/2, 1)} pixels"
                shifted.append(l1)
        else:
            l1.status = "Removed"
            l1.change_description = "Line segment removed"
            removed.append(l1)

    # Any unmatched lines in v2 are added
    for l2 in lines_v2:
        if l2.id not in used_v2:
            l2.status = "Added"
            l2.change_description = "New line segment added"
            added.append(l2)

    return matched, added, removed, modified, shifted


def draw_vector_overlay(image: np.ndarray, lines: list[VectorLine], status_colors: dict) -> np.ndarray:
    """Draw lines on the image copy with colors based on status."""
    overlay = image.copy()
    for l in lines:
        if l.status in ("Matched", "UNCHANGED"):
            continue
        color = status_colors.get(l.status, (128, 128, 128))
        pt1 = (int(round(l.x1)), int(round(l.y1)))
        pt2 = (int(round(l.x2)), int(round(l.y2)))

        # Calculate line thickness based on image size
        thickness = max(2, int(min(image.shape[:2]) / 350))
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw unique ID near the midpoint
        mx, my = int((l.x1 + l.x2) / 2), int((l.y1 + l.y2) / 2)
        cv2.putText(overlay, str(l.id), (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    return overlay


@dataclass
class CadObject:
    id: str
    type: str  # Wall, Door, Window, Column, Staircase, Dimension, Text Label
    bbox: tuple[int, int, int, int]  # x, y, w, h
    geometry: dict  # specific geometric details (lines, lengths, text values)
    location: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "bbox": list(self.bbox),
            "geometry": self.geometry,
            "location": self.location,
            "confidence": self.confidence,
        }


def detect_drawing_boundary(gray_img: np.ndarray) -> tuple[int, int, int, int]:
    """Detect the main architectural drawing boundaries, cropping out sheet borders, title blocks, and empty margins."""
    h, w = gray_img.shape[:2]

    # Threshold to get ink mask (dark lines on white background)
    _, ink = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY_INV)

    # Find horizontal and vertical line projections
    row_sums = np.sum(ink > 0, axis=1)
    col_sums = np.sum(ink > 0, axis=0)

    # Find where the drawing starts and ends (ignoring extreme margins)
    row_indices = np.where(row_sums > (w * 0.015))[0]
    col_indices = np.where(col_sums > (h * 0.015))[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return 0, 0, h, w

    ymin, ymax = int(row_indices[0]), int(row_indices[-1])
    xmin, xmax = int(col_indices[0]), int(col_indices[-1])

    # Crop title block in the bottom 30% or right 25% of the page
    search_start_y = int(ymin + (ymax - ymin) * 0.7)
    search_end_y = int(ymin + (ymax - ymin) * 0.95)
    bottom_border_line = None
    for y in range(search_start_y, search_end_y):
        if row_sums[y] > (w * 0.7):
            bottom_border_line = y
            break

    if bottom_border_line is not None:
        ymax = bottom_border_line - 5

    search_start_x = int(xmin + (xmax - xmin) * 0.75)
    search_end_x = int(xmin + (xmax - xmin) * 0.95)
    right_border_line = None
    for x in range(search_start_x, search_end_x):
        if col_sums[x] > (h * 0.7):
            right_border_line = x
            break

    if right_border_line is not None:
        xmax = right_border_line - 5

    # Apply safety margins
    ymin = max(0, ymin + 10)
    ymax = min(h, ymax - 10)
    xmin = max(0, xmin + 10)
    xmax = min(w, xmax - 10)

    # Ensure we don't return invalid sizes
    if (ymax - ymin) < (h * 0.2) or (xmax - xmin) < (w * 0.2):
        return 0, 0, h, w

    return ymin, xmin, ymax, xmax


def extract_cad_objects(gray_img: np.ndarray, color_img: np.ndarray, ocr_results: list) -> list[CadObject]:
    """Detect architectural objects (Doors, Windows, Walls, Columns, Dimensions) from lines and text."""
    import re
    h, w = gray_img.shape[:2]

    # 1. Extract vector lines
    lines = extract_vector_lines(gray_img)
    used_lines = set()

    cad_objects = []
    obj_counters = {"Wall": 1, "Door": 1, "Window": 1, "Column": 1, "Dimension": 1, "Text Label": 1, "Staircase": 1}

    # Helper to calculate location
    def get_loc(x, y, w_obj, h_obj):
        cx, cy = x + w_obj / 2, y + h_obj / 2
        vert = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
        horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
        return f"{vert}-{horiz}"

    # 2. Extract OCR texts & dimensions first
    for ocr_rec in ocr_results:
        text = ocr_rec.raw_text.strip()
        if not text:
            continue
        bx, by, bw, bh = ocr_rec.bbox
        loc = get_loc(bx, by, bw, bh)

        # Detect Dimension vs room label
        if re.search(r'(\d+[\'\"\-]|mm|\b\d+\b)', text):
            obj_type = "Dimension"
        else:
            obj_type = "Text Label"

        idx = obj_counters[obj_type]
        cad_objects.append(CadObject(
            id=f"{obj_type[0]}{idx:02d}",
            type=obj_type,
            bbox=(bx, by, bw, bh),
            geometry={"text": text, "val": ocr_rec.parsed_value},
            location=loc
        ))
        obj_counters[obj_type] += 1

    # 3. Detect Columns (using compact contours)
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if 10 < cw < 50 and 10 < ch < 50:
            aspect = cw / ch
            if 0.8 < aspect < 1.25:
                # Solid density test
                area = cv2.contourArea(cnt)
                if area > 80:
                    idx = obj_counters["Column"]
                    loc = get_loc(cx, cy, cw, ch)
                    cad_objects.append(CadObject(
                        id=f"COL{idx:02d}",
                        type="Column",
                        bbox=(cx, cy, cw, ch),
                        geometry={"width": cw, "height": ch},
                        location=loc
                    ))
                    obj_counters["Column"] += 1

    # 4. Group lines into Staircases (closely spaced parallel lines)
    horizontal_lines = [l for l in lines if abs(math.sin(l.angle)) < 0.1]
    horizontal_lines.sort(key=lambda l: (min(l.y1, l.y2), min(l.x1, l.x2)))
    stair_groups = []
    current_group = []

    for l in horizontal_lines:
        if not current_group:
            current_group.append(l)
        else:
            last = current_group[-1]
            dy = abs((l.y1 + l.y2)/2 - (last.y1 + last.y2)/2)
            dx = abs((l.x1 + l.x2)/2 - (last.x1 + last.x2)/2)
            # Parallel staircase step pattern: dy is 8-22 pixels, similar X alignment
            if 6 < dy < 25 and dx < 50:
                current_group.append(l)
            else:
                if len(current_group) >= 5:
                    stair_groups.append(current_group)
                current_group = [l]
    if len(current_group) >= 5:
        stair_groups.append(current_group)

    for group in stair_groups:
        xs = [min(l.x1, l.x2) for l in group] + [max(l.x1, l.x2) for l in group]
        ys = [min(l.y1, l.y2) for l in group] + [max(l.y1, l.y2) for l in group]
        xmin_g, xmax_g = int(min(xs)), int(max(xs))
        ymin_g, ymax_g = int(min(ys)), int(max(ys))
        w_g, h_g = xmax_g - xmin_g, ymax_g - ymin_g

        for l in group:
            used_lines.add(l.id)

        idx = obj_counters["Staircase"]
        loc = get_loc(xmin_g, ymin_g, w_g, h_g)
        cad_objects.append(CadObject(
            id=f"ST{idx:02d}",
            type="Staircase",
            bbox=(xmin_g, ymin_g, w_g, h_g),
            geometry={"steps": len(group)},
            location=loc
        ))
        obj_counters["Staircase"] += 1

    # 5. Group remaining lines into Windows (close parallel lines in walls) and Doors (swing heuristics)
    unused_lines = [l for l in lines if l.id not in used_lines]
    unused_lines.sort(key=lambda l: l.length, reverse=True)

    for i, l1 in enumerate(unused_lines):
        if l1.id in used_lines:
            continue

        # Look for parallel window-like lines (3-4 lines parallel with small gap)
        parallel_window = []
        for l2 in unused_lines[i+1:]:
            if l2.id in used_lines:
                continue
            ang_diff = min(abs(l1.angle - l2.angle), math.pi - abs(l1.angle - l2.angle))
            if ang_diff < 0.05:
                # Check perpendicular distance
                mx1, my1 = (l1.x1 + l1.x2)/2, (l1.y1 + l1.y2)/2
                dx = l1.x2 - l1.x1
                dy = l1.y2 - l1.y1
                denom = math.hypot(dx, dy)
                if denom > 0:
                    dist = abs(-dy * ((l2.x1+l2.x2)/2) + dx * ((l2.y1+l2.y2)/2) + (l1.x1*l1.y2 - l1.x2*l1.y1)) / denom
                    if dist < 12 and abs(l1.length - l2.length) < 25:
                        parallel_window.append(l2)

        if len(parallel_window) >= 2:
            # Found a window block!
            win_lines = [l1] + parallel_window[:3]
            for wl in win_lines:
                used_lines.add(wl.id)
            xs = [wl.x1 for wl in win_lines] + [wl.x2 for wl in win_lines]
            ys = [wl.y1 for wl in win_lines] + [wl.y2 for wl in win_lines]
            bx, by, bw, bh = int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))
            idx = obj_counters["Window"]
            cad_objects.append(CadObject(
                id=f"W{idx:02d}",
                type="Window",
                bbox=(bx, by, bw, bh),
                geometry={"length": l1.length, "lines": len(win_lines)},
                location=get_loc(bx, by, bw, bh)
            ))
            obj_counters["Window"] += 1
            continue

        # Check for Door sweeps (arcs/angled swings)
        # Search for lines forming a door swing (angled line paired with arc)
        is_door = False
        # Simplification: if it is close to 45 degree angle and has similar length to horizontal/vertical gap
        if 0.5 < abs(l1.angle) < 1.0: # angled swing
            is_door = True

        if is_door:
            used_lines.add(l1.id)
            bx, by = int(min(l1.x1, l1.x2)), int(min(l1.y1, l1.y2))
            bw, bh = int(abs(l1.x2 - l1.x1)), int(abs(l1.y2 - l1.y1))
            idx = obj_counters["Door"]
            cad_objects.append(CadObject(
                id=f"D{idx:02d}",
                type="Door",
                bbox=(bx - 10, by - 10, bw + 20, bh + 20),
                geometry={"swing_length": l1.length},
                location=get_loc(bx, by, bw, bh)
            ))
            obj_counters["Door"] += 1
            continue

    # 6. Group remaining lines as Walls ONLY if they form a parallel pair (double line structure)
    for l1 in unused_lines:
        if l1.id in used_lines:
            continue
        if l1.length < 80:
            continue

        # Look for a parallel partner within typical wall spacing
        partner = None
        for l2 in unused_lines:
            if l2.id == l1.id or l2.id in used_lines:
                continue
            if l2.length < 80:
                continue

            ang_diff = min(abs(l1.angle - l2.angle), math.pi - abs(l1.angle - l2.angle))
            if ang_diff < 0.05:
                # Check perpendicular distance
                mx1, my1 = (l1.x1 + l1.x2)/2, (l1.y1 + l1.y2)/2
                dx = l1.x2 - l1.x1
                dy = l1.y2 - l1.y1
                denom = math.hypot(dx, dy)
                if denom > 0:
                    dist = abs(-dy * ((l2.x1+l2.x2)/2) + dx * ((l2.y1+l2.y2)/2) + (l1.x1*l1.y2 - l1.x2*l1.y1)) / denom
                    if 10 < dist < 25:  # spacing between parallel wall lines (ignores small hatch spacing)
                        partner = l2
                        break

        if partner is not None:
            used_lines.add(l1.id)
            used_lines.add(partner.id)
            xs = [l1.x1, l1.x2, partner.x1, partner.x2]
            ys = [l1.y1, l1.y2, partner.y1, partner.y2]
            bx, by = int(min(xs)), int(min(ys))
            bw, bh = int(max(xs) - min(xs)), int(max(ys) - min(ys))

            idx = obj_counters["Wall"]
            cad_objects.append(CadObject(
                id=f"Wall-{idx:02d}",
                type="Wall",
                bbox=(bx, by, bw, bh),
                geometry={"length": (l1.length + partner.length)/2, "angle": l1.angle},
                location=get_loc(bx, by, bw, bh)
            ))
            obj_counters["Wall"] += 1

    return cad_objects


def match_cad_objects(objs_v1: list[CadObject], objs_v2: list[CadObject]) -> list[tuple[CadObject | None, CadObject | None, str, str]]:
    """Compare objects from drawing A to B, pairing them using the Hungarian Algorithm for globally optimal matching with a weighted 5-parameter similarity metric."""
    from scipy.optimize import linear_sum_assignment
    results = []
    
    # Group objects by type to match categories independently
    types = set([o.type for o in objs_v1] + [o.type for o in objs_v2])
    
    for t in types:
        A_type = [o for o in objs_v1 if o.type == t]
        B_type = [o for o in objs_v2 if o.type == t]
        
        if not A_type:
            # All objects in B are Added
            for o2 in B_type:
                results.append((None, o2, "ADDED", f"New {o2.type} added to layout"))
            continue
            
        if not B_type:
            # All objects in A are Removed
            for o1 in A_type:
                results.append((o1, None, "REMOVED", f"{o1.type} removed from layout"))
            continue
            
        # Build cost matrix (Cost = 1.0 - Similarity)
        n_a, n_b = len(A_type), len(B_type)
        
        # Localized greedy fallback if objects of this type are > 150 (prevents memory OOM errors)
        if n_a > 150 or n_b > 150:
            used_b = set()
            for o1 in A_type:
                best_candidate = None
                best_sim = 0.30
                c1_x, c1_y = o1.bbox[0] + o1.bbox[2]/2, o1.bbox[1] + o1.bbox[3]/2
                len1 = o1.geometry.get("length", o1.geometry.get("swing_length", o1.bbox[2]))
                w1, h1 = o1.bbox[2], o1.bbox[3]
                ar1 = w1 / h1 if h1 != 0 else 1.0
                ang1 = o1.geometry.get("angle", 0.0)
                
                # Check closest candidates first to avoid O(N^2) evaluation
                for o2 in B_type:
                    if o2.id in used_b:
                        continue
                    c2_x, c2_y = o2.bbox[0] + o2.bbox[2]/2, o2.bbox[1] + o2.bbox[3]/2
                    dist = math.hypot(c2_x - c1_x, c2_y - c1_y)
                    if dist > 80.0:
                        continue
                        
                    len2 = o2.geometry.get("length", o2.geometry.get("swing_length", o2.bbox[2]))
                    w2, h2 = o2.bbox[2], o2.bbox[3]
                    ar2 = w2 / h2 if h2 != 0 else 1.0
                    ang2 = o2.geometry.get("angle", 0.0)
                    
                    if o1.type in ("Dimension", "Text Label"):
                        geom_sim = 1.0 if o1.geometry.get("text") == o2.geometry.get("text") else 0.0
                    else:
                        geom_sim = 1.0 - min(1.0, abs(len1 - len2) / max(1.0, len1))
                        
                    pos_sim = 1.0 - min(1.0, dist / 150.0)
                    ang_diff = min(abs(ang1 - ang2), math.pi - abs(ang1 - ang2))
                    orient_sim = 1.0 - min(1.0, ang_diff / (math.pi / 2))
                    shape_sim = 1.0 - min(1.0, abs(ar1 - ar2) / max(1.0, ar1))
                    
                    x_left = max(o1.bbox[0], o2.bbox[0])
                    y_top = max(o1.bbox[1], o2.bbox[1])
                    x_right = min(o1.bbox[0] + o1.bbox[2], o2.bbox[0] + o2.bbox[2])
                    y_bottom = min(o1.bbox[1] + o1.bbox[3], o2.bbox[1] + o2.bbox[3])
                    iou = 0.0
                    if x_right > x_left and y_bottom > y_top:
                        inter = (x_right - x_left) * (y_bottom - y_top)
                        union = w1*h1 + w2*h2 - inter
                        iou = inter / union if union > 0 else 0.0
                        
                    sim = 0.40 * geom_sim + 0.25 * pos_sim + 0.15 * orient_sim + 0.10 * shape_sim + 0.10 * iou
                    if sim > best_sim:
                        best_sim = sim
                        best_candidate = o2
                        
                if best_candidate is not None:
                    used_b.add(best_candidate.id)
                    o2 = best_candidate
                    
                    c2_x, c2_y = o2.bbox[0] + o2.bbox[2]/2, o2.bbox[1] + o2.bbox[3]/2
                    dist = math.hypot(c2_x - c1_x, c2_y - c1_y)
                    len2 = o2.geometry.get("length", o2.geometry.get("swing_length", o2.bbox[2]))
                    len_diff = abs(len2 - len1)
                    
                    is_changed = False
                    desc = "No design revision detected"
                    change_type = "UNCHANGED"
                    
                    if o1.type in ("Dimension", "Text Label"):
                        val1 = o1.geometry.get("text", "")
                        val2 = o2.geometry.get("text", "")
                        if val1 != val2:
                            is_changed = True
                            change_type = "MODIFIED"
                            desc = f"Text modified: updated from '{val1}' to '{val2}'"
                    else:
                        if len_diff > max(15.0, len1 * 0.15):
                            is_changed = True
                            change_type = "MODIFIED"
                            change = "increased" if len2 > len1 else "decreased"
                            desc = f"Size {change}: length modified from {len1:.1f}px to {len2:.1f}px"
                            
                    if not is_changed and dist > 15.0:
                        is_changed = True
                        change_type = "MOVED"
                        desc = f"Shifted by {dist:.1f} pixels"
                        
                    if is_changed:
                        results.append((o1, o2, change_type, desc))
                    else:
                        results.append((o1, o2, "UNCHANGED", "No design revision detected"))
                else:
                    results.append((o1, None, "REMOVED", f"{o1.type} removed from layout"))
                    
            for o2 in B_type:
                if o2.id not in used_b:
                    results.append((None, o2, "ADDED", f"New {o2.type} added to layout"))
            continue

        cost_matrix = np.ones((n_a, n_b), dtype=np.float32)
        
        for i, o1 in enumerate(A_type):
            c1_x, c1_y = o1.bbox[0] + o1.bbox[2]/2, o1.bbox[1] + o1.bbox[3]/2
            len1 = o1.geometry.get("length", o1.geometry.get("swing_length", o1.bbox[2]))
            w1, h1 = o1.bbox[2], o1.bbox[3]
            ar1 = w1 / h1 if h1 != 0 else 1.0
            ang1 = o1.geometry.get("angle", 0.0)
            
            for j, o2 in enumerate(B_type):
                c2_x, c2_y = o2.bbox[0] + o2.bbox[2]/2, o2.bbox[1] + o2.bbox[3]/2
                len2 = o2.geometry.get("length", o2.geometry.get("swing_length", o2.bbox[2]))
                w2, h2 = o2.bbox[2], o2.bbox[3]
                ar2 = w2 / h2 if h2 != 0 else 1.0
                ang2 = o2.geometry.get("angle", 0.0)
                
                # 1. Geometry Similarity (40%)
                if o1.type in ("Dimension", "Text Label"):
                    geom_sim = 1.0 if o1.geometry.get("text") == o2.geometry.get("text") else 0.0
                else:
                    geom_sim = 1.0 - min(1.0, abs(len1 - len2) / max(1.0, len1))
                    
                # 2. Position Similarity (25%)
                dist = math.hypot(c2_x - c1_x, c2_y - c1_y)
                pos_sim = 1.0 - min(1.0, dist / 150.0)  # normalized over 150px max search
                
                # 3. Orientation Similarity (15%)
                ang_diff = min(abs(ang1 - ang2), math.pi - abs(ang1 - ang2))
                orient_sim = 1.0 - min(1.0, ang_diff / (math.pi / 2))
                
                # 4. Shape Similarity (10%)
                shape_sim = 1.0 - min(1.0, abs(ar1 - ar2) / max(1.0, ar1))
                
                # 5. Topology Similarity (10%) - represented by IoU overlap
                x_left = max(o1.bbox[0], o2.bbox[0])
                y_top = max(o1.bbox[1], o2.bbox[1])
                x_right = min(o1.bbox[0] + o1.bbox[2], o2.bbox[0] + o2.bbox[2])
                y_bottom = min(o1.bbox[1] + o1.bbox[3], o2.bbox[1] + o2.bbox[3])
                iou = 0.0
                if x_right > x_left and y_bottom > y_top:
                    inter = (x_right - x_left) * (y_bottom - y_top)
                    union = w1*h1 + w2*h2 - inter
                    iou = inter / union if union > 0 else 0.0
                topo_sim = iou
                
                # Calculate Weighted Similarity Score
                sim = 0.40 * geom_sim + 0.25 * pos_sim + 0.15 * orient_sim + 0.10 * shape_sim + 0.10 * topo_sim
                
                # Block matches that have low similarity (e.g. < 30%)
                if sim < 0.30:
                    cost_matrix[i, j] = 9999.0
                else:
                    cost_matrix[i, j] = 1.0 - sim
                    
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_A = set()
        matched_B = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 5.0:  # Valid match
                o1 = A_type[r]
                o2 = B_type[c]
                matched_A.add(o1.id)
                matched_B.add(o2.id)
                
                sim = 1.0 - cost_matrix[r, c]
                
                # Check for physical shifts/changes
                c1_x, c1_y = o1.bbox[0] + o1.bbox[2]/2, o1.bbox[1] + o1.bbox[3]/2
                c2_x, c2_y = o2.bbox[0] + o2.bbox[2]/2, o2.bbox[1] + o2.bbox[3]/2
                dist = math.hypot(c2_x - c1_x, c2_y - c1_y)
                
                len1 = o1.geometry.get("length", o1.geometry.get("swing_length", o1.bbox[2]))
                len2 = o2.geometry.get("length", o2.geometry.get("swing_length", o2.bbox[2]))
                len_diff = abs(len2 - len1)
                
                is_changed = False
                desc = "No design revision detected"
                change_type = "UNCHANGED"
                
                if o1.type in ("Dimension", "Text Label"):
                    val1 = o1.geometry.get("text", "")
                    val2 = o2.geometry.get("text", "")
                    if val1 != val2:
                        is_changed = True
                        change_type = "MODIFIED"
                        desc = f"Text modified: updated from '{val1}' to '{val2}'"
                else:
                    # Ignore geometry changes below 15% and 15 pixels (anti-aliasing, rasterization, DPI differences)
                    if len_diff > max(15.0, len1 * 0.15):
                        is_changed = True
                        change_type = "MODIFIED"
                        change = "increased" if len2 > len1 else "decreased"
                        desc = f"Size {change}: length modified from {len1:.1f}px to {len2:.1f}px"
                
                # Check for significant physical translation (only if shift is > 15 pixels)
                if not is_changed and dist > 15.0:
                    is_changed = True
                    change_type = "MOVED"
                    desc = f"Shifted by {dist:.1f} pixels"
                
                if is_changed:
                    results.append((o1, o2, change_type, desc))
                else:
                    results.append((o1, o2, "UNCHANGED", "No design revision detected"))
                        
        # Unmatched A are Removed
        for o1 in A_type:
            if o1.id not in matched_A:
                results.append((o1, None, "REMOVED", f"{o1.type} removed from layout"))
                
        # Unmatched B are Added
        for o2 in B_type:
            if o2.id not in matched_B:
                results.append((None, o2, "ADDED", f"New {o2.type} added to layout"))

    # Cap at top 25 most significant changes to keep logs precise and clean (20 - 30 range)
    changed = [r for r in results if r[2] != "UNCHANGED"]
    unchanged = [r for r in results if r[2] == "UNCHANGED"]

    if len(changed) > 25:
        def change_priority(item):
            o1, o2, status, desc = item
            obj = o2 if o2 is not None else o1
            # Priority: Doors/Windows/Columns > Walls > others
            p = 10
            if obj.type in ("Door", "Window", "Column"):
                p = 100
            elif obj.type == "Wall":
                p = 50
            
            # Severity magnitude
            mag = 0.0
            if status == "MOVED":
                c1_x, c1_y = o1.bbox[0] + o1.bbox[2]/2, o1.bbox[1] + o1.bbox[3]/2
                c2_x, c2_y = o2.bbox[0] + o2.bbox[2]/2, o2.bbox[1] + o2.bbox[3]/2
                mag = math.hypot(c2_x - c1_x, c2_y - c1_y)
            elif status == "MODIFIED":
                len1 = o1.geometry.get("length", o1.geometry.get("swing_length", o1.bbox[2]))
                len2 = o2.geometry.get("length", o2.geometry.get("swing_length", o2.bbox[2]))
                mag = abs(len2 - len1)
            else:
                mag = max(obj.bbox[2], obj.bbox[3])
                
            return (p, mag)

        changed.sort(key=change_priority, reverse=True)
        top_changed = changed[:25]
        demoted = changed[25:]
        for o1, o2, status, desc in demoted:
            unchanged.append((o1, o2, "UNCHANGED", "No design revision detected"))
        changed = top_changed

    return changed + unchanged


def draw_cad_overlay(image: np.ndarray, matched_results: list) -> np.ndarray:
    """Draw bounding boxes only around modified objects in Green/Red/Yellow/Blue."""
    overlay = image.copy()
    for o1, o2, status, desc in matched_results:
        if status == "UNCHANGED":
            continue

        obj = o2 if o2 is not None else o1
        if obj is None:
            continue

        x, y, w_obj, h_obj = obj.bbox

        # Highlight color mapping (Green = Added, Red = Removed, Yellow = Modified, Blue = Shifted)
        if status == "ADDED":
            color = (0, 180, 0)       # Green
        elif status == "REMOVED":
            color = (0, 0, 255)       # Red
        elif status == "MODIFIED":
            color = (0, 220, 220)     # Yellow
        elif status == "MOVED":
            color = (255, 0, 0)       # Blue
        else:
            continue

        cv2.rectangle(overlay, (x, y), (x + w_obj, y + h_obj), color, 2)

        # Label drawing card
        label = f"{obj.id} ({status})"
        cv2.putText(overlay, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return overlay



