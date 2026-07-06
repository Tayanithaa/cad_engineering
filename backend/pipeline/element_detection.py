import cv2
import numpy as np
import config

def snap_to_grid(bbox, grid):
    """
    Snaps a bounding box [x, y, w, h] to the closest columns (x_peaks) and rows (y_peaks).
    Returns (col_idx, row_idx).
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    
    x_peaks = grid.get("x_peaks", [])
    y_peaks = grid.get("y_peaks", [])
    
    col_idx = -1
    row_idx = -1
    
    if x_peaks:
        # Find index of closest column peak
        col_idx = int(np.argmin([abs(cx - p) for p in x_peaks]))
    if y_peaks:
        # Find index of closest row peak
        row_idx = int(np.argmin([abs(cy - p) for p in y_peaks]))
        
    return col_idx, row_idx

def detect_windows_and_doors(img, grid):
    """
    Detects windows and doors using cv2.findContours + aspect ratio filtering.
    Doors must be in the bottom 20% of the building height band.
    """
    elements = []
    h_img, w_img = img.shape[:2]
    
    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to find lines and boxes
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological closing to join broken rect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find height band boundaries from grid peaks if available
    y_peaks = grid.get("y_peaks", [])
    if y_peaks:
        min_y_grid = min(y_peaks)
        max_y_grid = max(y_peaks)
        building_height = max_y_grid - min_y_grid
    else:
        min_y_grid = 0
        max_y_grid = h_img
        building_height = h_img

    door_threshold_y = max_y_grid - 0.25 * building_height

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_CONTOUR_AREA or area > (w_img * h_img * 0.1):
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Windows are usually aspect ratio ~0.5 to ~2.0, doors are taller (AR ~0.3 to ~0.8)
        # Filters for doors vs windows
        col_idx, row_idx = snap_to_grid((x, y, w, h), grid)
        
        if y + h >= door_threshold_y and 1.2 < (float(h) / w) < 3.0:
            # Taller than wide, located at the bottom portion of building
            elements.append({
                "type": "door",
                "bbox": [x, y, w, h],
                "grid_pos": [col_idx, row_idx]
            })
        elif 0.4 <= aspect_ratio <= 2.5 and w > 20 and h > 20:
            elements.append({
                "type": "window",
                "bbox": [x, y, w, h],
                "grid_pos": [col_idx, row_idx]
            })
            
    return elements

def detect_pillars(img, grid):
    """
    Detects vertical pillars/columns using Hough Line Transform.
    Look for vertical parallel lines at regular spacing.
    """
    elements = []
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Run HoughLinesP to find lines
    # minLineLength should be reasonably long to represent columns
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return elements
        
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if line is nearly vertical (slope angle near 90 deg)
        if abs(x2 - x1) < 5:  # Tolerance in pixels
            vertical_lines.append((x1, min(y1, y2), max(y1, y2)))
            
    # Group vertical lines that are close to form pillars (columns have a width, so two vertical lines close together)
    vertical_lines.sort(key=lambda item: item[0])
    
    i = 0
    while i < len(vertical_lines) - 1:
        x1, y1_start, y1_end = vertical_lines[i]
        j = i + 1
        found_pillar = False
        while j < len(vertical_lines):
            x2, y2_start, y2_end = vertical_lines[j]
            # If line is within, say, 15 to 60 pixels, it could be a column
            width = x2 - x1
            if 15 <= width <= 60:
                # Check overlap in Y
                y_start = max(y1_start, y2_start)
                y_end = min(y1_end, y2_end)
                if (y_end - y_start) > 80:  # height of overlap
                    # Form a pillar bounding box
                    bbox = [x1, y_start, width, y_end - y_start]
                    col_idx, row_idx = snap_to_grid(bbox, grid)
                    elements.append({
                        "type": "pillar",
                        "bbox": bbox,
                        "grid_pos": [col_idx, row_idx]
                    })
                    found_pillar = True
                    i = j  # Skip forward
                    break
            elif width > 60:
                break
            j += 1
        if not found_pillar:
            i += 1
            
    return elements

def detect_roof(img):
    """
    Detects the roof outline by identifying the topmost silhouette.
    Simplifies contour to segments using approxPolyDP.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to binary
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
        
    # Get largest outer contour (assumed building outline)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    building_cnt = contours[0]
    
    # Simplify outline
    epsilon = 0.01 * cv2.arcLength(building_cnt, True)
    approx = cv2.approxPolyDP(building_cnt, epsilon, True)
    
    # Roof lines are typically at the top part of the polygon
    # Find bounding box of the building
    bx, by, bw, bh = cv2.boundingRect(building_cnt)
    roof_threshold_y = by + 0.3 * bh  # topmost 30% of building height
    
    roof_segments = []
    points = [pt[0] for pt in approx]
    
    for i in range(len(points)):
        pt1 = points[i]
        pt2 = points[(i + 1) % len(points)]
        # If both points are in the top region, classify as roof segment
        if pt1[1] <= roof_threshold_y and pt2[1] <= roof_threshold_y:
            # Store segment line
            roof_segments.append({
                "type": "roof",
                "bbox": [int(min(pt1[0], pt2[0])), int(min(pt1[1], pt2[1])), int(abs(pt2[0] - pt1[0])), int(abs(pt2[1] - pt1[1]))],
                "points": [[int(pt1[0]), int(pt1[1])], [int(pt2[0]), int(pt2[1])]],
                "grid_pos": [-1, -1]
            })
            
    return roof_segments

def repeated_element_confirmation(img, confirmed_elements, grid):
    """
    Uses Template Matching to find elements missed by contour detection.
    """
    if not confirmed_elements:
        return confirmed_elements
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_elements = list(confirmed_elements)
    
    # Pick a good window template
    windows = [e for e in confirmed_elements if e["type"] == "window"]
    if not windows:
        return confirmed_elements
        
    # Pick the one with the median area as template
    windows.sort(key=lambda e: e["bbox"][2] * e["bbox"][3])
    template_elem = windows[len(windows) // 2]
    tx, ty, tw, th = template_elem["bbox"]
    
    if tw < 10 or th < 10:
        return confirmed_elements
        
    template = gray[ty:ty+th, tx:tx+tw]
    
    # Match template over full gray image
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    loc = np.where(res >= threshold)
    
    # Non-maximum suppression to filter matches close to each other
    detected_boxes = []
    for pt in zip(*loc[::-1]): # x, y
        detected_boxes.append([pt[0], pt[1], tw, th])
        
    # Filter out boxes that overlap significantly with already confirmed elements
    for box in detected_boxes:
        bx, by, bw, bh = box
        # Check overlap
        is_overlapping = False
        for ext in all_elements:
            ex, ey, ew, eh = ext["bbox"]
            # Intersection area
            ix = max(bx, ex)
            iy = max(by, ey)
            iw = min(bx + bw, ex + ew) - ix
            ih = min(by + bh, ey + eh) - iy
            if iw > 0 and ih > 0:
                overlap_area = iw * ih
                min_area = min(bw * bh, ew * eh)
                if overlap_area / min_area > 0.4:
                    is_overlapping = True
                    break
        if not is_overlapping:
            col_idx, row_idx = snap_to_grid(box, grid)
            all_elements.append({
                "type": "window",
                "bbox": box,
                "grid_pos": [col_idx, row_idx]
            })
            
    return all_elements

def detect_all_elements(img, grid):
    """
    Executes element detection pipeline.
    """
    windows_doors = detect_windows_and_doors(img, grid)
    pillars = detect_pillars(img, grid)
    roof = detect_roof(img)
    
    combined = windows_doors + pillars + roof
    
    # Run template matching to find missed window elements
    confirmed = repeated_element_confirmation(img, combined, grid)
    return confirmed
