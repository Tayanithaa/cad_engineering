import cv2
import numpy as np
from backend.pipeline import diff
from backend.pipeline import ocr_extraction
from backend.pipeline import compare
import config

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)
    intersection = iw * ih
    
    union = (w1 * h1) + (box2[2] * box2[3]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def get_crop(img, bbox):
    """
    Helper to safely crop bounding box from image.
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    # Clamp coordinates to image boundaries
    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2]

def merge_pipeline_results(img1, img2_aligned, elements_v1, elements_v2, grid, ocr_engine=None):
    """
    Merges structural elements and OCR text detections into a single list of change records.
    """
    change_records = []
    
    unmatched_v2 = list(elements_v2)
    region_counter = 1
    
    # 1. Process all elements in v1 and match them with v2
    for el1 in elements_v1:
        type1 = el1["type"]
        box1 = el1["bbox"]
        grid1 = el1["grid_pos"]
        
        # Try to find a match in unmatched_v2
        match_idx = -1
        best_iou = 0.0
        
        for idx, el2 in enumerate(unmatched_v2):
            if el2["type"] != type1:
                continue
            
            # Check grid position match
            if grid1 != [-1, -1] and el2["grid_pos"] == grid1:
                match_idx = idx
                break
                
            # Fallback to IoU overlap
            iou = calculate_iou(box1, el2["bbox"])
            if iou > 0.25 and iou > best_iou:
                best_iou = iou
                match_idx = idx
                
        # Crops for comparison
        crop1 = get_crop(img1, box1)
        
        if match_idx != -1:
            # Matched!
            el2 = unmatched_v2.pop(match_idx)
            box2 = el2["bbox"]
            crop2 = get_crop(img2_aligned, box2)
            
            # Compute structural similarities
            ssim_score = diff.get_ssim_for_crops(crop1, crop2)
            pixel_diff = diff.compute_absolute_pixel_diff(crop1, crop2)
            
            # Perform OCR on these regions to see if there is any label/text changes
            ocr1 = ocr_extraction.run_ocr_with_rotations(crop1, ocr_engine)
            ocr2 = ocr_extraction.run_ocr_with_rotations(crop2, ocr_engine)
            
            pv1, u1 = ocr_extraction.parse_dimension_string(ocr1["text"])
            pv2, u2 = ocr_extraction.parse_dimension_string(ocr2["text"])
            
            ocr1["parsed_val"] = pv1
            ocr1["unit"] = u1
            ocr2["parsed_val"] = pv2
            ocr2["unit"] = u2
            
            # Decide change type
            txt_change, txt_low_conf = compare.compare_labels_and_values(ocr1, ocr2)
            
            if ssim_score >= config.SSIM_THRESHOLD and txt_change == "Unchanged":
                change_type = "Unchanged"
            else:
                change_type = "Modified"
                
            change_records.append({
                "region_id": f"REG_{region_counter:03d}",
                "element_type": type1,
                "bbox_v1": box1,
                "bbox_v2": box2,
                "grid_pos": grid1,
                "change_type": change_type,
                "v1_value": ocr1["text"] if ocr1["text"] else f"Pos: {box1}",
                "v2_value": ocr2["text"] if ocr2["text"] else f"Pos: {box2}",
                "ssim_score": ssim_score,
                "pixel_diff": pixel_diff,
                "ocr_confidence": min(ocr1["confidence"], ocr2["confidence"]),
                "low_confidence": txt_low_conf or (ssim_score < config.SSIM_THRESHOLD and ssim_score > 0.7)
            })
        else:
            # Removed in v2
            ocr1 = ocr_extraction.run_ocr_with_rotations(crop1, ocr_engine)
            change_records.append({
                "region_id": f"REG_{region_counter:03d}",
                "element_type": type1,
                "bbox_v1": box1,
                "bbox_v2": None,
                "grid_pos": grid1,
                "change_type": "Removed",
                "v1_value": ocr1["text"] if ocr1["text"] else f"Pos: {box1}",
                "v2_value": "N/A",
                "ssim_score": 0.0,
                "pixel_diff": 255.0,
                "ocr_confidence": ocr1["confidence"],
                "low_confidence": ocr1["confidence"] < config.OCR_CONFIDENCE_THRESHOLD
            })
        region_counter += 1
        
    # 2. Process all remaining unmatched v2 elements as Added
    for el2 in unmatched_v2:
        box2 = el2["bbox"]
        crop2 = get_crop(img2_aligned, box2)
        ocr2 = ocr_extraction.run_ocr_with_rotations(crop2, ocr_engine)
        
        change_records.append({
            "region_id": f"REG_{region_counter:03d}",
            "element_type": el2["type"],
            "bbox_v1": None,
            "bbox_v2": box2,
            "grid_pos": el2["grid_pos"],
            "change_type": "Added",
            "v1_value": "N/A",
            "v2_value": ocr2["text"] if ocr2["text"] else f"Pos: {box2}",
            "ssim_score": 0.0,
            "pixel_diff": 255.0,
            "ocr_confidence": ocr2["confidence"],
            "low_confidence": ocr2["confidence"] < config.OCR_CONFIDENCE_THRESHOLD
        })
        region_counter += 1

    # 3. OCR Text Block Detections (to compare label-only annotations that are not windows/doors/pillars/roof)
    # Run OCR on full images, filter out annotations that overlap with already processed elements
    ocr_v1_full = ocr_extraction.run_ocr_on_image(img1, ocr_engine)
    ocr_v2_full = ocr_extraction.run_ocr_on_image(img2_aligned, ocr_engine)
    
    unmatched_ocr_v2 = list(ocr_v2_full)
    
    for item1 in ocr_v1_full:
        box1 = item1["bbox"]
        text1 = item1["text"]
        conf1 = item1["confidence"]
        
        # Skip if overlaps with any structural element we already matched
        is_structural = False
        for rec in change_records:
            if rec["bbox_v1"] and calculate_iou(box1, rec["bbox_v1"]) > 0.3:
                is_structural = True
                break
        if is_structural:
            continue
            
        # Try to find a match in unmatched_ocr_v2
        match_idx = -1
        best_iou = 0.0
        
        for idx, item2 in enumerate(unmatched_ocr_v2):
            iou = calculate_iou(box1, item2["bbox"])
            if iou > 0.3 or (abs(box1[0] - item2["bbox"][0]) < 20 and abs(box1[1] - item2["bbox"][1]) < 20):
                match_idx = idx
                break
                
        if match_idx != -1:
            item2 = unmatched_ocr_v2.pop(match_idx)
            text2 = item2["text"]
            conf2 = item2["confidence"]
            
            # Compare
            v1_info = {"text": text1, "confidence": conf1}
            v2_info = {"text": text2, "confidence": conf2}
            
            change_type, low_conf = compare.compare_labels_and_values(v1_info, v2_info)
            
            change_records.append({
                "region_id": f"REG_{region_counter:03d}",
                "element_type": "text_label",
                "bbox_v1": box1,
                "bbox_v2": item2["bbox"],
                "grid_pos": [-1, -1],
                "change_type": change_type,
                "v1_value": text1,
                "v2_value": text2,
                "ssim_score": 1.0 if change_type == "Unchanged" else 0.5,
                "pixel_diff": 0.0 if change_type == "Unchanged" else 50.0,
                "ocr_confidence": min(conf1, conf2),
                "low_confidence": low_conf
            })
        else:
            # Removed label
            change_records.append({
                "region_id": f"REG_{region_counter:03d}",
                "element_type": "text_label",
                "bbox_v1": box1,
                "bbox_v2": None,
                "grid_pos": [-1, -1],
                "change_type": "Removed",
                "v1_value": text1,
                "v2_value": "N/A",
                "ssim_score": 0.0,
                "pixel_diff": 255.0,
                "ocr_confidence": conf1,
                "low_confidence": conf1 < config.OCR_CONFIDENCE_THRESHOLD
            })
        region_counter += 1
        
    for item2 in unmatched_ocr_v2:
        # Skip if overlaps with any structural element
        is_structural = False
        for rec in change_records:
            if rec["bbox_v2"] and calculate_iou(item2["bbox"], rec["bbox_v2"]) > 0.3:
                is_structural = True
                break
        if is_structural:
            continue
            
        change_records.append({
            "region_id": f"REG_{region_counter:03d}",
            "element_type": "text_label",
            "bbox_v1": None,
            "bbox_v2": item2["bbox"],
            "grid_pos": [-1, -1],
            "change_type": "Added",
            "v1_value": "N/A",
            "v2_value": item2["text"],
            "ssim_score": 0.0,
            "pixel_diff": 255.0,
            "ocr_confidence": item2["confidence"],
            "low_confidence": item2["confidence"] < config.OCR_CONFIDENCE_THRESHOLD
        })
        region_counter += 1
        
    return change_records
