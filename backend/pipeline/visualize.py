import cv2
import numpy as np
import base64
from backend.pipeline import merge

def encode_image_base64(img):
    """
    Encodes an OpenCV BGR image to a base64 PNG string.
    """
    if img is None or img.size == 0:
        return ""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def draw_legend(img):
    """
    Draws a legend on the image for the bounding boxes.
    """
    h, w = img.shape[:2]
    # Draw legend background in top right
    start_x = w - 180
    start_y = 20
    
    cv2.rectangle(img, (start_x, start_y), (start_x + 160, start_y + 90), (240, 240, 240), -1)
    cv2.rectangle(img, (start_x, start_y), (start_x + 160, start_y + 90), (180, 180, 180), 1)
    
    # Legend items: (Text, Color BGR)
    legend_items = [
        ("Modified", (0, 0, 255)),  # Red
        ("Added", (255, 0, 0)),     # Blue
        ("Removed", (0, 255, 0))    # Green
    ]
    
    for idx, (label, color) in enumerate(legend_items):
        y_pos = start_y + 25 + idx * 25
        # Draw small color square
        cv2.rectangle(img, (start_x + 10, y_pos - 10), (start_x + 25, y_pos + 5), color, -1)
        # Draw text
        cv2.putText(img, label, (start_x + 35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
    return img

def generate_visualizations(img1, img2_aligned, change_records):
    """
    Generates annotated overlays and crop thumbnails.
    Modifies v2 (or copies) to overlay colored boxes.
    Returns:
        annotated_v2: BGR image with overlays
        records_with_thumbnails: change_records with base64 encoded crops
    """
    annotated_v2 = img2_aligned.copy()
    annotated_v1 = img1.copy()
    
    records_with_thumbnails = []
    
    for rec in change_records:
        ch_type = rec["change_type"]
        bbox_v1 = rec["bbox_v1"]
        bbox_v2 = rec["bbox_v2"]
        elem_type = rec["element_type"]
        
        # Determine color
        # red = modified, blue = added, green = removed
        if ch_type == "Modified":
            color = (0, 0, 255)  # Red
        elif ch_type == "Added":
            color = (255, 0, 0)  # Blue
        elif ch_type == "Removed":
            color = (0, 255, 0)  # Green
        else:
            color = (128, 128, 128)  # Gray for unchanged
            
        # Extract crops for thumbnails if changed
        crop_v1_b64 = ""
        crop_v2_b64 = ""
        
        if ch_type != "Unchanged":
            if bbox_v1:
                c1 = merge.get_crop(img1, bbox_v1)
                crop_v1_b64 = encode_image_base64(c1)
            if bbox_v2:
                c2 = merge.get_crop(img2_aligned, bbox_v2)
                crop_v2_b64 = encode_image_base64(c2)
                
        # Draw bounding boxes
        if ch_type != "Unchanged":
            if bbox_v2:
                x, y, w, h = bbox_v2
                cv2.rectangle(annotated_v2, (x, y), (x + w, y + h), color, 2)
                # Put Label ID
                cv2.putText(annotated_v2, rec["region_id"], (x, max(15, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            elif bbox_v1:
                # Removed element - draw green on v1 image and on v2 as a dashed/marker box if possible,
                # or draw green box on v1. Let's draw it on annotated_v2 as well to show where it was removed.
                x, y, w, h = bbox_v1
                cv2.rectangle(annotated_v2, (x, y), (x + w, y + h), color, 2)
                cv2.line(annotated_v2, (x, y), (x + w, y + h), color, 1)
                cv2.line(annotated_v2, (x, y + h), (x + w, y), color, 1)
                cv2.putText(annotated_v2, rec["region_id"], (x, max(15, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
        rec_copy = rec.copy()
        rec_copy["crop_v1"] = crop_v1_b64
        rec_copy["crop_v2"] = crop_v2_b64
        records_with_thumbnails.append(rec_copy)
        
    # Draw legend on annotated v2
    annotated_v2 = draw_legend(annotated_v2)
    
    return annotated_v2, records_with_thumbnails
