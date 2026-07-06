import cv2
import numpy as np
import config

def detect_and_crop_border(img):
    """
    Detects the outer drawing border rectangle using contour detection and crops the image to it.
    If no clear border is found, crops to the bounding box of non-white pixels or returns original.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to binary (assuming dark drawings on light background or vice versa)
        # Check if background is mostly light or dark
        mean_val = np.mean(gray)
        if mean_val > 127:
            # Light background, invert
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img, (0, 0, img.shape[1], img.shape[0])
            
        # Filter for large rectangular contours
        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h
        
        candidate_rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            # We want contours that are large (e.g. > 15% of the total page area)
            # but not necessarily the full image boundaries (e.g. < 99% of page area)
            if (area > 0.15 * img_area) and (w < 0.99 * img_w or h < 0.99 * img_h):
                # Calculate extent / rectangularity
                rect_area = w * h
                extent = float(area) / rect_area
                # Outer borders are rectangular
                if extent > 0.8:
                    candidate_rects.append((area, (x, y, w, h)))
                    
        if candidate_rects:
            # Sort by area descending and pick the largest valid rectangle
            candidate_rects.sort(key=lambda x: x[0], reverse=True)
            best_area, bbox = candidate_rects[0]
            x, y, w, h = bbox
            # Add some padding or return the crop
            crop = img[y:y+h, x:x+w]
            return crop, bbox
            
        # Fallback: Crop to bounding box of all non-white pixels
        # Let's find coords where gray is not near 255 (background)
        # assuming light background
        if mean_val > 127:
            non_bg_coords = np.argwhere(gray < 250)
        else:
            non_bg_coords = np.argwhere(gray > 10)
            
        if non_bg_coords.size > 0:
            y_min, x_min = non_bg_coords.min(axis=0)
            y_max, x_max = non_bg_coords.max(axis=0)
            # Ensure boundaries are correct
            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)
            crop = img[y_min:y_min+h, x_min:x_min+w]
            return crop, (int(x_min), int(y_min), int(w), int(h))
            
        return img, (0, 0, img_w, img_h)
        
    except Exception as e:
        # Fallback to returning original image if any error occurs
        print(f"Warning in border detection: {e}. Returning original.")
        return img, (0, 0, img.shape[1], img.shape[0])
