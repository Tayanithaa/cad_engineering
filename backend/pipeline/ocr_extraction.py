import cv2
import numpy as np
import re
import config
from PIL import Image

# Global OCR instances to avoid reloading
_paddle_ocr = None
_easy_ocr_reader = None

def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            # Disable logging/system info output to keep logs clean
            _paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}. Falling back to EasyOCR if available.")
    return _paddle_ocr

def get_easy_ocr():
    global _easy_ocr_reader
    if _easy_ocr_reader is None:
        try:
            import easyocr
            _easy_ocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
    return _easy_ocr_reader

def run_ocr_on_image(img, ocr_engine=None):
    """
    Runs the selected OCR engine on a single image and returns a list of dicts:
    [{"text": text, "confidence": conf, "bbox": [x, y, w, h]}]
    """
    engine = ocr_engine or config.OCR_ENGINE
    results = []
    
    if engine == "paddleocr":
        ocr_inst = get_paddle_ocr()
        if ocr_inst is not None:
            try:
                # PaddleOCR expects BGR image or file path
                ocr_res = ocr_inst.ocr(img, cls=False)
                if ocr_res and ocr_res[0]:
                    for line in ocr_res[0]:
                        box = line[0]  # List of 4 points [[x,y], [x,y], [x,y], [x,y]]
                        text, conf = line[1]
                        
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x, y = int(min(xs)), int(min(ys))
                        w, h = int(max(xs) - x), int(max(ys) - y)
                        
                        results.append({
                            "text": text,
                            "confidence": float(conf),
                            "bbox": [x, y, w, h]
                        })
                    return results
            except Exception as e:
                print(f"PaddleOCR invocation failed: {e}. Trying EasyOCR fallback.")
        
        # Fall back to EasyOCR if PaddleOCR fails or is not installed
        engine = "easyocr"

    if engine == "easyocr":
        reader = get_easy_ocr()
        if reader is not None:
            try:
                # EasyOCR expects numpy array or PIL Image
                ocr_res = reader.readtext(img)
                for bbox, text, conf in ocr_res:
                    # bbox: [[x,y], [x,y], [x,y], [x,y]]
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    
                    results.append({
                        "text": text,
                        "confidence": float(conf),
                        "bbox": [x, y, w, h]
                    })
                return results
            except Exception as e:
                print(f"EasyOCR invocation failed: {e}")
                
    return results

def run_ocr_with_rotations(crop, ocr_engine=None):
    """
    Upscales the crop 2-4x, runs OCR at 0, 90, 180, and 270 degree rotations,
    and returns the result with the highest confidence.
    """
    if crop is None or crop.size == 0:
        return {"text": "", "confidence": 0.0}
        
    # Upscale crop 3x using cubic interpolation
    h, w = crop.shape[:2]
    upscaled = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    
    rotations = {
        0: upscaled,
        90: cv2.rotate(upscaled, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(upscaled, cv2.ROTATE_180),
        270: cv2.rotate(upscaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    
    best_text = ""
    best_conf = -1.0
    
    for angle, rotated_img in rotations.items():
        ocr_res = run_ocr_on_image(rotated_img, ocr_engine)
        if ocr_res:
            # Aggregate confidence
            mean_conf = np.mean([item["confidence"] for item in ocr_res])
            combined_text = " ".join([item["text"] for item in ocr_res])
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_text = combined_text
                
    # If no text found
    if best_conf < 0:
        best_conf = 0.0
        
    return {
        "text": best_text,
        "confidence": best_conf
    }

def parse_dimension_string(text):
    """
    Regex parsing of dimension strings (feet/inches or metric) into normalized value and unit.
    Returns:
        (parsed_value, unit) or (None, None)
    """
    text = text.strip().lower()
    
    # 1. Match imperial: e.g. 5'-6", 10', 8 1/2", 10 - 2 3/4"
    # Try to extract numbers
    # feet part: (\d+)' or (\d+)\s*(?:feet|ft)
    # inches part: (\d+)(?:\s*(\d+)/(\d+))?\" or (\d+)\s*(?:inches|in)
    
    imperial_pattern = r"(?:(\d+)\s*(?:\'|ft|foot|feet))?\s*(?:-?\s*(\d+)?\s*(?:(\d+)/(\d+))?\s*(?:\"|in|inch|inches))?"
    
    # Check if it matches metric first
    metric_pattern = r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)"
    metric_match = re.search(metric_pattern, text)
    if metric_match:
        val = float(metric_match.group(1))
        unit = metric_match.group(2)
        return val, unit
        
    # Imperial check
    if "'" in text or '"' in text or "ft" in text or "in" in text:
        match = re.search(imperial_pattern, text)
        if match and (match.group(1) or match.group(2) or match.group(3)):
            feet = float(match.group(1)) if match.group(1) else 0.0
            inches = float(match.group(2)) if match.group(2) else 0.0
            
            # Fraction of an inch
            fraction = 0.0
            if match.group(3) and match.group(4):
                num = float(match.group(3))
                denom = float(match.group(4))
                if denom > 0:
                    fraction = num / denom
                    
            total_inches = (feet * 12.0) + inches + fraction
            return total_inches, "inches"
            
    # Try simple numeric parsing
    simple_num = re.search(r"(\d+(?:\.\d+)?)", text)
    if simple_num:
        return float(simple_num.group(1)), "px"  # Default unit is px when none specified
        
    return None, None
