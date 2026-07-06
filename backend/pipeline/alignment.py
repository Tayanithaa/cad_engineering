import cv2
import numpy as np
import math
import config
import re

# We will import OCR extraction later in the function to avoid circular imports if any
# from backend.pipeline import ocr_extraction

def decompose_homography_scale(H):
    """
    Extracts the scale factor from the 3x3 homography matrix H.
    Specifically, we approximate the scaling factor from the top-left 2x2 matrix.
    """
    if H is None:
        return 1.0
    # scale_x = sqrt(h00^2 + h10^2)
    # scale_y = sqrt(h01^2 + h11^2)
    scale_x = math.sqrt(H[0, 0]**2 + H[1, 0]**2)
    scale_y = math.sqrt(H[0, 1]**2 + H[1, 1]**2)
    # Return average scale
    return (scale_x + scale_y) / 2.0

def parse_scale_text(text):
    """
    Parses scale strings like '1/8" = 1\'-0"', '3/16" = 1\'-0"', '1:100', '1 = 50'
    Returns a float representing the scale value (e.g. 1/8 / 12 = 0.0104)
    """
    text = text.lower().strip()
    # Match fractional inch scales like 1/8" = 1'-0" or 3/16 = 1-0
    match = re.search(r"(\d+)/(\d+)\s*(?:\"|inch|in)?\s*=\s*(\d+)\s*(?:\'|foot|ft)?\s*(?:-\s*(\d+))?", text)
    if match:
        num = float(match.group(1))
        denom = float(match.group(2))
        feet = float(match.group(3))
        inches = float(match.group(4)) if match.group(4) else 0.0
        
        fractional_inch = num / denom
        total_inches = (feet * 12.0) + inches
        if total_inches > 0:
            return fractional_inch / total_inches

    # Match ratio scales like 1:100 or 1/100 or scale 1 to 50
    match_ratio = re.search(r"1\s*(?::|/|to)\s*(\d+)", text)
    if match_ratio:
        val = float(match_ratio.group(1))
        if val > 0:
            return 1.0 / val

    return None

def find_scale_in_image(img, ocr_engine=None):
    """
    Scans the image (especially bottom section or title block) for scale annotations.
    """
    from backend.pipeline import ocr_extraction
    
    # Crop to the bottom 30% of the image where scale text usually is
    h, w = img.shape[:2]
    bottom_crop = img[int(0.7 * h):h, :]
    
    # Run OCR on bottom crop
    results = ocr_extraction.run_ocr_on_image(bottom_crop, ocr_engine=ocr_engine)
    
    for item in results:
        text = item.get("text", "")
        scale_val = parse_scale_text(text)
        if scale_val is not None:
            return scale_val, text
            
    # Try the full image if not found in bottom crop
    results_full = ocr_extraction.run_ocr_on_image(img, ocr_engine=ocr_engine)
    for item in results_full:
        text = item.get("text", "")
        scale_val = parse_scale_text(text)
        if scale_val is not None:
            return scale_val, text
            
    return None, None

def align_images(img1, img2, ocr_engine=None):
    """
    Aligns img2 to img1's coordinate system using SIFT keypoint detection and FLANN matching.
    Cross-checks scale with OCR if scale text is found.
    Returns:
        warped_img2: aligned version of img2
        alignment_metadata: dict containing homography, scale_ratio, confidence, and status logs.
    """
    # Convert to grayscale and denoise
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.fastNlMeansDenoising(gray1, h=3)
    gray2 = cv2.fastNlMeansDenoising(gray2, h=3)
    
    # SIFT feature extraction
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    alignment_confidence = "high"
    status_msg = "Alignment successful"
    sift_scale_ratio = 1.0
    H = None
    warped_img2 = img2.copy()
    
    if des1 is None or des2 is None or len(kp1) < config.SIFT_MIN_MATCHES or len(kp2) < config.SIFT_MIN_MATCHES:
        alignment_confidence = "low"
        status_msg = "Insufficient SIFT features found for reliable alignment."
        return warped_img2, {
            "scale_ratio": 1.0,
            "alignment_confidence": alignment_confidence,
            "status_message": status_msg,
            "homography": None,
            "matches_count": 0
        }
        
    # FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception as e:
        # Fallback to Brute-Force if FLANN fails
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < config.SIFT_RATIO_TEST * n.distance:
            good_matches.append(m)
            
    matches_count = len(good_matches)
    
    if matches_count < config.SIFT_MIN_MATCHES:
        alignment_confidence = "low"
        status_msg = f"Low alignment confidence: Only {matches_count} good matches found (min threshold is {config.SIFT_MIN_MATCHES})."
    else:
        # Compute Homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # dst is img2, src is img1. We want to warp img2 to match img1.
        # Find homography from img2 to img1
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            sift_scale_ratio = decompose_homography_scale(H)
            
            # Warp img2
            h1, w1 = img1.shape[:2]
            warped_img2 = cv2.warpPerspective(img2, H, (w1, h1))
        else:
            alignment_confidence = "low"
            status_msg = "Homography matrix computation failed."
            
    # OCR Scale Cross-check
    ocr_scale_ratio = None
    ocr_scale1_text = None
    ocr_scale2_text = None
    try:
        scale1, ocr_scale1_text = find_scale_in_image(img1, ocr_engine)
        scale2, ocr_scale2_text = find_scale_in_image(img2, ocr_engine)
        
        if scale1 and scale2:
            # We want to warp img2 (v2) to match img1 (v1).
            # The scaling applied to v2 to match v1 is scale1 / scale2
            ocr_scale_ratio = scale1 / scale2
            
            # Compare with SIFT ratio
            ratio_diff = abs(sift_scale_ratio - ocr_scale_ratio)
            if ratio_diff > config.SCALE_TOLERANCE:
                alignment_confidence = "low"
                status_msg += f" Scale mismatch: SIFT ratio {sift_scale_ratio:.3f} vs OCR ratio {ocr_scale_ratio:.3f}."
    except Exception as e:
        # Don't fail the alignment if OCR fails, just log it
        print(f"Warning in OCR scale cross-check: {e}")
        
    return warped_img2, {
        "scale_ratio": sift_scale_ratio,
        "ocr_scale_ratio": ocr_scale_ratio,
        "ocr_scale1_text": ocr_scale1_text,
        "ocr_scale2_text": ocr_scale2_text,
        "alignment_confidence": alignment_confidence,
        "status_message": status_msg,
        "homography": H.tolist() if H is not None else None,
        "matches_count": matches_count
    }
