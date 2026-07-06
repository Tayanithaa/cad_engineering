import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import config

def get_ssim_for_crops(crop1, crop2):
    """
    Computes structural similarity index (SSIM) between two image crops.
    Returns:
        ssim_val: float value between -1.0 and 1.0 (1.0 meaning identical)
    """
    try:
        # Convert to grayscale
        if len(crop1.shape) == 3:
            gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = crop1.copy()
            
        if len(crop2.shape) == 3:
            gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = crop2.copy()
            
        # Ensure identical sizes
        h1, w1 = gray1.shape[:2]
        h2, w2 = gray2.shape[:2]
        if (h1 != h2) or (w1 != w2):
            # Resize crop2 to match crop1
            gray2 = cv2.resize(gray2, (w1, h1))
            
        # Calculate SSIM
        # Set win_size based on crop size
        min_dim = min(h1, w1)
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size = max(3, win_size - 1)
            
        if min_dim < 3:
            # Too small to compute SSIM, return pixel-level equality
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            return 1.0 - (mean_diff / 255.0)

        score, _ = ssim(gray1, gray2, win_size=win_size, full=True)
        return float(score)
    except Exception as e:
        print(f"Error in SSIM calculation: {e}")
        return 0.0

def compute_absolute_pixel_diff(crop1, crop2):
    """
    Computes absolute pixel difference as a secondary comparison signal.
    Returns:
        mean_diff: average difference value per pixel (0-255)
    """
    try:
        if len(crop1.shape) == 3:
            gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = crop1.copy()
            
        if len(crop2.shape) == 3:
            gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = crop2.copy()
            
        h1, w1 = gray1.shape[:2]
        h2, w2 = gray2.shape[:2]
        if (h1 != h2) or (w1 != w2):
            gray2 = cv2.resize(gray2, (w1, h1))
            
        abs_diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(abs_diff))
    except Exception as e:
        print(f"Error in pixel diff: {e}")
        return 255.0

def generate_ssim_heatmap(img1, img2_aligned):
    """
    Generates a full-image SSIM heatmap representation.
    Returns a BGR heatmap image where color indicates similarity (blue/green = high, red = low).
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    
    # Ensure matching dimensions
    h1, w1 = gray1.shape[:2]
    h2, w2 = gray2.shape[:2]
    if (h1 != h2) or (w1 != w2):
        gray2 = cv2.resize(gray2, (w1, h1))
        
    # Full image SSIM
    score, diff_img = ssim(gray1, gray2, full=True)
    
    # diff_img is float [0, 1]. Invert: 0 is identical (white), 1 is complete diff (black)
    diff_u8 = (diff_img * 255).astype(np.uint8)
    
    # Invert so 0 = matching (high similarity), 255 = completely different
    inverted_diff = 255 - diff_u8
    
    # Apply colormap (JET or VIRIDIS)
    heatmap = cv2.applyColorMap(inverted_diff, cv2.COLORMAP_JET)
    
    # Overlay heatmap with original image for context
    overlay = cv2.addWeighted(img1, 0.6, heatmap, 0.4, 0)
    return overlay

def compute_pixel_diff_mask(img1, img2_aligned):
    """
    Computes morphological-cleaned absolute pixel difference mask for full image.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    
    # Ensure same size
    h1, w1 = gray1.shape[:2]
    h2, w2 = gray2.shape[:2]
    if (h1 != h2) or (w1 != w2):
        gray2 = cv2.resize(gray2, (w1, h1))
        
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological open/close to remove anti-aliasing noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
