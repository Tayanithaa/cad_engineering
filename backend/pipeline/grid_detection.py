import cv2
import numpy as np
from scipy.signal import find_peaks
import config

def compute_projection_profiles(img):
    """
    Computes 1D projection profiles for vertical and horizontal lines.
    Returns:
        x_profile: sum along columns (for vertical bays/columns)
        y_profile: sum along rows (for horizontal floors/bands)
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Invert so that black lines/elements on white background are high-density peaks
    mean_val = np.mean(gray)
    if mean_val > 127:
        processed = 255 - gray
    else:
        processed = gray

    # Threshold to reduce noise
    _, binary = cv2.threshold(processed, 30, 255, cv2.THRESH_TOZERO)

    # Sum along axes
    # Vertical profile (x-coordinate): sum over height (axis 0)
    x_profile = np.sum(binary, axis=0).astype(np.float32)
    # Horizontal profile (y-coordinate): sum over width (axis 1)
    y_profile = np.sum(binary, axis=1).astype(np.float32)
    
    return x_profile, y_profile

def detect_facade_grid(img1, img2_aligned):
    """
    Detects repeating horizontal and vertical grid divisions on the aligned drawings.
    Uses projection profile analysis and peak finding.
    """
    # Combine or average profiles of both images to get a consensus grid
    x_prof1, y_prof1 = compute_projection_profiles(img1)
    x_prof2, y_prof2 = compute_projection_profiles(img2_aligned)
    
    # Pad to match sizes if slightly off, but since aligned they should be identical size
    min_w = min(len(x_prof1), len(x_prof2))
    min_h = min(len(y_prof1), len(y_prof2))
    
    x_profile = (x_prof1[:min_w] + x_prof2[:min_w]) / 2.0
    y_profile = (y_prof1[:min_h] + y_prof2[:min_h]) / 2.0
    
    # Smooth profiles with moving average to reduce noise before peak finding
    window_size = 15
    if len(x_profile) > window_size:
        x_smooth = np.convolve(x_profile, np.ones(window_size)/window_size, mode='same')
    else:
        x_smooth = x_profile
        
    if len(y_profile) > window_size:
        y_smooth = np.convolve(y_profile, np.ones(window_size)/window_size, mode='same')
    else:
        y_smooth = y_profile

    # Determine peak threshold/prominence based on profile range
    x_prom = (np.max(x_smooth) - np.min(x_smooth)) * config.PEAK_PROMINENCE_FACTOR
    y_prom = (np.max(y_smooth) - np.min(y_smooth)) * config.PEAK_PROMINENCE_FACTOR

    # Find peaks representing columns and floor lines
    # distance constraints: e.g. columns must be at least 40 pixels apart (adjust as necessary)
    x_peaks, _ = find_peaks(x_smooth, prominence=x_prom, distance=40)
    y_peaks, _ = find_peaks(y_smooth, prominence=y_prom, distance=40)
    
    # Convert peaks to sorted standard Python lists of integers (pixel coordinates)
    grid_cols = sorted(x_peaks.tolist())
    grid_rows = sorted(y_peaks.tolist())
    
    return {
        "x_peaks": grid_cols,
        "y_peaks": grid_rows
    }
