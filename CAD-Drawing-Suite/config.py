"""
Central configuration for the CAD Drawing Revision Comparator.
All tunable thresholds for the classical CV / OCR pipeline live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
PDF_RENDER_DPI = 150

# ---------------------------------------------------------------------------
# Alignment (SIFT + FLANN + RANSAC)
# ---------------------------------------------------------------------------
SIFT_N_FEATURES = 0  # 0 = unlimited
LOWE_RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 15  # below this -> "low-confidence alignment" flag
RANSAC_REPROJ_THRESHOLD = 5.0

# ---------------------------------------------------------------------------
# Difference detection
# ---------------------------------------------------------------------------
SSIM_WINDOW_SIZE = 7
MIN_CONTOUR_AREA = 60  # px^2, filters tiny noise contours
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITERATIONS = 0
MORPH_CLOSE_ITERATIONS = 1
DIFF_BINARY_THRESHOLD = 15  # 0-255, applied to (1 - SSIM) map after normalization

# Vector comparison tolerances (line-level matching)
LINE_DISTANCE_TOLERANCE = 3.0    # maximum pixel distance for midpoints
LINE_ANGLE_TOLERANCE = 0.05      # radians (approx 3 degrees)
LINE_SHIFT_TOLERANCE = 15.0      # maximum shift distance in pixels
LINE_LENGTH_TOLERANCE = 5.0      # length difference in pixels for modified
MIN_LINE_LENGTH = 8.0            # ignore lines shorter than this (noise)

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_UPSCALE_FACTORS = [2, 3, 4]  # tried in order until confidence improves; we pick best
OCR_ROTATIONS = [0, 90, 180, 270]
OCR_MIN_CONFIDENCE = 0.35  # below this, treated as "no text found"
OCR_CROP_PADDING = 6  # px padding added around each bbox before crop

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
NUMERIC_TOLERANCE = 1e-3  # inches/mm equivalence tolerance after unit normalization

# ---------------------------------------------------------------------------
# Groq AI summary (the ONLY AI call in the whole pipeline)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = 40
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
