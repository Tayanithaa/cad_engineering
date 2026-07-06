import os

# DPI for rendering PDFs to images
DPI = 300

# Minimum contour area for border detection and elements
MIN_CONTOUR_AREA = 100

# SIFT matching thresholds
SIFT_MIN_MATCHES = 15
SIFT_RATIO_TEST = 0.7  # Lowe's ratio test threshold

# SSIM structural difference threshold
SSIM_THRESHOLD = 0.90

# OCR configurations
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr")  # "paddleocr" or "easyocr"
OCR_CONFIDENCE_THRESHOLD = 0.60

# Grid detection sensitivity
PEAK_PROMINENCE_FACTOR = 0.15

# Alignment scale difference tolerance (OCR vs SIFT ratio)
SCALE_TOLERANCE = 0.05

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Recommended Groq text models: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")
