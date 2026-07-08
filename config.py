import os

# Manually load .env file if it exists
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in choices else default


DPI = _env_int("CAD_COMPARE_DPI", 300)
MIN_CONTOUR_AREA = _env_int("CAD_COMPARE_MIN_CONTOUR_AREA", 120)
ORB_MIN_GOOD_MATCHES = _env_int("CAD_COMPARE_ORB_MIN_GOOD_MATCHES", 15)
ORB_MIN_INLIER_RATIO = _env_float("CAD_COMPARE_ORB_MIN_INLIER_RATIO", 0.5)
SSIM_CHANGE_THRESHOLD = _env_float("CAD_COMPARE_SSIM_CHANGE_THRESHOLD", 0.86)
OCR_CONFIDENCE_THRESHOLD = _env_float("CAD_COMPARE_OCR_CONFIDENCE_THRESHOLD", 0.6)
OCR_ENGINE = _env_choice("CAD_COMPARE_OCR_ENGINE", "paddleocr", {"paddleocr", "easyocr"})
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = _env_float("GROQ_TEMPERATURE", 0.2)

MORPH_KERNEL_SIZE = _env_int("CAD_COMPARE_MORPH_KERNEL_SIZE", 3)
OCR_UPSCALE_FACTOR = _env_int("CAD_COMPARE_OCR_UPSCALE_FACTOR", 3)
MAX_AI_RETRIES = _env_int("CAD_COMPARE_MAX_AI_RETRIES", 3)
AI_RETRY_BASE_SECONDS = _env_float("CAD_COMPARE_AI_RETRY_BASE_SECONDS", 1.0)

WINDOW_MAX_AREA_RATIO = _env_float("CAD_COMPARE_WINDOW_MAX_AREA_RATIO", 0.03)
DOOR_BOTTOM_BAND_RATIO = _env_float("CAD_COMPARE_DOOR_BOTTOM_BAND_RATIO", 0.20)
ROOF_TOP_BAND_RATIO = _env_float("CAD_COMPARE_ROOF_TOP_BAND_RATIO", 0.25)
