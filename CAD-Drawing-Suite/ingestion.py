"""
Stage 1: Upload & Ingestion.

Accepts PDF/JPG/PNG uploads, validates them, and renders PDFs to PNG at a
fixed DPI using PyMuPDF. Produces both a grayscale analysis copy and a
color display copy for each input.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import fitz  # PyMuPDF

from config import PDF_RENDER_DPI, MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS


class IngestionError(Exception):
    """Raised when an uploaded file cannot be safely ingested."""


@dataclass
class IngestedImage:
    filename: str
    color_bgr: np.ndarray  # color, for display
    gray: np.ndarray  # grayscale, for analysis
    source_kind: str  # "pdf" or "image"


def _validate_size(raw_bytes: bytes, filename: str) -> None:
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise IngestionError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB} MB limit."
        )
    if size_mb == 0:
        raise IngestionError(f"'{filename}' is empty.")


def _extension_of(filename: str) -> str:
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx:].lower()


def render_pdf_to_bgr(raw_bytes: bytes, filename: str, dpi: int = PDF_RENDER_DPI) -> np.ndarray:
    """Render the first page of a PDF to a BGR numpy array, capping max dimension at 2000px for memory safety."""
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:  # malformed PDF
        raise IngestionError(f"Could not open '{filename}' as a PDF: {exc}") from exc

    if doc.page_count == 0:
        raise IngestionError(f"'{filename}' has no pages.")

    try:
        page = doc.load_page(0)
        rect = page.rect
        w_pdf, h_pdf = rect.width, rect.height
        
        # Cap max dimension at 2000 pixels to prevent OpenCV OOM allocation crashes
        MAX_DIM = 2000
        zoom = min(dpi / 72.0, MAX_DIM / max(w_pdf, h_pdf))
        
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as exc:
        raise IngestionError(f"Failed to render '{filename}' at {dpi} DPI: {exc}") from exc
    finally:
        doc.close()


def decode_image_to_bgr(raw_bytes: bytes, filename: str) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.thumbnail((2000, 2000))  # Downscale efficiently in PIL memory space
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise IngestionError(f"Could not decode '{filename}' as an image: {exc}") from exc


def ingest_file(raw_bytes: bytes, filename: str) -> IngestedImage:
    """
    Validate and ingest a single uploaded file (PDF or image).

    Returns an IngestedImage with both a color and grayscale representation.
    """
    _validate_size(raw_bytes, filename)

    ext = _extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext}' for '{filename}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        color_bgr = render_pdf_to_bgr(raw_bytes, filename)
        source_kind = "pdf"
    else:
        color_bgr = decode_image_to_bgr(raw_bytes, filename)
        source_kind = "image"

    if color_bgr is None or color_bgr.size == 0:
        raise IngestionError(f"'{filename}' produced an empty image after decoding.")

    # Downscale high-resolution drawings to prevent Out-Of-Memory errors on restricted machines
    h_orig, w_orig = color_bgr.shape[:2]
    MAX_DIM = 2000
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        w_scaled = int(round(w_orig * scale))
        h_scaled = int(round(h_orig * scale))
        color_bgr = cv2.resize(color_bgr, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return IngestedImage(filename=filename, color_bgr=color_bgr, gray=gray, source_kind=source_kind)
