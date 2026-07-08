from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import fitz
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import DPI


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


@dataclass
class IngestedDrawing:
    image: np.ndarray
    filename: str
    source_type: str
    metadata: dict


def _read_input(input_data: str | Path | bytes | BinaryIO, filename: str | None) -> tuple[bytes, str]:
    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Input file not found: {path}")
        return path.read_bytes(), filename or path.name

    if isinstance(input_data, bytes):
        if not filename:
            raise ValueError("A filename is required when ingesting raw bytes.")
        return input_data, filename

    if hasattr(input_data, "read"):
        data = input_data.read()
        if not isinstance(data, bytes):
            raise ValueError("File-like input did not return bytes.")
        resolved_name = filename or getattr(input_data, "name", None)
        if not resolved_name:
            raise ValueError("A filename is required when ingesting a file-like object.")
        return data, Path(resolved_name).name

    raise ValueError("Unsupported input type. Provide a file path, bytes, or a binary file object.")


def _render_pdf(data: bytes, filename: str) -> np.ndarray:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{filename}'. The file may be corrupt.") from exc

    try:
        if doc.page_count < 1:
            raise ValueError(f"PDF '{filename}' has no pages.")
        page = doc.load_page(0)
        zoom = DPI / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            image = np.repeat(image, 3, axis=2)
        return image[:, :, :3].copy()
    except Exception as exc:
        raise ValueError(f"Could not render PDF '{filename}' at {DPI} DPI.") from exc
    finally:
        doc.close()


def _load_image(data: bytes, filename: str) -> np.ndarray:
    try:
        with Image.open(__import__("io").BytesIO(data)) as img:
            img.verify()
        with Image.open(__import__("io").BytesIO(data)) as img:
            return np.array(img.convert("RGB"))
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not identify image '{filename}'. The file may be corrupt.") from exc
    except Exception as exc:
        raise ValueError(f"Could not load image '{filename}'.") from exc


def ingest_drawing(input_data: str | Path | bytes | BinaryIO, filename: str | None = None) -> IngestedDrawing:
    data, resolved_filename = _read_input(input_data, filename)
    ext = Path(resolved_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext or 'unknown'}'. Supported types: PDF, PNG, JPG."
        )

    if ext == ".pdf":
        image = _render_pdf(data, resolved_filename)
        source_type = "pdf"
    else:
        image = _load_image(data, resolved_filename)
        source_type = "image"

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"'{resolved_filename}' did not produce a valid RGB image.")

    metadata = {
        "filename": resolved_filename,
        "source_type": source_type,
        "dpi": DPI if source_type == "pdf" else None,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    print(f"[Stage 1] Ingested {resolved_filename}: {image.shape[1]}x{image.shape[0]} ({source_type})")
    return IngestedDrawing(image=image, filename=resolved_filename, source_type=source_type, metadata=metadata)


def ingest_pair(
    drawing_a: str | Path | bytes | BinaryIO,
    drawing_b: str | Path | bytes | BinaryIO,
    filename_a: str | None = None,
    filename_b: str | None = None,
) -> tuple[IngestedDrawing, IngestedDrawing]:
    return ingest_drawing(drawing_a, filename_a), ingest_drawing(drawing_b, filename_b)
