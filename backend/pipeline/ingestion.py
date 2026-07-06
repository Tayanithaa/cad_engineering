import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import config

def ingest_image_or_pdf(file_path_or_bytes, filename=None):
    """
    Accepts a file path or bytes, checks the type (PDF/JPG/PNG),
    renders PDF to a NumPy image at config.DPI, and returns a BGR NumPy array.
    """
    # If it is bytes, we need to know the filename to determine the extension
    is_pdf = False
    
    if isinstance(file_path_or_bytes, str):
        if not os.path.exists(file_path_or_bytes):
            raise FileNotFoundError(f"Input file not found: {file_path_or_bytes}")
        ext = os.path.splitext(file_path_or_bytes)[1].lower()
        if ext == '.pdf':
            is_pdf = True
        elif ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        # It's bytes
        if not filename:
            raise ValueError("filename must be provided when passing bytes")
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            is_pdf = True
        elif ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            raise ValueError(f"Unsupported file format: {ext}")

    if is_pdf:
        # Load PDF using PyMuPDF
        try:
            if isinstance(file_path_or_bytes, str):
                doc = fitz.open(file_path_or_bytes)
            else:
                doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
            
            if len(doc) == 0:
                raise ValueError("PDF file contains no pages")
            
            # Load first page
            page = doc.load_page(0)
            
            # Render to image at config.DPI
            zoom = config.DPI / 72.0  # 72 is default PDF points per inch
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to numpy array
            img_data = pix.tobytes("png")
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode rendered PDF page to image")
            return img
        except Exception as e:
            raise ValueError(f"Failed to process PDF file: {str(e)}")
    else:
        # Image file
        try:
            if isinstance(file_path_or_bytes, str):
                img = cv2.imread(file_path_or_bytes, cv2.IMREAD_COLOR)
            else:
                img = cv2.imdecode(np.frombuffer(file_path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image. File might be corrupt.")
            return img
        except Exception as e:
            raise ValueError(f"Failed to process image file: {str(e)}")
