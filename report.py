from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_engineering.visualize import CHANGE_COLORS, crop_image


def image_to_compressed_jpeg_bytes(image: np.ndarray, max_dim: int = 3600) -> bytes:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # 1. Resize image if its longest side exceeds max_dim
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h, new_w = max(new_h, 1), max(new_w, 1)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    # 2. Compress using JPEG format with 85% quality for sharp, readable details around 20MB total size
    with BytesIO() as buffer:
        Image.fromarray(image).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


def build_pdf_report(
    image_a: np.ndarray,
    aligned_b: np.ndarray,
    diff_map: np.ndarray,
    records: list[dict],
    visuals: dict,
    run_metadata: dict,
    ai_summary: str,
) -> bytes:
    import fitz
    
    filenames = run_metadata.get("filenames", {})
    alignment = run_metadata.get("alignment", {})
    diff = run_metadata.get("diff", {})
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = fitz.open()

    c_dark = (0.12, 0.16, 0.20)
    c_white = (1.0, 1.0, 1.0)
    c_text = (0.2, 0.2, 0.2)

    # PAGE 1: Header + Side-by-Side
    page = doc.new_page(width=842, height=595)
    page.draw_rect(fitz.Rect(0, 0, 842, 100), color=c_dark, fill=c_dark)
    page.insert_text(fitz.Point(30, 40), "CAD Elevation Revision Comparison Report", fontsize=20, color=c_white, fontname="helvetica-bold")
    
    meta_text = (
        f"Drawing A: {filenames.get('drawing_a', '')}\n"
        f"Drawing B: {filenames.get('drawing_b', '')}\n"
        f"Date: {generated}\n"
        f"Alignment: {alignment.get('alignment_method', '')} (Scale: {alignment.get('scale_factor', 1.0):.3f})"
    )
    page.insert_textbox(fitz.Rect(30, 50, 812, 95), meta_text, fontsize=9, color=c_white, fontname="helvetica")

    # Compress side-by-side to max 3600px width
    page.insert_image(fitz.Rect(30, 140, 812, 550), stream=image_to_compressed_jpeg_bytes(visuals["side_by_side"], max_dim=3600))

    # PAGE 2: Annotated Bounding Box Overlay
    page = doc.new_page(width=842, height=595)
    page.draw_rect(fitz.Rect(0, 0, 842, 50), color=c_dark, fill=c_dark)
    page.insert_text(fitz.Point(30, 32), "Annotated Bounding Box Overlay", fontsize=16, color=c_white, fontname="helvetica-bold")
    
    legend_text = "Legend:   [Added (Green)]   [Removed (Red)]   [Modified (Yellow)]   [Possible Change (Purple)]"
    page.insert_text(fitz.Point(30, 70), legend_text, fontsize=10, color=c_text, fontname="helvetica")
    # Compress overlay to max 3600px width
    page.insert_image(fitz.Rect(30, 85, 812, 550), stream=image_to_compressed_jpeg_bytes(visuals["overlay"], max_dim=3600))

    # PAGE 3: SSIM Heatmap
    page = doc.new_page(width=842, height=595)
    page.draw_rect(fitz.Rect(0, 0, 842, 50), color=c_dark, fill=c_dark)
    page.insert_text(fitz.Point(30, 32), "SSIM Structural Similarity Heatmap", fontsize=16, color=c_white, fontname="helvetica-bold")
    # Compress heatmap to max 3600px width
    page.insert_image(fitz.Rect(30, 85, 812, 550), stream=image_to_compressed_jpeg_bytes(visuals["heatmap"], max_dim=3600))

    # PAGE 4: Change Log & AI Summary
    page = doc.new_page(width=842, height=595)
    page.draw_rect(fitz.Rect(0, 0, 842, 50), color=c_dark, fill=c_dark)
    page.insert_text(fitz.Point(30, 32), "AI Summary & Change Log", fontsize=16, color=c_white, fontname="helvetica-bold")
    
    page.insert_text(fitz.Point(30, 75), "AI-Generated Summary Report:", fontsize=12, color=c_dark, fontname="helvetica-bold")
    page.insert_textbox(fitz.Rect(30, 90, 812, 190), ai_summary, fontsize=10, color=c_text, fontname="helvetica")

    page.insert_text(fitz.Point(30, 215), f"Change Log ({len(records)} regions detected):", fontsize=12, color=c_dark, fontname="helvetica-bold")
    
    headers = ["ID", "Category", "Location", "Change Type", "Drawing A Crop", "Drawing B Crop"]
    header_x_positions = [30, 80, 240, 340, 480, 650]
    
    y_draw = 235
    for header, x_pos in zip(headers, header_x_positions):
        page.insert_text(fitz.Point(x_pos, y_draw), header, fontsize=10, color=c_dark, fontname="helvetica-bold")
    
    page.draw_line(fitz.Point(30, y_draw + 5), fitz.Point(812, y_draw + 5), color=c_dark, width=1)
    
    y_draw += 15
    for record in records:
        if y_draw + 45 > 570:
            page = doc.new_page(width=842, height=595)
            page.draw_rect(fitz.Rect(0, 0, 842, 50), color=c_dark, fill=c_dark)
            page.insert_text(fitz.Point(30, 32), "Change Log (Continued)", fontsize=16, color=c_white, fontname="helvetica-bold")
            y_draw = 70
            for header, x_pos in zip(headers, header_x_positions):
                page.insert_text(fitz.Point(x_pos, y_draw), header, fontsize=10, color=c_dark, fontname="helvetica-bold")
            page.draw_line(fitz.Point(30, y_draw + 5), fitz.Point(812, y_draw + 5), color=c_dark, width=1)
            y_draw += 15
            
        page.insert_text(fitz.Point(header_x_positions[0], y_draw + 15), record["region_id"], fontsize=9, color=c_text, fontname="helvetica")
        page.insert_text(fitz.Point(header_x_positions[1], y_draw + 15), str(record["element_type_or_category"]), fontsize=9, color=c_text, fontname="helvetica")
        page.insert_text(fitz.Point(header_x_positions[2], y_draw + 15), str(record["location_description"]), fontsize=9, color=c_text, fontname="helvetica")
        page.insert_text(fitz.Point(header_x_positions[3], y_draw + 15), str(record["change_type"]), fontsize=9, color=c_text, fontname="helvetica")
        
        # Crop A and B side-by-side inside the table row
        crop_a = crop_image(image_a, record["bbox"], pad=20)
        crop_b = crop_image(aligned_b, record["bbox"], pad=20)
        
        # Compress thumbnails to max 800px for table columns to preserve deep zoom sharpness
        page.insert_image(
            fitz.Rect(header_x_positions[4], y_draw, header_x_positions[4] + 120, y_draw + 35),
            stream=image_to_compressed_jpeg_bytes(crop_a, max_dim=800)
        )
        page.insert_image(
            fitz.Rect(header_x_positions[5], y_draw, header_x_positions[5] + 120, y_draw + 35),
            stream=image_to_compressed_jpeg_bytes(crop_b, max_dim=800)
        )
        
        page.draw_line(fitz.Point(30, y_draw + 38), fitz.Point(812, y_draw + 38), color=(0.9, 0.9, 0.9), width=0.5)
        y_draw += 42

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes
