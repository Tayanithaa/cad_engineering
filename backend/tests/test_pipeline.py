import os
import sys
import cv2
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from backend.pipeline import ingestion
from backend.pipeline import preprocess
from backend.pipeline import alignment
from backend.pipeline import grid_detection
from backend.pipeline import element_detection
from backend.pipeline import diff
from backend.pipeline import merge
from backend.pipeline import visualize
from backend.pipeline import report

def create_mock_drawings(output_dir):
    """
    Creates two mock elevation drawings (v1 and v2) as PNGs to test the comparison pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Version 1 (v1)
    # 2400 x 1800 (landscape high-resolution drawing, typical of 300 DPI layout)
    v1 = np.ones((1800, 2400, 3), dtype=np.uint8) * 255
    
    # Drawing border
    cv2.rectangle(v1, (100, 100), (2300, 1700), (0, 0, 0), 5)
    
    # Facade Outline / Ground Line
    cv2.line(v1, (200, 1500), (2200, 1500), (0, 0, 0), 4)
    cv2.rectangle(v1, (400, 600), (2000, 1500), (0, 0, 0), 3) # facade walls
    
    # Grid of Windows: 4 windows in a 2x2 layout
    # Window centers at X=[800, 1600], Y=[800, 1200]
    win_w, win_h = 200, 200
    win_centers_x = [800, 1600]
    win_centers_y = [800, 1200]
    
    for cx in win_centers_x:
        for cy in win_centers_y:
            # Draw window frame and panes
            cv2.rectangle(v1, (cx - 100, cy - 100), (cx + 100, cy + 100), (0, 0, 0), 2)
            cv2.line(v1, (cx, cy - 100), (cx, cy + 100), (0, 0, 0), 1)
            cv2.line(v1, (cx - 100, cy), (cx + 100, cy), (0, 0, 0), 1)
            
    # Door at the bottom center (X=1200, Y=1350)
    cv2.rectangle(v1, (1100, 1200), (1300, 1500), (0, 0, 0), 2)
    # Door handle
    cv2.circle(v1, (1270, 1350), 6, (0, 0, 0), -1)
    
    # Roof outline (triangular top)
    cv2.line(v1, (400, 600), (1200, 300), (0, 0, 0), 3)
    cv2.line(v1, (1200, 300), (2000, 600), (0, 0, 0), 3)
    
    # Scale Callout Text
    cv2.putText(v1, "SCALE: 1/8\" = 1'-0\"", (1800, 1620), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(v1, "FRONT ELEVATION", (1100, 1620), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    v1_path = os.path.join(output_dir, "drawing_v1.png")
    cv2.imwrite(v1_path, v1)
    
    # 2. Version 2 (v2) - has some changes:
    # - Slightly shifted drawing border to test alignment
    # - Window at X=1600, Y=800 is deleted (removed)
    # - Window at X=800, Y=1200 has modified size (modified)
    # - Scale callout text updated to 3/16" = 1'-0"
    v2 = np.ones((1800, 2400, 3), dtype=np.uint8) * 255
    
    # Slightly shifted border (by 20px horizontally, 10px vertically)
    cv2.rectangle(v2, (120, 110), (2320, 1710), (0, 0, 0), 5)
    
    # Facade Outline
    cv2.line(v2, (220, 1510), (2220, 1510), (0, 0, 0), 4)
    cv2.rectangle(v2, (420, 610), (2020, 1510), (0, 0, 0), 3)
    
    # Windows
    # Window 1 (X=800, Y=800 -> shifted to X=820, Y=810) - Unchanged structurally
    cx1, cy1 = 820, 810
    cv2.rectangle(v2, (cx1 - 100, cy1 - 100), (cx1 + 100, cy1 + 100), (0, 0, 0), 2)
    cv2.line(v2, (cx1, cy1 - 100), (cx1, cy1 + 100), (0, 0, 0), 1)
    cv2.line(v2, (cx1 - 100, cy1), (cx1 + 100, cy1), (0, 0, 0), 1)
    
    # Window 2 (X=1600, Y=800 -> shifted to X=1620, Y=810) - DELETED (not drawn)
    
    # Window 3 (X=800, Y=1200 -> shifted to X=820, Y=1210) - MODIFIED: Make it taller
    cx3, cy3 = 820, 1210
    cv2.rectangle(v2, (cx3 - 100, cy3 - 140), (cx3 + 100, cy3 + 100), (0, 0, 0), 2) # taller
    
    # Window 4 (X=1600, Y=1200 -> shifted to X=1620, Y=1210) - Unchanged structurally
    cx4, cy4 = 1620, 1210
    cv2.rectangle(v2, (cx4 - 100, cy4 - 100), (cx4 + 100, cy4 + 100), (0, 0, 0), 2)
    cv2.line(v2, (cx4, cy4 - 100), (cx4, cy4 + 100), (0, 0, 0), 1)
    
    # Door
    cv2.rectangle(v2, (1120, 1210), (1320, 1510), (0, 0, 0), 2)
    cv2.circle(v2, (1290, 1360), 6, (0, 0, 0), -1)
    
    # Roof (triangular top)
    cv2.line(v2, (420, 610), (1220, 310), (0, 0, 0), 3)
    cv2.line(v2, (1220, 310), (2020, 610), (0, 0, 0), 3)
    
    # Scale Callout Text (Modified value)
    cv2.putText(v2, "SCALE: 3/16\" = 1'-0\"", (1820, 1630), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(v2, "FRONT ELEVATION", (1120, 1630), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    v2_path = os.path.join(output_dir, "drawing_v2.png")
    cv2.imwrite(v2_path, v2)
    
    return v1_path, v2_path

def run_test_pipeline():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
    v1_path, v2_path = create_mock_drawings(output_dir)
    print(f"Mock drawings created:\n v1: {v1_path}\n v2: {v2_path}")
    
    # Ingest
    print("\n--- STAGE 1: Ingestion ---")
    img1 = ingestion.ingest_image_or_pdf(v1_path)
    img2 = ingestion.ingest_image_or_pdf(v2_path)
    print(f"Loaded v1 shape: {img1.shape}, v2 shape: {img2.shape}")
    
    # Preprocess
    print("\n--- STAGE 2: Border Crop ---")
    img1_cropped, bbox1 = preprocess.detect_and_crop_border(img1)
    img2_cropped, bbox2 = preprocess.detect_and_crop_border(img2)
    print(f"Cropped v1 bbox: {bbox1}, Cropped v2 bbox: {bbox2}")
    
    # Alignment
    print("\n--- STAGE 3: Alignment ---")
    # For testing, bypass OCR or use fallback
    img2_aligned, align_meta = alignment.align_images(img1_cropped, img2_cropped, ocr_engine="easyocr")
    print(f"Alignment Meta: {json.dumps(align_meta, indent=2)}")
    
    # Grid detection
    print("\n--- STAGE 4: Grid Detection ---")
    grid = grid_detection.detect_facade_grid(img1_cropped, img2_aligned)
    print(f"Detected Facade Grid peaks:\n X (cols): {grid['x_peaks']}\n Y (rows): {grid['y_peaks']}")
    
    # Element detection
    print("\n--- STAGE 5: Element Detection ---")
    elements_v1 = element_detection.detect_all_elements(img1_cropped, grid)
    elements_v2 = element_detection.detect_all_elements(img2_aligned, grid)
    print(f"Detected elements v1: {len(elements_v1)}")
    print(f"Detected elements v2: {len(elements_v2)}")
    
    # Diff, Compare, Merge
    print("\n--- STAGES 6-9: Diff, OCR, Compare, Merge ---")
    change_records = merge.merge_pipeline_results(
        img1_cropped, img2_aligned, elements_v1, elements_v2, grid, ocr_engine="easyocr"
    )
    
    # Print the JSON Change Log (user request to inspect it before Groq/Summary wiring)
    print("\n--- CONSOLIDATED CHANGE RECORD JSON ---")
    print(json.dumps(change_records, indent=2))
    
    # Save JSON to disk for inspection
    json_path = os.path.join(output_dir, "change_records.json")
    with open(json_path, "w") as f:
        json.dump(change_records, f, indent=2)
    print(f"Saved change records to {json_path}")
    
    # Visualize & Report
    print("\n--- STAGES 10-12: Visuals & PDF Report ---")
    annotated_v2, records_with_thumbs = visualize.generate_visualizations(
        img1_cropped, img2_aligned, change_records
    )
    heatmap = diff.generate_ssim_heatmap(img1_cropped, img2_aligned)
    
    metadata = {
        "file1_name": "drawing_v1.png",
        "file2_name": "drawing_v2.png",
        "scale_ratio": align_meta["scale_ratio"],
        "alignment_confidence": align_meta["alignment_confidence"],
        "status_message": align_meta["status_message"],
        "total_regions": len(change_records)
    }
    
    pdf_path = os.path.join(output_dir, "mock_report.pdf")
    report.generate_pdf_report(
        img1_cropped, img2_aligned, annotated_v2, heatmap, 
        records_with_thumbs, metadata, 
        "Mock AI summary: Windows at grid column indices were modified/deleted. Scale updated from 1/8 to 3/16.", 
        pdf_path
    )
    print(f"PDF report generated at {pdf_path}")

if __name__ == "__main__":
    run_test_pipeline()
