import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
import tempfile
import json

from backend.pipeline import ingestion
from backend.pipeline import preprocess
from backend.pipeline import alignment
from backend.pipeline import grid_detection
from backend.pipeline import element_detection
from backend.pipeline import diff
from backend.pipeline import merge
from backend.pipeline import visualize
from backend.pipeline import report
from backend.pipeline import ai_summary

app = FastAPI(title="AI-Assisted CAD Revision Comparator API")

@app.post("/compare")
async def compare_drawings(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    ocr_engine: str = Form("paddleocr")
):
    try:
        # Read uploaded files
        f1_bytes = await file1.read()
        f2_bytes = await file2.read()
        
        # 1. Ingestion
        try:
            img1 = ingestion.ingest_image_or_pdf(f1_bytes, file1.filename)
            img2 = ingestion.ingest_image_or_pdf(f2_bytes, file2.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")
            
        # 2. Border crop / normalization
        img1_cropped, bbox1 = preprocess.detect_and_crop_border(img1)
        img2_cropped, bbox2 = preprocess.detect_and_crop_border(img2)
        
        # 3. Alignment
        img2_aligned, align_meta = alignment.align_images(img1_cropped, img2_cropped, ocr_engine)
        
        # 4. Grid/Rhythm Detection
        grid = grid_detection.detect_facade_grid(img1_cropped, img2_aligned)
        
        # 5. Element Detection
        elements_v1 = element_detection.detect_all_elements(img1_cropped, grid)
        elements_v2 = element_detection.detect_all_elements(img2_aligned, grid)
        
        # 6 & 7 & 8 & 9. Diff, OCR, Compare and Merge
        change_records = merge.merge_pipeline_results(
            img1_cropped, img2_aligned, elements_v1, elements_v2, grid, ocr_engine
        )
        
        # Generate overlays and heatmap
        annotated_v2, records_with_thumbnails = visualize.generate_visualizations(
            img1_cropped, img2_aligned, change_records
        )
        heatmap = diff.generate_ssim_heatmap(img1_cropped, img2_aligned)
        
        # Create metadata structure
        metadata = {
            "file1_name": file1.filename,
            "file2_name": file2.filename,
            "scale_ratio": align_meta["scale_ratio"],
            "alignment_confidence": align_meta["alignment_confidence"],
            "status_message": align_meta["status_message"],
            "total_regions": len(change_records)
        }
        
        # 10. AI Summary
        # Generate summary (will return error message fallback if API key is not present)
        ai_narrative = ai_summary.generate_revision_summary(records_with_thumbnails, metadata)
        
        # 11. PDF Report Generation
        # We will save the PDF to a temporary file and return its path/response
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, "CAD_Revision_Report.pdf")
        
        report.generate_pdf_report(
            img1_cropped, img2_aligned, annotated_v2, heatmap, 
            records_with_thumbnails, metadata, ai_narrative, pdf_path
        )
        
        # We return the JSON containing metadata, the change log, and the path to download the PDF
        # To avoid keeping large image strings in memory, we can return the thumbnail base64s too
        return JSONResponse(content={
            "metadata": metadata,
            "ai_summary": ai_narrative,
            "change_records": records_with_thumbnails,
            "pdf_report_path": pdf_path
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@app.get("/download-report")
async def download_report(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(path, media_type="application/pdf", filename="CAD_Revision_Report.pdf")
