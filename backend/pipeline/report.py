import os
import cv2
import numpy as np
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64

def decode_base64_to_tempfile(b64_str, temp_dir):
    """
    Decodes base64 string to a temporary file and returns the path.
    """
    if not b64_str:
        # Return a small blank placeholder image path
        placeholder = np.ones((50, 50, 3), dtype=np.uint8) * 240
        fd, path = tempfile.mkstemp(suffix=".png", dir=temp_dir)
        os.close(fd)
        cv2.imwrite(path, placeholder)
        return path
        
    try:
        data = base64.b64decode(b64_str)
        fd, path = tempfile.mkstemp(suffix=".png", dir=temp_dir)
        os.close(fd)
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

def save_cv_image_to_temp(img, temp_dir, prefix="img"):
    """
    Saves OpenCV image to a temp file and returns the path.
    """
    fd, path = tempfile.mkstemp(suffix=".png", prefix=prefix, dir=temp_dir)
    os.close(fd)
    cv2.imwrite(path, img)
    return path

def generate_pdf_report(
    img1, img2_aligned, annotated_v2, heatmap, change_records, 
    metadata, ai_summary, output_pdf_path
):
    """
    Generates a beautifully structured PDF report of the drawing comparison using ReportLab.
    """
    # Create temp directory inside workspace for images
    temp_dir = os.path.join(os.path.dirname(output_pdf_path), "temp_report_assets")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Setup document - landscape might fit drawings better
        doc = SimpleDocTemplate(
            output_pdf_path, 
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Define styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=15
        )
        
        h2_style = ParagraphStyle(
            'H2Style',
            parent=styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=12,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#2D3748")
        )
        
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=body_style,
            fontSize=11,
            leading=15,
            backColor=colors.HexColor("#EDF2F7"),
            borderColor=colors.HexColor("#CBD5E0"),
            borderWidth=1,
            borderPadding=10,
            spaceBefore=10,
            spaceAfter=15
        )
        
        story = []
        
        # 1. Header
        story.append(Paragraph("AI-Assisted CAD ElevationDrawing Revision Report", title_style))
        story.append(Spacer(1, 10))
        
        # 2. Metadata Table
        meta_data = [
            [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style)],
            [Paragraph("v1 File Name", body_style), Paragraph(str(metadata.get("file1_name", "N/A")), body_style)],
            [Paragraph("v2 File Name", body_style), Paragraph(str(metadata.get("file2_name", "N/A")), body_style)],
            [Paragraph("Scale Factor (SIFT)", body_style), Paragraph(f"{metadata.get('scale_ratio', 1.0):.4f}", body_style)],
            [Paragraph("Alignment Confidence", body_style), Paragraph(str(metadata.get("alignment_confidence", "N/A")).upper(), body_style)],
            [Paragraph("Total Regions Compared", body_style), Paragraph(str(len(change_records)), body_style)],
        ]
        
        meta_table = Table(meta_data, colWidths=[200, 300])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))
        
        # 3. AI Summary
        story.append(Paragraph("Executive Revision Summary", h2_style))
        story.append(Paragraph(ai_summary, summary_style))
        story.append(Spacer(1, 15))
        
        # Page break before images
        story.append(PageBreak())
        
        # 4. Drawings Showcase
        story.append(Paragraph("Visual Comparison Overlays", h2_style))
        
        # Save overlay & heatmap to temp files for embedding
        overlay_path = save_cv_image_to_temp(annotated_v2, temp_dir, "overlay")
        heatmap_path = save_cv_image_to_temp(heatmap, temp_dir, "heatmap")
        
        # Add Bounding Box Overlay
        story.append(Paragraph("<b>Annotated Elevation Drawing (v2 with changes marked)</b>", body_style))
        story.append(Spacer(1, 5))
        story.append(Image(overlay_path, width=7.2*inch, height=4.5*inch))
        story.append(Spacer(1, 15))
        
        story.append(PageBreak())
        
        # Add Heatmap
        story.append(Paragraph("<b>Full-Image SSIM Similarity Heatmap</b>", body_style))
        story.append(Spacer(1, 5))
        story.append(Image(heatmap_path, width=7.2*inch, height=4.5*inch))
        story.append(Spacer(1, 15))
        
        story.append(PageBreak())
        
        # 5. Change Log Table
        story.append(Paragraph("Detailed Change Log", h2_style))
        
        # Table columns: Region ID | Element Type | Change Type | v1 Value | v2 Value | Confidence | v1 Crop | v2 Crop
        log_headers = [
            Paragraph("<b>ID</b>", body_style),
            Paragraph("<b>Element</b>", body_style),
            Paragraph("<b>Change</b>", body_style),
            Paragraph("<b>v1 Value</b>", body_style),
            Paragraph("<b>v2 Value</b>", body_style),
            Paragraph("<b>v1 Crop</b>", body_style),
            Paragraph("<b>v2 Crop</b>", body_style)
        ]
        
        log_rows = [log_headers]
        
        # Keep track of crop filepaths to clean up later
        crop_filepaths = []
        
        # Only log Modified, Added, or Removed elements to keep report concise
        changed_records = [r for r in change_records if r["change_type"] != "Unchanged"]
        
        for rec in changed_records:
            c1_path = decode_base64_to_tempfile(rec.get("crop_v1", ""), temp_dir)
            c2_path = decode_base64_to_tempfile(rec.get("crop_v2", ""), temp_dir)
            crop_filepaths.extend([c1_path, c2_path])
            
            # Format row elements
            region_id = rec["region_id"]
            if rec.get("low_confidence"):
                region_id += " *"
                
            row = [
                Paragraph(region_id, body_style),
                Paragraph(rec["element_type"], body_style),
                Paragraph(rec["change_type"], body_style),
                Paragraph(rec["v1_value"], body_style),
                Paragraph(rec["v2_value"], body_style),
                Image(c1_path, width=0.8*inch, height=0.6*inch) if c1_path else "",
                Image(c2_path, width=0.8*inch, height=0.6*inch) if c2_path else ""
            ]
            log_rows.append(row)
            
        log_table = Table(log_rows, colWidths=[55, 65, 65, 120, 120, 65, 65])
        log_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4A5568")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        
        story.append(log_table)
        
        if any(r.get("low_confidence") for r in changed_records):
            story.append(Spacer(1, 10))
            story.append(Paragraph("<i>* Note: Region marked with an asterisk (*) indicates low OCR / alignment confidence.</i>", body_style))
            
        # Build document
        doc.build(story)
        
    finally:
        # Clean up temp files
        try:
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    os.remove(os.path.join(root, f))
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp report assets: {e}")
