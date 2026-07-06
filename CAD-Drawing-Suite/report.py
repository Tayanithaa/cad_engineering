"""
Report generation: builds a single self-contained downloadable HTML report
covering metadata, side-by-side images, overlays, heatmap, change log,
statistics, AI summary, and an appendix of crop pairs.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
import tempfile
import io
import os

import cv2
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

CHANGE_TYPE_COLORS_BGR = {
    "Modified": (0, 0, 255),   # red
    "Added": (255, 0, 0),      # blue
    "Removed": (0, 200, 0),    # green
}


def _bgr_to_data_uri(image: np.ndarray, fmt: str = ".png") -> str:
    ok, buf = cv2.imencode(fmt, image)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def draw_annotated_overlay(v2_aligned_color: np.ndarray, records: list) -> np.ndarray:
    """Draw colored bounding boxes on v2 based on change_type."""
    overlay = v2_aligned_color.copy()
    h_img, w_img = overlay.shape[:2]

    # Calculate adaptive sizing based on resolution
    base_dim = min(h_img, w_img)
    thickness = max(2, int(base_dim / 300))
    font_scale = max(0.6, base_dim / 1200.0)
    text_thickness = max(1, int(thickness / 2))

    for r in records:
        x, y, w, h = r.bbox
        color = CHANGE_TYPE_COLORS_BGR.get(r.change_type, (0, 165, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
        label = f"{r.region_id}"

        # Draw text background box for better visibility
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        ty = max(th + 10, y - 6)
        cv2.rectangle(overlay, (x, ty - th - 4), (x + tw + 6, ty + baseline), (255, 255, 255), -1) # white bg
        cv2.putText(overlay, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)
    return overlay


def _esc(text) -> str:
    return html.escape(str(text))


def build_statistics(records: list) -> dict:
    from collections import Counter

    change_counts = Counter(r.change_type for r in records)
    category_counts = Counter(r.category for r in records)
    return {
        "total_regions": len(records),
        "by_change_type": dict(change_counts),
        "by_category": dict(category_counts),
    }


def generate_html_report(
    *,
    v1_filename: str,
    v2_filename: str,
    v1_color: np.ndarray,
    v2_aligned_color: np.ndarray,
    annotated_overlay: np.ndarray,
    heatmap: np.ndarray,
    scale_ratio: float,
    low_confidence: bool,
    alignment_message: str,
    records: list,
    percent_area_changed: float,
    ai_summary: str,
    ai_summary_succeeded: bool,
    region_crops: dict,
) -> str:
    stats = build_statistics(records)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    v1_uri = _bgr_to_data_uri(v1_color)
    v2_uri = _bgr_to_data_uri(v2_aligned_color)
    overlay_uri = _bgr_to_data_uri(annotated_overlay)
    heatmap_uri = _bgr_to_data_uri(heatmap)

    rows_html = []
    for r in records:
        rows_html.append(
            f"<tr data-category='{_esc(r.category)}' data-changetype='{_esc(r.change_type)}'>"
            f"<td>{_esc(r.region_id)}</td>"
            f"<td>{_esc(r.category)}</td>"
            f"<td>{_esc(r.location_description)}</td>"
            f"<td class='ct-{_esc(r.change_type.lower())}'>{_esc(r.change_type)}</td>"
            f"<td>{_esc(r.v1_value)}</td>"
            f"<td>{_esc(r.v2_value)}</td>"
            f"</tr>"
        )
    change_log_rows = "\n".join(rows_html) if rows_html else "<tr><td colspan='6'>No changes detected.</td></tr>"

    category_rows = "\n".join(
        f"<tr><td>{_esc(cat)}</td><td>{count}</td></tr>" for cat, count in stats["by_category"].items()
    ) or "<tr><td colspan='2'>—</td></tr>"

    change_type_rows = "\n".join(
        f"<tr><td>{_esc(ct)}</td><td>{count}</td></tr>" for ct, count in stats["by_change_type"].items()
    ) or "<tr><td colspan='2'>—</td></tr>"

    appendix_items = []
    for r in records:
        crops = region_crops.get(r.region_id)
        if not crops:
            continue
        c1_uri = _bgr_to_data_uri(crops[0]) if crops[0] is not None and crops[0].size else ""
        c2_uri = _bgr_to_data_uri(crops[1]) if crops[1] is not None and crops[1].size else ""
        appendix_items.append(
            f"<details class='appendix-item'>"
            f"<summary>{_esc(r.region_id)} — {_esc(r.category)} ({_esc(r.change_type)})</summary>"
            f"<div class='crop-pair'>"
            f"<div><p>v1</p>{'<img src=' + repr(c1_uri) + ' />' if c1_uri else '<p>(no crop)</p>'}</div>"
            f"<div><p>v2</p>{'<img src=' + repr(c2_uri) + ' />' if c2_uri else '<p>(no crop)</p>'}</div>"
            f"</div></details>"
        )
    appendix_html = "\n".join(appendix_items) if appendix_items else "<p>No regions to display.</p>"

    low_conf_banner = (
        "<div class='banner warning'>⚠ Low-confidence alignment detected — "
        "results below may be less reliable than usual.</div>"
        if low_confidence else ""
    )

    ai_note = "" if ai_summary_succeeded else "<p class='muted'>AI summary generation did not complete successfully.</p>"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>CAD Drawing Revision Comparison Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0; background: #f5f6f8; color: #1a1a1a; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 80px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin-top: 40px; border-bottom: 2px solid #e2e4e8; padding-bottom: 8px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #fff; }}
  th, td {{ border: 1px solid #e2e4e8; padding: 8px 10px; text-align: left; font-size: 14px; }}
  th {{ background: #eef0f4; cursor: pointer; user-select: none; }}
  th.sortable::after {{ content: ' \\21C5'; color: #999; font-size: 11px; }}
  .meta-table td:first-child {{ font-weight: 600; width: 220px; }}
  .images-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
  .images-grid img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .legend {{ display: flex; gap: 20px; margin-top: 10px; font-size: 13px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
  .ct-modified {{ color: #c0392b; font-weight: 600; }}
  .ct-added {{ color: #2454c7; font-weight: 600; }}
  .ct-removed {{ color: #1e8449; font-weight: 600; }}
  .banner {{ padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 14px; }}
  .banner.warning {{ background: #fff3cd; border: 1px solid #ffe08a; color: #7a5b00; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }}
  .stat-card {{ background: #fff; border: 1px solid #e2e4e8; border-radius: 6px; padding: 16px; text-align: center; }}
  .stat-card .value {{ font-size: 28px; font-weight: 700; }}
  .stat-card .label {{ color: #666; font-size: 12px; margin-top: 4px; }}
  .breakdown-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .summary-box {{ background: #fff; border: 1px solid #e2e4e8; border-radius: 6px; padding: 20px; line-height: 1.6; margin-top: 12px; }}
  .muted {{ color: #999; font-size: 13px; }}
  .appendix-item {{ background: #fff; border: 1px solid #e2e4e8; border-radius: 6px; margin-top: 10px; padding: 10px 14px; }}
  .appendix-item summary {{ cursor: pointer; font-weight: 600; }}
  .crop-pair {{ display: flex; gap: 20px; margin-top: 10px; }}
  .crop-pair img {{ max-width: 320px; border: 1px solid #ddd; }}
  footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 60px; }}
</style>
</head>
<body>
<div class="container">
  <h1>CAD Drawing Revision Comparison Report</h1>
  <div class="subtitle">Generated {_esc(now)}</div>

  {low_conf_banner}

  <h2>Header / Metadata</h2>
  <table class="meta-table">
    <tr><td>v1 filename</td><td>{_esc(v1_filename)}</td></tr>
    <tr><td>v2 filename</td><td>{_esc(v2_filename)}</td></tr>
    <tr><td>Date generated</td><td>{_esc(now)}</td></tr>
    <tr><td>Detected scale ratio</td><td>{scale_ratio:.3f}x</td></tr>
    <tr><td>Alignment note</td><td>{_esc(alignment_message)}</td></tr>
    <tr><td>Total regions changed</td><td>{len(records)}</td></tr>
    <tr><td>Percent area changed</td><td>{percent_area_changed:.2f}%</td></tr>
  </table>

  <h2>Side-by-Side Comparison</h2>
  <div class="images-grid">
    <div><p><strong>v1 (reference)</strong></p><img src="{v1_uri}" /></div>
    <div><p><strong>v2 (aligned)</strong></p><img src="{v2_uri}" /></div>
  </div>

  <h2>Annotated Overlay (v2)</h2>
  <img src="{overlay_uri}" style="max-width:100%; border:1px solid #ddd; border-radius:4px;" />
  <div class="legend">
    <span><span class="swatch" style="background:#e74c3c"></span> Modified</span>
    <span><span class="swatch" style="background:#2454c7"></span> Added</span>
    <span><span class="swatch" style="background:#1e8449"></span> Removed</span>
  </div>

  <h2>Heatmap (Change Density)</h2>
  <img src="{heatmap_uri}" style="max-width:100%; border:1px solid #ddd; border-radius:4px;" />

  <h2>Change Log</h2>
  <table id="changelog">
    <thead>
      <tr>
        <th>Region ID</th>
        <th class="sortable" onclick="sortTable(1)">Category</th>
        <th>Location</th>
        <th class="sortable" onclick="sortTable(3)">Change Type</th>
        <th>v1 Value</th>
        <th>v2 Value</th>
      </tr>
    </thead>
    <tbody>
      {change_log_rows}
    </tbody>
  </table>

  <h2>Statistics</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{stats['total_regions']}</div><div class="label">Total Regions</div></div>
    <div class="stat-card"><div class="value">{percent_area_changed:.2f}%</div><div class="label">Area Changed</div></div>
    <div class="stat-card"><div class="value">{scale_ratio:.2f}x</div><div class="label">Scale Ratio</div></div>
  </div>
  <div class="breakdown-grid">
    <div>
      <h3>By Category</h3>
      <table><thead><tr><th>Category</th><th>Count</th></tr></thead><tbody>{category_rows}</tbody></table>
    </div>
    <div>
      <h3>By Change Type</h3>
      <table><thead><tr><th>Change Type</th><th>Count</th></tr></thead><tbody>{change_type_rows}</tbody></table>
    </div>
  </div>

  <h2>AI-Generated Summary</h2>
  <div class="summary-box">{_esc(ai_summary)}</div>
  {ai_note}

  <h2>Appendix: Full-Resolution Crop Pairs</h2>
  {appendix_html}

  <footer>CAD Drawing Revision Comparator — classical CV/OCR detection, single Groq AI summary call.</footer>
</div>
<script>
function sortTable(colIdx) {{
  const table = document.getElementById('changelog');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const asc = table.dataset.sortAsc !== 'true';
  table.dataset.sortAsc = asc;
  rows.sort((a, b) => {{
    const av = a.children[colIdx].innerText.toLowerCase();
    const bv = b.children[colIdx].innerText.toLowerCase();
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>
"""
    return html_doc


def generate_pdf_report(
    *,
    v1_filename: str,
    v2_filename: str,
    v1_color: np.ndarray,
    v2_aligned_color: np.ndarray,
    annotated_overlay: np.ndarray,
    heatmap: np.ndarray,
    scale_ratio: float,
    low_confidence: bool,
    alignment_message: str,
    records: list,
    percent_area_changed: float,
    ai_summary: str,
    ai_summary_succeeded: bool,
) -> bytes:
    buffer = io.BytesIO()

    # Page dimensions on letter: 612 x 792. Margins: 36 pt. Printable area: 540 x 720.
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2D3748')
    )

    summary_style = ParagraphStyle(
        'AISummaryStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1A202C')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2D3748')
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    story = []
    temp_files = []

    def add_image(img, target_width=500):
        try:
            h, w = img.shape[:2]
            aspect = h / w
            target_height = int(target_width * aspect)
            if target_height > 300:
                target_height = 300
                target_width = int(target_height / aspect)

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            cv2.imwrite(tmp_path, img)
            temp_files.append(tmp_path)

            return Image(tmp_path, width=target_width, height=target_height)
        except Exception:
            return Paragraph("[Image rendering error]", body_style)

    try:
        # Title
        story.append(Paragraph("CAD Drawing Revision Comparison Report", title_style))

        # Metadata / Info Block
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_data = [
            [Paragraph("<b>Original (v1):</b>", body_style), Paragraph(v1_filename, body_style),
             Paragraph("<b>Date Generated:</b>", body_style), Paragraph(now_str, body_style)],
            [Paragraph("<b>Revised (v2):</b>", body_style), Paragraph(v2_filename, body_style),
             Paragraph("<b>Area Changed:</b>", body_style), Paragraph(f"{percent_area_changed:.2f}%", body_style)],
            [Paragraph("<b>Scale Ratio:</b>", body_style), Paragraph(f"{scale_ratio:.2f}x", body_style),
             Paragraph("<b>Confidence:</b>", body_style), Paragraph("Low Confidence" if low_confidence else "High Confidence", body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[90, 180, 90, 180])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#EDF2F7')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # AI Summary Callout
        story.append(Paragraph("Summary of Changes", section_style))
        summary_text = ai_summary if ai_summary else "No summary available."
        summary_p = Paragraph(summary_text, summary_style)

        summary_table = Table([[summary_p]], colWidths=[540])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),  # soft light blue
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#BEE3F8')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # Annotated Overlay
        story.append(Paragraph("Annotated Visual Differences", section_style))
        story.append(Paragraph("Red boxes indicate modified areas, blue indicates added elements, and green indicates removed elements.", body_style))
        story.append(Spacer(1, 6))
        story.append(add_image(annotated_overlay, target_width=500))

        story.append(PageBreak())

        # Heatmap (only if area changed percentage is greater than 0, i.e., in Standard mode)
        if percent_area_changed > 0.0:
            story.append(Paragraph("Change-Density Heatmap", section_style))
            story.append(add_image(heatmap, target_width=500))
            story.append(Spacer(1, 10))

        # Detailed Change Log Table
        story.append(Paragraph("Detailed Change Log", section_style))

        headers = ["ID", "Category", "Location", "Change Type", "v1 Value", "v2 Value"]
        table_data = [[Paragraph(h, table_header_style) for h in headers]]

        for r in records:
            # Color coding change type cell
            ct = r.change_type
            ct_style = ParagraphStyle(
                'CTStyle',
                parent=table_cell_bold,
                textColor=colors.HexColor('#E53E3E') if ct == "Modified" else (colors.HexColor('#3182CE') if ct == "Added" else colors.HexColor('#38A169'))
            )
            table_data.append([
                Paragraph(r.region_id, table_cell_bold),
                Paragraph(r.category, table_cell_style),
                Paragraph(r.location_description, table_cell_style),
                Paragraph(ct, ct_style),
                Paragraph(r.v1_value or "—", table_cell_style),
                Paragraph(r.v2_value or "—", table_cell_style),
            ])

        log_table = Table(table_data, colWidths=[40, 80, 140, 70, 105, 105])

        # Style table
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])
        # Alternating row colors
        for idx in range(1, len(table_data)):
            bg = colors.HexColor('#F7FAFC') if idx % 2 == 0 else colors.white
            table_style.add('BACKGROUND', (0, idx), (-1, idx), bg)

        log_table.setStyle(table_style)
        story.append(log_table)

        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        return pdf_bytes
    finally:
        # Clean up all temporary files created for images
        for path in temp_files:
            try:
                os.remove(path)
            except OSError:
                pass

