"""
CAD Drawing Revision Comparator — Streamlit app.

Pipeline (classical CV/OCR, deterministic — NO AI except the final summary):
  Upload -> Align (SIFT/FLANN/RANSAC) -> Diff (SSIM) -> OCR extraction ->
  Rule-based comparison -> Change records -> Visualization + Stats ->
  Groq AI summary (single call) -> Downloadable HTML report.
"""

import streamlit as st
import cv2
import numpy as np

from ingestion import ingest_file, IngestionError
from alignment import estimate_alignment, AlignmentResult
from diff import (
    detect_differences, extract_vector_lines, compare_vector_lines, draw_vector_overlay,
    detect_drawing_boundary, extract_cad_objects, match_cad_objects, draw_cad_overlay
)
from ocr_extraction import extract_region_ocr, extract_full_image_ocr
from compare import compare_region, build_vector_change_records, build_object_change_records
from report import generate_html_report, generate_pdf_report, draw_annotated_overlay, build_statistics
from ai_summary import generate_ai_summary
from config import GROQ_API_KEY

st.set_page_config(page_title="CAD Drawing Revision Comparator", layout="wide")

st.title("CAD Drawing Revision Comparator")
st.caption(
    "Compares two CAD drawing revisions using classical computer vision and OCR. "
    "AI (Groq) is used exactly once, at the end, to write the summary paragraph."
)

if not GROQ_API_KEY:
    st.warning(
        "GROQ_API_KEY is not set. The pipeline will still run and produce the full "
        "report, but the AI summary step will show a fallback message.",
        icon="⚠️",
    )

col1, col2 = st.columns(2)
with col1:
    v1_file = st.file_uploader("Upload v1 (original)", type=["pdf", "jpg", "jpeg", "png"], key="v1")
with col2:
    v2_file = st.file_uploader("Upload v2 (revised)", type=["pdf", "jpg", "jpeg", "png"], key="v2")

min_area = st.sidebar.slider("Minimum change area (px²)", 20, 2000, 60, step=10)
st.sidebar.caption("Filters out tiny noise contours from the diff mask.")
align_images = st.sidebar.checkbox("Enable automatic alignment (SIFT/RANSAC)", value=True)
st.sidebar.caption("Disable this for digital vector PDFs that are already perfectly aligned to prevent alignment artifacts.")
comparison_mode = st.sidebar.selectbox("Comparison Mode", ["Standard (Regions)", "Vector (Line-by-Line)", "Engineering (Object-Level)"])
st.sidebar.caption("Engineering mode matches CAD objects (doors, walls, columns, etc.). Vector mode matches lines. Standard mode matches pixel groups.")

compare_clicked = st.button("Compare", type="primary", disabled=not (v1_file and v2_file))

if "report_html" not in st.session_state:
    st.session_state.report_html = None

if compare_clicked and v1_file and v2_file:
    st.session_state.report_html = None
    progress = st.progress(0, text="Starting…")

    try:
        progress.progress(5, text="Ingesting files…")
        v1_img = ingest_file(v1_file.getvalue(), v1_file.name)
        v2_img = ingest_file(v2_file.getvalue(), v2_file.name)

        if comparison_mode == "Engineering (Object-Level)":
            progress.progress(10, text="Detecting drawing boundary to ignore margins/title blocks…")
            ymin1, xmin1, ymax1, xmax1 = detect_drawing_boundary(v1_img.gray)
            ymin2, xmin2, ymax2, xmax2 = detect_drawing_boundary(v2_img.gray)

            v1_gray_in = v1_img.gray[ymin1:ymax1, xmin1:xmax1]
            v1_color_in = v1_img.color_bgr[ymin1:ymax1, xmin1:xmax1]
            v2_gray_in = v2_img.gray[ymin2:ymax2, xmin2:xmax2]
            v2_color_in = v2_img.color_bgr[ymin2:ymax2, xmin2:xmax2]
        else:
            v1_gray_in = v1_img.gray
            v1_color_in = v1_img.color_bgr
            v2_gray_in = v2_img.gray
            v2_color_in = v2_img.color_bgr

        v1_img.gray = v1_gray_in
        v1_img.color_bgr = v1_color_in

        if align_images:
            progress.progress(20, text="Estimating scale + alignment (SIFT/FLANN/RANSAC)…")
            alignment = estimate_alignment(v1_gray_in, v2_gray_in, v2_color_in)
        else:
            progress.progress(20, text="Bypassing alignment (direct comparison mode)…")
            h1, w1 = v1_gray_in.shape[:2]
            alignment = AlignmentResult(
                aligned_v2_gray=cv2.resize(v2_gray_in, (w1, h1)),
                aligned_v2_color=cv2.resize(v2_color_in, (w1, h1)),
                homography=None,
                scale_ratio=1.0,
                good_match_count=0,
                low_confidence=False,
                message="Alignment bypassed (direct comparison mode)."
            )
        st.info(alignment.message)

        if comparison_mode == "Engineering (Object-Level)":
            progress.progress(30, text="Extracting text labels via EasyOCR (single pass)…")
            ocr_v1 = extract_full_image_ocr(v1_img.color_bgr)
            ocr_v2 = extract_full_image_ocr(alignment.aligned_v2_color)

            progress.progress(50, text="Detecting architectural CAD objects (Doors, Windows, Walls, Columns, Dimensions)…")
            objs_v1 = extract_cad_objects(v1_img.gray, v1_img.color_bgr, ocr_v1)
            objs_v2 = extract_cad_objects(alignment.aligned_v2_gray, alignment.aligned_v2_color, ocr_v2)

            progress.progress(70, text=f"Matching {len(objs_v1)} objects in v1 against {len(objs_v2)} objects in v2…")
            matched_results = match_cad_objects(objs_v1, objs_v2)

            progress.progress(85, text="Generating smart visual overlays…")
            overlay_v1 = draw_cad_overlay(v1_img.color_bgr, matched_results)
            overlay_v2 = draw_cad_overlay(alignment.aligned_v2_color, matched_results)

            progress.progress(90, text="Compiling object-level Change Records…")
            img_h, img_w = v1_img.gray.shape[:2]
            records = build_object_change_records(matched_results, img_w, img_h)
            stats = build_statistics(records)

            # Generate a change-density heatmap based on modified objects
            heatmap_mask = np.zeros_like(v1_img.gray)
            for o1, o2, status, desc in matched_results:
                if status == "UNCHANGED":
                    continue
                for obj in (o1, o2):
                    if obj is not None:
                        x, y, w_obj, h_obj = obj.bbox
                        cv2.rectangle(heatmap_mask, (x, y), (x + w_obj, y + h_obj), 255, -1)

            if heatmap_mask.max() > 0:
                # Apply a Gaussian blur to create a density smoothing effect
                heatmap_blur = cv2.GaussianBlur(heatmap_mask, (51, 51), 0)
                cv2.normalize(heatmap_blur, heatmap_blur, 0, 255, cv2.NORM_MINMAX)
                heatmap = cv2.applyColorMap(heatmap_blur, cv2.COLORMAP_JET)
                diff_mask = (heatmap_mask > 0).astype(np.uint8) * 255
            else:
                heatmap = np.zeros_like(v1_img.color_bgr)
                diff_mask = np.zeros_like(v1_img.gray)

            progress.progress(93, text="Generating AI summary (single Groq call)…")
            change_records_dicts = [r.to_dict() for r in records]
            ai_summary_text, ai_ok = generate_ai_summary(change_records_dicts, stats)

            progress.progress(97, text="Assembling report…")
            report_html = generate_html_report(
                v1_filename=v1_file.name,
                v2_filename=v2_file.name,
                v1_color=v1_img.color_bgr,
                v2_aligned_color=alignment.aligned_v2_color,
                annotated_overlay=overlay_v2,
                heatmap=heatmap,
                scale_ratio=alignment.scale_ratio,
                low_confidence=alignment.low_confidence,
                alignment_message=alignment.message,
                records=records,
                percent_area_changed=0.0,
                ai_summary=ai_summary_text,
                ai_summary_succeeded=ai_ok,
                region_crops={},
            )
            report_pdf = generate_pdf_report(
                v1_filename=v1_file.name,
                v2_filename=v2_file.name,
                v1_color=v1_img.color_bgr,
                v2_aligned_color=alignment.aligned_v2_color,
                annotated_overlay=overlay_v2,
                heatmap=heatmap,
                scale_ratio=alignment.scale_ratio,
                low_confidence=alignment.low_confidence,
                alignment_message=alignment.message,
                records=records,
                percent_area_changed=0.0,
                ai_summary=ai_summary_text,
                ai_summary_succeeded=ai_ok,
            )
            st.session_state.report_html = report_html
            st.session_state.report_pdf = report_pdf
            st.session_state.report_records = records
            st.session_state.report_stats = stats
            st.session_state.report_overlay_v1 = overlay_v1
            st.session_state.report_overlay_v2 = overlay_v2
            st.session_state.report_heatmap = heatmap
            st.session_state.report_mask = diff_mask
            st.session_state.ai_summary_text = ai_summary_text
            st.session_state.ai_ok = ai_ok

            progress.progress(100, text="Done.")
        elif comparison_mode == "Vector (Line-by-Line)":
            progress.progress(40, text="Extracting engineering line segments (LSD)…")
            lines_v1 = extract_vector_lines(v1_img.gray)
            lines_v2 = extract_vector_lines(alignment.aligned_v2_gray)

            progress.progress(60, text=f"Matching {len(lines_v1)} lines against {len(lines_v2)} lines…")
            matched, added, removed, modified, shifted = compare_vector_lines(lines_v1, lines_v2)

            progress.progress(80, text="Drawing vector overlays…")
            VECTOR_COLORS = {
                "Matched": (0, 180, 0),     # Dark Green
                "Added": (255, 0, 0),       # Blue
                "Removed": (0, 0, 255),     # Red
                "Modified": (0, 140, 255),  # Orange
                "Shifted": (200, 0, 200)    # Purple
            }
            # For v1 overlay, draw Removed, Modified, Shifted
            overlay_v1 = draw_vector_overlay(v1_img.color_bgr, removed + modified + shifted, VECTOR_COLORS)
            # For v2 overlay, draw Added, Modified, Shifted
            overlay_v2 = draw_vector_overlay(alignment.aligned_v2_color, added + modified + shifted, VECTOR_COLORS)

            progress.progress(85, text="Compiling line-level Change Records…")
            img_h, img_w = v1_img.gray.shape[:2]
            records = build_vector_change_records(matched, added, removed, modified, shifted, img_w, img_h)
            stats = build_statistics(records)

            heatmap = np.zeros_like(v1_img.color_bgr)
            diff_mask = np.zeros_like(v1_img.gray)

            progress.progress(92, text="Generating AI summary (single Groq call)…")
            change_records_dicts = [r.to_dict() for r in records]
            ai_summary_text, ai_ok = generate_ai_summary(change_records_dicts, stats)

            progress.progress(97, text="Assembling report…")
            report_html = generate_html_report(
                v1_filename=v1_file.name,
                v2_filename=v2_file.name,
                v1_color=v1_img.color_bgr,
                v2_aligned_color=alignment.aligned_v2_color,
                annotated_overlay=overlay_v2,
                heatmap=heatmap,
                scale_ratio=alignment.scale_ratio,
                low_confidence=alignment.low_confidence,
                alignment_message=alignment.message,
                records=records,
                percent_area_changed=0.0,
                ai_summary=ai_summary_text,
                ai_summary_succeeded=ai_ok,
                region_crops={},
            )
            report_pdf = generate_pdf_report(
                v1_filename=v1_file.name,
                v2_filename=v2_file.name,
                v1_color=v1_img.color_bgr,
                v2_aligned_color=alignment.aligned_v2_color,
                annotated_overlay=overlay_v2,
                heatmap=heatmap,
                scale_ratio=alignment.scale_ratio,
                low_confidence=alignment.low_confidence,
                alignment_message=alignment.message,
                records=records,
                percent_area_changed=0.0,
                ai_summary=ai_summary_text,
                ai_summary_succeeded=ai_ok,
            )
            st.session_state.report_html = report_html
            st.session_state.report_pdf = report_pdf
            st.session_state.report_records = records
            st.session_state.report_stats = stats
            st.session_state.report_overlay_v1 = overlay_v1
            st.session_state.report_overlay_v2 = overlay_v2
            st.session_state.report_heatmap = heatmap
            st.session_state.report_mask = diff_mask
            st.session_state.ai_summary_text = ai_summary_text
            st.session_state.ai_ok = ai_ok

            progress.progress(100, text="Done.")
        else:
            progress.progress(35, text="Detecting differences (SSIM + contours)…")
            diff_result = detect_differences(v1_img.gray, alignment.aligned_v2_gray, min_contour_area=min_area)

            if not diff_result.boxes:
                progress.progress(100, text="Done — no differences detected.")
                st.success("No significant differences were detected between v1 and v2.")
            else:
                progress.progress(45, text=f"Running OCR on {len(diff_result.boxes)} region(s)…")
                crops_list = []
                for i, box in enumerate(diff_result.boxes):
                    crops_list.append(extract_region_ocr(v1_img.color_bgr, alignment.aligned_v2_color, box))
                    pct = 45 + int(30 * (i + 1) / len(diff_result.boxes))
                    progress.progress(pct, text=f"OCR extraction {i + 1}/{len(diff_result.boxes)} regions…")

                progress.progress(78, text="Comparing regions (rule-based, no AI)…")
                img_h, img_w = v1_img.gray.shape[:2]

                all_pairs = []
                for box, crops in zip(diff_result.boxes, crops_list):
                    rec = compare_region(box, crops, img_w, img_h)
                    if rec.change_type != "Unchanged":
                        all_pairs.append((rec, crops))

                for i, (rec, _crops) in enumerate(all_pairs, start=1):
                    rec.region_id = f"R-{i:03d}"
                records = [rec for rec, _ in all_pairs]
                region_crops = {rec.region_id: (crops.crop_v1, crops.crop_v2) for rec, crops in all_pairs}

                progress.progress(85, text="Building visualizations…")
                overlay_v1 = draw_annotated_overlay(v1_img.color_bgr, records)
                overlay_v2 = draw_annotated_overlay(alignment.aligned_v2_color, records)
                stats = build_statistics(records)

                progress.progress(92, text="Generating AI summary (single Groq call)…")
                change_records_dicts = [r.to_dict() for r in records]
                ai_summary_text, ai_ok = generate_ai_summary(change_records_dicts, stats)

                progress.progress(97, text="Assembling report…")
                report_html = generate_html_report(
                    v1_filename=v1_file.name,
                    v2_filename=v2_file.name,
                    v1_color=v1_img.color_bgr,
                    v2_aligned_color=alignment.aligned_v2_color,
                    annotated_overlay=overlay_v2,
                    heatmap=diff_result.heatmap,
                    scale_ratio=alignment.scale_ratio,
                    low_confidence=alignment.low_confidence,
                    alignment_message=alignment.message,
                    records=records,
                    percent_area_changed=diff_result.percent_area_changed,
                    ai_summary=ai_summary_text,
                    ai_summary_succeeded=ai_ok,
                    region_crops=region_crops,
                )
                report_pdf = generate_pdf_report(
                    v1_filename=v1_file.name,
                    v2_filename=v2_file.name,
                    v1_color=v1_img.color_bgr,
                    v2_aligned_color=alignment.aligned_v2_color,
                    annotated_overlay=overlay_v2,
                    heatmap=diff_result.heatmap,
                    scale_ratio=alignment.scale_ratio,
                    low_confidence=alignment.low_confidence,
                    alignment_message=alignment.message,
                    records=records,
                    percent_area_changed=diff_result.percent_area_changed,
                    ai_summary=ai_summary_text,
                    ai_summary_succeeded=ai_ok,
                )
                st.session_state.report_html = report_html
                st.session_state.report_pdf = report_pdf
                st.session_state.report_records = records
                st.session_state.report_stats = stats
                st.session_state.report_overlay_v1 = overlay_v1
                st.session_state.report_overlay_v2 = overlay_v2
                st.session_state.report_heatmap = diff_result.heatmap
                st.session_state.report_mask = diff_result.diff_mask
                st.session_state.ai_summary_text = ai_summary_text
                st.session_state.ai_ok = ai_ok

                progress.progress(100, text="Done.")

    except IngestionError as exc:
        progress.empty()
        st.error(f"Upload error: {exc}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        progress.empty()
        st.error(f"Unexpected error during comparison: {exc}")

if st.session_state.report_html:
    st.divider()
    st.subheader("Comparison Result")

    if not st.session_state.report_records:
        st.success("No engineering changes detected.")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(st.session_state.report_overlay_v1[:, :, ::-1], caption="v1 (reference) - Changes Highlighted", use_container_width=True)
    with img_col2:
        st.image(st.session_state.report_overlay_v2[:, :, ::-1], caption="v2 (aligned) - Changes Highlighted", use_container_width=True)

    show_heatmap = st.toggle("Show change-density heatmap", value=False)
    if show_heatmap:
        st.image(st.session_state.report_heatmap[:, :, ::-1], caption="Heatmap", use_container_width=True)

    show_binary = st.toggle("Show raw difference mask (debug)", value=False)
    if show_binary:
        st.image(st.session_state.report_mask, caption="Raw Difference Mask", use_container_width=True)

    stats = st.session_state.report_stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Regions", stats["total_regions"])
    m2.metric("By Change Type", ", ".join(f"{k}: {v}" for k, v in stats["by_change_type"].items()) or "—")
    m3.metric("By Category", ", ".join(f"{k}: {v}" for k, v in stats["by_category"].items()) or "—")

    st.subheader("Change Log")
    st.dataframe(
        [
            {
                "Region ID": r.region_id,
                "Category": r.category,
                "Location": r.location_description,
                "Change Type": r.change_type,
                "v1 Value": r.v1_value,
                "v2 Value": r.v2_value,
            }
            for r in st.session_state.report_records
        ],
        use_container_width=True,
    )

    st.subheader("AI Summary")
    if not st.session_state.ai_ok:
        st.caption("AI summary unavailable — see change log table above.")
    st.write(st.session_state.ai_summary_text)

    st.download_button(
        "Download PDF Report",
        data=st.session_state.report_pdf,
        file_name="cad_comparison_report.pdf",
        mime="application/pdf",
        type="primary",
    )
