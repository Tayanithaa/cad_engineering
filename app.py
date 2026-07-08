from __future__ import annotations

import base64
from typing import Callable


from cad_engineering.ai_summary import generate_ai_summary
from cad_engineering.alignment import align_drawing_b_to_a
from cad_engineering.diff import run_structural_diff
from cad_engineering.element_classification import classify_regions
from cad_engineering.ingestion import ingest_pair
from cad_engineering.merge import merge_change_records
from cad_engineering.ocr_extraction import extract_ocr_for_regions
from cad_engineering.preprocess import crop_pair_to_borders
from cad_engineering.report import build_pdf_report
from cad_engineering.visualize import build_visuals


ProgressCallback = Callable[[str, str], None]


def _notify(callback: ProgressCallback | None, stage: str, detail: str) -> None:
    print(f"[{stage}] {detail}")
    if callback:
        callback(stage, detail)


def compare_drawings(
    drawing_a_bytes: bytes,
    drawing_b_bytes: bytes,
    filename_a: str,
    filename_b: str,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    _notify(progress_callback, "Ingestion", "Loading and validating inputs")
    ingested_a, ingested_b = ingest_pair(drawing_a_bytes, drawing_b_bytes, filename_a, filename_b)

    _notify(progress_callback, "Border Crop", "Detecting outer drawing borders")
    crop_a, crop_b = crop_pair_to_borders(ingested_a.image, ingested_b.image)

    _notify(progress_callback, "Alignment", "Running ORB alignment with SIFT fallback if needed")
    alignment = align_drawing_b_to_a(crop_a.image, crop_b.image)
    _notify(
        progress_callback,
        "Alignment",
        f"Used {alignment.metadata['alignment_method']} with {alignment.metadata['good_matches']} matches",
    )

    _notify(progress_callback, "SSIM Diff", "Computing structural similarity map")
    diff_result = run_structural_diff(alignment.image_a, alignment.aligned_b)

    _notify(progress_callback, "Contour Extraction", f"Found {len(diff_result.regions)} changed regions")

    _notify(progress_callback, "Element Classification", "Classifying changed regions by geometry")
    classifications = classify_regions(alignment.aligned_b, diff_result.regions)

    _notify(progress_callback, "OCR Extraction", "Checking changed regions for text")
    ocr_results = extract_ocr_for_regions(alignment.image_a, alignment.aligned_b, diff_result.regions)

    _notify(progress_callback, "Compare/Merge", "Building structured change records")
    records = merge_change_records(
        alignment.image_a,
        alignment.aligned_b,
        diff_result.diff_map,
        classifications,
        ocr_results,
    )

    run_metadata = {
        "filenames": {"drawing_a": filename_a, "drawing_b": filename_b},
        "ingestion": {"drawing_a": ingested_a.metadata, "drawing_b": ingested_b.metadata},
        "border_crop": {"drawing_a": crop_a.metadata, "drawing_b": crop_b.metadata},
        "alignment": alignment.metadata,
        "diff": diff_result.metadata,
    }

    _notify(progress_callback, "Visualization", "Rendering overlays and heatmap")
    visuals = build_visuals(alignment.image_a, alignment.aligned_b, diff_result.diff_map, records)

    _notify(progress_callback, "AI Summary", "Generating final text-only Groq summary")
    ai_summary = generate_ai_summary(records, run_metadata)

    _notify(progress_callback, "Report Ready", "Assembling PDF report")
    pdf_report = build_pdf_report(
        alignment.image_a,
        alignment.aligned_b,
        diff_result.diff_map,
        records,
        visuals,
        run_metadata,
        ai_summary,
    )
    return {
        "records": records,
        "metadata": run_metadata,
        "ai_summary": ai_summary,
        "pdf_report": pdf_report,
        "visuals": visuals,
    }


def streamlit_main() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="CAD Elevation Revision Comparator", layout="wide")
    st.title("CAD Elevation Revision Comparator")

    col_a, col_b = st.columns(2)
    with col_a:
        drawing_a = st.file_uploader("Drawing A (original)", type=["pdf", "png", "jpg", "jpeg"])
    with col_b:
        drawing_b = st.file_uploader("Drawing B (revised)", type=["pdf", "png", "jpg", "jpeg"])

    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    if st.button("Compare", type="primary", disabled=not (drawing_a and drawing_b)):
        progress = st.progress(0)
        status = st.empty()
        stages = [
            "Ingestion",
            "Border Crop",
            "Alignment",
            "SSIM Diff",
            "Contour Extraction",
            "Element Classification",
            "OCR Extraction",
            "Compare/Merge",
            "Visualization",
            "AI Summary",
            "Report Ready",
        ]
        stage_index = {stage: index for index, stage in enumerate(stages, start=1)}

        def on_progress(stage: str, detail: str) -> None:
            value = stage_index.get(stage, 1) / len(stages)
            progress.progress(min(value, 1.0))
            status.info(f"{stage}: {detail}")

        try:
            st.session_state.comparison_result = compare_drawings(
                drawing_a.getvalue(),
                drawing_b.getvalue(),
                drawing_a.name,
                drawing_b.name,
                on_progress,
            )
            progress.progress(1.0)
            status.success("Report ready")
        except Exception as exc:
            st.session_state.comparison_result = None
            status.error(str(exc))

    result = st.session_state.comparison_result
    if result:
        st.subheader("AI Revision Summary Report")
        st.info(result["ai_summary"])

        st.subheader("Comparison Visuals")
        tab1, tab2, tab3 = st.tabs(["Annotated Bounding Boxes", "Side-by-Side View", "SSIM Structural Heatmap"])
        with tab1:
            st.image(result["visuals"]["overlay"], caption="Green=Added | Red=Removed | Yellow=Modified | Purple=Possible Change", use_container_width=True)
        with tab2:
            st.image(result["visuals"]["side_by_side"], caption="Original (Left) vs Revised (Right)", use_container_width=True)
        with tab3:
            st.image(result["visuals"]["heatmap"], caption="SSIM Heatmap (Darker = greater change)", use_container_width=True)

        st.subheader(f"Change Log ({len(result['records'])} revisions detected)")
        import pandas as pd
        if result["records"]:
            df = pd.DataFrame([
                {
                    "Region ID": r["region_id"],
                    "Category": r["element_type_or_category"],
                    "Location": r["location_description"],
                    "Change Type": r["change_type"],
                    "Value A": r["value_a"],
                    "Value B": r["value_b"],
                } for r in result["records"]
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No revisions detected.")

        st.download_button(
            "Download PDF Report",
            data=result["pdf_report"],
            file_name="cad_revision_comparison_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    streamlit_main()
