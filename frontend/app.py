import streamlit as st
import requests
import json
import base64
import os

# Set page config
st.set_page_config(
    page_title="AI CAD Elevation Revision Comparator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .summary-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 4px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1E3A8A;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_content_html=True)

st.markdown('<div class="main-title">📐 AI-Assisted CAD Elevation Revision Comparator</div>', unsafe_content_html=True)
st.markdown('<div class="sub-title">Upload two versions (v1, v2) of an architectural elevation drawing to automatically align, detect element revisions, extract labels, and summarize changes.</div>', unsafe_content_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Pipeline Configuration")
ocr_engine = st.sidebar.selectbox(
    "OCR Engine",
    ["paddleocr", "easyocr"],
    help="PaddleOCR is primary (fast and highly accurate for drawing texts). EasyOCR is available as fallback."
)

backend_url = st.sidebar.text_input("FastAPI Backend URL", "http://127.0.0.1:8000")

st.sidebar.markdown("---")
st.sidebar.markdown("### Change Type Legend")
st.sidebar.markdown("🔴 **Modified**: Red bounding box")
st.sidebar.markdown("🔵 **Added**: Blue bounding box")
st.sidebar.markdown("🟢 **Removed**: Green bounding box on v1 position")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("Drawing Version 1 (v1)")
    file1 = st.file_uploader("Upload v1 PDF or Image", type=["pdf", "png", "jpg", "jpeg"], key="v1")
    
with col2:
    st.subheader("Drawing Version 2 (v2)")
    file2 = st.file_uploader("Upload v2 PDF or Image", type=["pdf", "png", "jpg", "jpeg"], key="v2")

if file1 and file2:
    if st.button("Compare Drawings", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0, text="Initializing comparison pipeline...")
        
        # We perform stages sequentially and update the progress indicator
        # 1. Ingestion
        progress_bar.progress(10, text="1. Ingesting and rendering drawings at 300 DPI...")
        # 2. Border crop
        progress_bar.progress(25, text="2. Detecting margins and cropping to outer border...")
        # 3. Alignment
        progress_bar.progress(40, text="3. Performing SIFT feature mapping and homography alignment...")
        # 4. Grid detection
        progress_bar.progress(55, text="4. Detecting facade grid rhythms (columns/floors)...")
        # 5. Element detection
        progress_bar.progress(70, text="5. Detecting building elements (windows, doors, pillars, roof segments)...")
        # 6. Diff and OCR
        progress_bar.progress(85, text="6. Computing SSIM, running regional multi-angle OCR, and comparing values...")
        # 7. Merge and summary
        progress_bar.progress(95, text="7. Generating final change log and requesting AI summary...")

        try:
            # Call backend FastAPI endpoint
            files = {
                "file1": (file1.name, file1.getvalue(), file1.type),
                "file2": (file2.name, file2.getvalue(), file2.type),
            }
            data = {
                "ocr_engine": ocr_engine
            }
            
            response = requests.post(f"{backend_url}/compare", files=files, data=data)
            
            progress_bar.progress(100, text="Comparison complete!")
            
            if response.status_code == 200:
                result = response.json()
                metadata = result["metadata"]
                ai_summary_text = result["ai_summary"]
                change_records = result["change_records"]
                pdf_report_path = result["pdf_report_path"]
                
                st.success("Revision Comparison completed successfully!")
                
                # Metadata row
                st.header("📊 Comparison Overview")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metadata['alignment_confidence'].upper()}</div>
                        <div class="metric-label">Alignment Confidence</div>
                    </div>
                    """, unsafe_content_html=True)
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metadata['scale_ratio']:.3f}</div>
                        <div class="metric-label">SIFT Scale Ratio</div>
                    </div>
                    """, unsafe_content_html=True)
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(change_records)}</div>
                        <div class="metric-label">Total Elements Analyzed</div>
                    </div>
                    """, unsafe_content_html=True)
                with m_col4:
                    changed_count = sum(1 for r in change_records if r["change_type"] != "Unchanged")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{changed_count}</div>
                        <div class="metric-label">Revisions Detected</div>
                    </div>
                    """, unsafe_content_html=True)
                
                st.markdown("---")
                
                # AI Narrative Summary
                st.header("📝 Executive Revision Narrative")
                st.markdown(f'<div class="summary-box">{ai_summary_text}</div>', unsafe_content_html=True)
                
                # Download Button for PDF Report
                try:
                    dl_response = requests.get(f"{backend_url}/download-report", params={"path": pdf_report_path})
                    if dl_response.status_code == 200:
                        st.download_button(
                            label="📥 Download Standalone PDF Report",
                            data=dl_response.content,
                            file_name="CAD_Revision_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Failed to load PDF download link: {e}")
                
                st.markdown("---")
                
                # Change Log Table
                st.header("📋 Detailed Change Log")
                
                # Filter down to actual changes
                changed_records = [r for r in change_records if r["change_type"] != "Unchanged"]
                
                if changed_records:
                    # Let's display it nicely
                    for rec in changed_records:
                        with st.container():
                            c_col1, c_col2, c_col3, c_col4 = st.columns([1, 2, 2, 3])
                            with c_col1:
                                badge = "🔴 MODIFIED" if rec["change_type"] == "Modified" else ("🔵 ADDED" if rec["change_type"] == "Added" else "🟢 REMOVED")
                                st.markdown(f"**{rec['region_id']}**")
                                st.markdown(f"Category: `{rec['element_type']}`")
                                st.markdown(badge)
                                if rec.get("low_confidence"):
                                    st.warning("⚠️ Low Confidence")
                            with c_col2:
                                st.markdown("**Version 1 (v1)**")
                                if rec["crop_v1"]:
                                    st.image(base64.b64decode(rec["crop_v1"]), use_column_width=False, width=150)
                                    st.caption(f"Value: {rec['v1_value']}")
                                else:
                                    st.write("N/A")
                            with c_col3:
                                st.markdown("**Version 2 (v2)**")
                                if rec["crop_v2"]:
                                    st.image(base64.b64decode(rec["crop_v2"]), use_column_width=False, width=150)
                                    st.caption(f"Value: {rec['v2_value']}")
                                else:
                                    st.write("N/A")
                            with c_col4:
                                st.markdown("**Analysis Metrics**")
                                st.write(f"SSIM Similarity: `{rec['ssim_score']:.3f}`")
                                st.write(f"Mean Pixel Difference: `{rec['pixel_diff']:.2f}`")
                                if rec["ocr_confidence"] > 0:
                                    st.write(f"OCR Confidence: `{rec['ocr_confidence']:.2f}`")
                            st.markdown("---")
                else:
                    st.info("No structural or label revisions detected. Both versions match perfectly.")
                    
            else:
                st.error(f"Error from FastAPI backend: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to backend or process comparison: {str(e)}")
