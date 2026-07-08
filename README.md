# CAD Elevation Drawing Revision Comparator

Python application for comparing two architectural elevation drawings. It ingests PDF/JPG/PNG files, crops drawing borders, aligns revisions with ORB-first/SIFT-fallback homography, computes SSIM structural differences, classifies changed regions with deterministic geometry rules, optionally OCRs text-bearing regions, and produces a single self-contained HTML report.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Secrets

Set `GROQ_API_KEY` only when you want the final natural-language summary. The report still renders if the Groq call fails or the key is absent.

```powershell
$env:GROQ_API_KEY="your_api_key"
$env:GROQ_TEXT_MODEL="llama-3.3-70b-versatile"
```

`GROQ_TEXT_MODEL` is read from the environment and defaults to `llama-3.3-70b-versatile`. Before running production comparisons, verify current Groq model availability at [Groq supported models](https://console.groq.com/docs/models), because model IDs can change.

## Run Streamlit UI

```powershell
streamlit run app.py
```

The UI provides two uploaders, a Compare button, stage-by-stage progress, inline report rendering, and a downloadable HTML report.

## Optional FastAPI Backend

The FastAPI app is exposed as `main:api`.

```powershell
uvicorn main:api --reload
```

`POST /compare` accepts JSON with base64-encoded `drawing_a_base64`, `drawing_b_base64`, `filename_a`, and `filename_b`.

## Pipeline Stages

1. Ingestion via `ingestion.py`, with PDFs rendered at fixed 300 DPI.
2. Border crop via `preprocess.py`.
3. Alignment via `alignment.py`, using ORB first and SIFT fallback when match quality is poor.
4. SSIM diff via `diff.py`.
5. Thresholding and contour extraction via `diff.py`.
6. Geometry classification via `element_classification.py`.
7. OCR extraction via `ocr_extraction.py`, using PaddleOCR by default or EasyOCR via `CAD_COMPARE_OCR_ENGINE=easyocr`.
8. Compare and merge via `compare.py` and `merge.py`.
9. Visualization via `visualize.py`.
10. HTML report, table, thumbnails, and statistics via `report.py`.
11. One final Groq text-only call via `ai_summary.py`.
12. Streamlit and optional FastAPI orchestration via `app.py`.

All thresholds are centralized in `config.py`.
