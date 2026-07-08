from __future__ import annotations

import base64
from fastapi import FastAPI
from pydantic import BaseModel

from app import compare_drawings

api = FastAPI(title="CAD Elevation Revision Comparator")


class ComparePayload(BaseModel):
    drawing_a_base64: str
    drawing_b_base64: str
    filename_a: str = "drawing_a.png"
    filename_b: str = "drawing_b.png"


@api.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api.post("/compare")
def compare_endpoint(payload: ComparePayload) -> dict:
    drawing_a = base64.b64decode(payload.drawing_a_base64)
    drawing_b = base64.b64decode(payload.drawing_b_base64)
    result = compare_drawings(drawing_a, drawing_b, payload.filename_a, payload.filename_b)
    # Encode binary PDF bytes as base64 for JSON serialization
    pdf_b64 = base64.b64encode(result["pdf_report"]).decode("ascii")
    return {
        "records": result["records"],
        "metadata": result["metadata"],
        "ai_summary": result["ai_summary"],
        "pdf_report_base64": pdf_b64,
    }
