from __future__ import annotations

import json
import os
import time

from cad_engineering.config import AI_RETRY_BASE_SECONDS, GROQ_TEMPERATURE, GROQ_TEXT_MODEL, MAX_AI_RETRIES


SYSTEM_PROMPT = (
    "Base the summary strictly on the provided JSON; never invent or assume details not present in the data. "
    "Describe any record with low_confidence=true as a possible change. "
    "Explain what was added, removed, or changed on the building drawing in very simple, friendly, and easy-to-understand language, "
    "as if explaining it to a 10-year-old child. Avoid technical terminology and jargon (e.g. do not discuss matrix calculations or homography). "
    "Keep it fun and simple. Output 1-2 concise paragraphs, no markdown, no bullet points, and no preamble."
)


def build_summary_payload(change_records: list[dict], run_metadata: dict) -> dict:
    return {"run_metadata": run_metadata, "change_records": change_records}


def generate_ai_summary(change_records: list[dict], run_metadata: dict) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "AI summary unavailable — see change log table above"

    payload = build_summary_payload(change_records, run_metadata)
    serialized = json.dumps(payload, indent=2, default=str)
    last_error: Exception | None = None

    for attempt in range(1, MAX_AI_RETRIES + 1):
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=GROQ_TEXT_MODEL,
                temperature=GROQ_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": serialized},
                ],
            )
            summary = completion.choices[0].message.content
            print("[Stage 11] Groq AI summary generated.")
            return (summary or "").strip() or "AI summary unavailable — see change log table above"
        except Exception as exc:
            last_error = exc
            if attempt < MAX_AI_RETRIES:
                time.sleep(AI_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))

    print(f"[Stage 11] Groq AI summary unavailable after retries: {last_error}")
    return "AI summary unavailable — see change log table above"
