"""
Stage 10: AI Summary — the ONLY Groq API call in the whole pipeline.

Takes the final structured change-record JSON and asks a single Groq
text-only chat completion to write a natural-language summary paragraph.
Detection, alignment, and extraction upstream are fully classical/
deterministic; Groq is used here exclusively for prose generation.
"""

from __future__ import annotations

import json

from config import GROQ_API_KEY, GROQ_TEXT_MODEL

FALLBACK_NOTE = "AI summary unavailable — see change log table above."


def _build_prompt(change_records_json: str, stats: dict) -> str:
    return (
        "Given this structured list of detected changes between two CAD drawing "
        "revisions, write a concise, clear paragraph summarizing the overall "
        "comparison result, the major changes, their approximate locations, and what was modified. "
        "Write it in a friendly, easy-to-understand way that is clear even to a non-engineer or layman. "
        "Avoid overly dense technical jargon, and focus on describing the real-world impact of the changes "
        "(e.g., 'a dimension was modified,' 'a new text label was added near the entrance'). "
        "Base the summary strictly on the provided data — do not invent details not present in the JSON.\n\n"
        f"Summary statistics: {json.dumps(stats)}\n\n"
        f"Change records JSON:\n{change_records_json}"
    )


def generate_ai_summary(change_records: list[dict], stats: dict) -> tuple[str, bool]:
    """
    Generate a natural-language summary paragraph from the final change log.

    Returns (summary_text, succeeded). On any failure (missing key, API error,
    rate limit, etc.) returns the fallback note and succeeded=False rather than
    raising — the rest of the report must still render.
    """
    if not GROQ_API_KEY:
        return (
            f"{FALLBACK_NOTE} (No GROQ_API_KEY configured.)",
            False,
        )

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        change_records_json = json.dumps(change_records, indent=2)
        prompt = _build_prompt(change_records_json, stats)

        response = client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a writer summarizing drawing comparisons. "
                        "Your goal is to explain changes clearly, using simple language "
                        "so that non-technical users and clients can easily understand what has changed."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return f"{FALLBACK_NOTE} (Empty response from AI model.)", False
        return text, True
    except Exception as exc:
        return f"{FALLBACK_NOTE} (Error: {exc})", False
