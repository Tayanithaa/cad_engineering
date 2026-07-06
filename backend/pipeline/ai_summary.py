import os
import json
import time
import config

def generate_revision_summary(change_records, metadata):
    """
    Sends the change records and metadata to Groq API to generate a narrative summary.
    Enforces rules via system prompt.
    """
    api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "AI summary unavailable — see change log table above (API key not set)."

    # Filter out Unchanged records for the LLM prompt to minimize tokens
    changed_records = [
        {
            "region_id": r["region_id"],
            "element_type": r["element_type"],
            "change_type": r["change_type"],
            "v1_value": r["v1_value"],
            "v2_value": r["v2_value"],
            "low_confidence": r.get("low_confidence", False)
        }
        for r in change_records if r["change_type"] != "Unchanged"
    ]
    
    payload = {
        "metadata": {
            "scale_ratio": metadata.get("scale_ratio", 1.0),
            "alignment_confidence": metadata.get("alignment_confidence", "high"),
            "status_message": metadata.get("status_message", ""),
            "total_regions_compared": len(change_records),
            "changed_regions_count": len(changed_records)
        },
        "changes": changed_records
    }
    
    # Check if we should use Groq Python client
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Could not import groq client: {e}")
        return "AI summary unavailable — see change log table above (Groq client not installed)."
        
    model = config.GROQ_TEXT_MODEL
    
    system_prompt = (
        "You are an expert architectural draftsperson and assistant. Your task is to write a natural language "
        "summary of changes between two drawing versions (v1 and v2) based ONLY on the provided JSON data.\n"
        "Rules:\n"
        "1. Base your output strictly on the provided JSON data. Do not invent details.\n"
        "2. Describe low_confidence records as 'possible changes' needing manual verification.\n"
        "3. Group related changes in the narrative (e.g. windows, doors, text/labels).\n"
        "4. Mention drawing scale mismatch or low alignment confidence if present in metadata.\n"
        "5. Output must be exactly 1-2 concise paragraphs.\n"
        "6. Do NOT use markdown, lists, bullet points, headers, or bold text. Plain text only."
    )
    
    user_prompt = f"JSON Change Log and Metadata:\n{json.dumps(payload, indent=2)}"
    
    # Retry with exponential backoff
    max_retries = 3
    backoff = 1.0
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model,
                temperature=0.2,
                max_tokens=500
            )
            summary = chat_completion.choices[0].message.content
            return summary.strip()
        except Exception as e:
            print(f"Groq API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                return f"AI summary unavailable — see change log table above (API Error: {str(e)})."
                
    return "AI summary unavailable — see change log table above."
