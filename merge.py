from __future__ import annotations

from collections import Counter

from cad_engineering.compare import derive_geometric_change_type, derive_text_change_type


def location_description(bbox: list[int], image_shape: tuple[int, int, int] | tuple[int, int]) -> str:
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]
    cx = x + w / 2
    cy = y + h / 2
    col = "left" if cx < img_w / 3 else "center" if cx < 2 * img_w / 3 else "right"
    row = "upper" if cy < img_h / 3 else "middle" if cy < 2 * img_h / 3 else "lower"
    return f"{row}-{col}"


def _display_value(ocr_result: dict, suffix: str) -> str | float | None:
    parsed = ocr_result.get(f"parsed_value_{suffix}")
    unit = ocr_result.get("unit")
    if parsed is not None:
        return f"{parsed:g} {unit}".strip() if unit else parsed
    return ocr_result.get(f"raw_text_{suffix}") or None


def merge_change_records(
    image_a,
    aligned_b,
    diff_map,
    geometric_classifications: list[dict],
    ocr_results: list[dict],
) -> list[dict]:
    ocr_by_id = {item["region_id"]: item for item in ocr_results}
    records = []
    for geom in geometric_classifications:
        bbox = geom["bbox"]
        ocr = ocr_by_id.get(geom["region_id"], {})
        geometric_type, ssim = derive_geometric_change_type(image_a, aligned_b, diff_map, bbox)
        text_type = derive_text_change_type(ocr) if ocr else "Unchanged"
        has_text = bool(ocr.get("likely_text"))

        if has_text and text_type != "Unchanged":
            change_type = text_type
            category = "Text/Note"
            if geom["element_type"] != "Geometric change (unclassified)":
                category = f"{geom['element_type']} + Text/Note"
        else:
            change_type = geometric_type
            category = geom["element_type"]

        low_confidence = bool(ocr.get("low_confidence", False))
        if low_confidence and has_text:
            change_type = "Possible change"

        # 1. Skip unchanged records entirely to keep report focused on actual revisions
        if change_type == "Unchanged":
            continue

        # 2. Skip unclassified geometric changes under 5000px area (to filter out line noise while retaining doors/windows/pillars)
        if category == "Geometric change (unclassified)" and (bbox[2] * bbox[3]) < 5000:
            continue

        records.append(
            {
                "region_id": geom["region_id"],
                "bbox": bbox,
                "element_type_or_category": category,
                "location_description": location_description(bbox, image_a.shape),
                "change_type": change_type,
                "value_a": _display_value(ocr, "a") if has_text else None,
                "value_b": _display_value(ocr, "b") if has_text else None,
                "ssim_score": ssim,
                "ocr_confidence": ocr.get("ocr_confidence") if has_text else None,
                "low_confidence": low_confidence,
                "classification_confidence": geom.get("classification_confidence"),
                "classification_reason": geom.get("classification_reason"),
            }
        )
    # Re-index remaining region IDs to be sequential (R001, R002, etc.)
    for idx, r in enumerate(records, 1):
        r["region_id"] = f"R{idx:03d}"
        
    print(f"[Stage 7] Merged {len(records)} unified change records.")
    return records


def summarize_records(records: list[dict], changed_area_percent: float) -> dict:
    counts = Counter(record["change_type"] for record in records)
    categories = Counter(record["element_type_or_category"] for record in records)
    return {
        "total_regions_compared": len(records),
        "counts": {
            "Added": counts.get("Added", 0),
            "Removed": counts.get("Removed", 0),
            "Modified": counts.get("Modified", 0),
            "Unchanged": counts.get("Unchanged", 0),
            "Possible change": counts.get("Possible change", 0),
        },
        "changed_area_percent": changed_area_percent,
        "category_breakdown": dict(categories),
    }
