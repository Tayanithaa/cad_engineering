import config

def compare_labels_and_values(val1_info, val2_info):
    """
    Compares two values/labels.
    val1_info, val2_info are dictionaries with:
        {"text": text, "confidence": conf, "parsed_val": val, "unit": unit}
    Returns:
        change_type: "Unchanged", "Modified", "Added", "Removed"
        low_confidence: bool
    """
    t1 = val1_info.get("text", "").strip()
    t2 = val2_info.get("text", "").strip()
    
    c1 = val1_info.get("confidence", 1.0)
    c2 = val2_info.get("confidence", 1.0)
    
    # If confidence is below cutoff, set low_confidence flag
    low_confidence = (c1 < config.OCR_CONFIDENCE_THRESHOLD) or (c2 < config.OCR_CONFIDENCE_THRESHOLD)
    
    # If both empty
    if not t1 and not t2:
        return "Unchanged", False
        
    # If only v2 exists
    if not t1 and t2:
        return "Added", low_confidence
        
    # If only v1 exists
    if t1 and not t2:
        return "Removed", low_confidence
        
    # Both exist: Compare strings
    # Normalize strings (lowercase and replace multiple whitespaces)
    norm1 = " ".join(t1.lower().split())
    norm2 = " ".join(t2.lower().split())
    
    if norm1 == norm2:
        return "Unchanged", low_confidence
        
    # Also check parsed numeric values
    pv1 = val1_info.get("parsed_val")
    pv2 = val2_info.get("parsed_val")
    u1 = val1_info.get("unit")
    u2 = val2_info.get("unit")
    
    if pv1 is not None and pv2 is not None and u1 == u2:
        # Check numerical equality
        if abs(pv1 - pv2) < 1e-4:
            return "Unchanged", low_confidence
            
    return "Modified", low_confidence
