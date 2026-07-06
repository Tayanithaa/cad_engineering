# CAD Drawing Revision Comparator

An engineering-grade revision comparison engine for architectural CAD drawings (PDFs or raster images). 

The system operates like professional BIM / Autodesk revision tools, upgrading traditional pixel difference tracking to **semantic CAD object matching** with extreme precision, memory safeguards, and low false positive rates.

---

## Technical Pipeline

### 1. Ingestion & Scale Capping (`ingestion.py`)
- Decodes image uploads and renders vector PDFs at a fixed DPI.
- **Memory Safety Safeguard:** Pre-calculates scale constraints to cap dimensions at a maximum of `2000px`. This keeps color image footprints under ~12MB, preventing out-of-memory errors on large sheets.

### 2. Drawing Region Crop (`diff.py`)
- Automatically analyzes horizontal and vertical line projections to locate drawing borders, title blocks, revision tables, logos, and margins.
- Crops the comparison target to focus purely on the architectural floor plan.

### 3. Image Alignment (`alignment.py`)
- Uses SIFT feature detection, FLANN keypoint matching, and RANSAC estimation to calculate scale differences and perspective warp drawing revisions into a unified coordinate frame.

### 4. CAD Object Detection (`diff.py` & `ocr_extraction.py`)
- Extracts text elements using a single-pass full-sheet OCR scan (`extract_full_image_ocr`).
- Groups lines, contours, and OCR elements into high-level architectural categories:
  - **Walls:** Parallel double-line pairs with typical wall spacing (10–25px) and minimum structural lengths (80px), filtering out hatching lines.
  - **Doors:** Curved swing-arcs/frames.
  - **Windows:** Close parallel window frame lines.
  - **Columns:** Circular or square solid structural contours.
  - **Staircases:** Multi-step closely spaced parallel vectors.
  - **Dimensions & Text Labels:** Dimension labels and annotation notes.

### 5. Weighted Hungarian Object Matching (`diff.py`)
- Computes a global bipartite matching between drawings using the **Hungarian Assignment Algorithm** (`scipy.optimize.linear_sum_assignment`).
- Pairings are optimized based on a weighted 5-parameter similarity metric:
  - **40% Geometry** (dimension/label text values, length profiles)
  - **25% Position** (centroid distance)
  - **15% Orientation** (angle alignment)
  - **10% Shape** (bounding box aspect ratio)
  - **10% Topology** (bounding box Intersection-over-Union)
- **High-Density Fallback:** Gracefully falls back to a localized nearest-neighbor search if a category contains over 150 elements to prevent $O(N^3)$ computational bottlenecks.

### 6. False Positive Suppression & Revision Capping
- Ignores micro-translations under 15 pixels and dimensional variations under 15% (which represent normal anti-aliasing, rasterization, or DPI differences).
- Limits output visualizations to the top **25** most significant engineering changes, sorting by component type and severity of alteration.

### 7. Overlays & Reports (`report.py`)
- Generates clean overlays showing only changed components color-coded:
  - **Green:** Added
  - **Red:** Removed
  - **Yellow:** Modified
  - **Blue:** Shifted
- Computes an engineering change log table and change-density heatmap.
- Packages findings into a downloadable HTML or PDF report, complete with a single-call Llama-3 (Groq) AI design overview.

---

## How to Run

Initialize the Streamlit server locally:

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

*Note: Ensure your `GROQ_API_KEY` is specified in your `.env` file to support the AI summary feature.*
