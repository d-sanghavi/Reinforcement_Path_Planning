"""
pdf_to_grid.py
──────────────
Local PDF floor plan → DXF converter.

Zero paid APIs. Zero cloud services. Fully local.

Conversion pipeline:
  1. pypdfium2 (pure-Python, embedded PDF renderer) → RGB bitmap
     OR
     pdf2image + Poppler CLI (fallback, requires local Poppler install)
  2. OpenCV: grayscale → adaptive threshold → Canny edge detection
  3. HoughLinesP: extract dominant line segments (walls, partitions)
  4. ezdxf: write line segments as DXF R2013 LINE entities

The resulting DXF is a synthetic reconstruction — geometric fidelity is
high for clean CAD-exported PDFs, and good-enough for rasterized scans.

Usage:
    from cad_ingestion.pdf_to_grid import pdf_to_dxf
    dxf_path = pdf_to_dxf("floor_plan.pdf", output_dir="./tmp")
"""

import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import ezdxf
import numpy as np

logger = logging.getLogger(__name__)

# ── PDF raster resolution ─────────────────────────────────────────────────────
PDF_RENDER_DPI = 150      # 150 DPI -> good detail, manageable file size
PDF_MAX_PX = 4000         # cap longest dimension to avoid memory issues

# ── HoughLinesP parameters (tuned for architectural drawings) ─────────────────
HOUGH_THRESHOLD   = 60    # vote threshold (original tuned value)
HOUGH_MIN_LENGTH  = 40    # minimum line length in pixels
HOUGH_MAX_GAP     = 7     # max gap to bridge: merges rendering artifact gaps (2-6px)
                           # without bridging real door openings (~15-25px at this scale)
HOUGH_RHO         = 1     # distance resolution in pixels
HOUGH_THETA       = np.pi / 180  # angle resolution


def pdf_to_dxf(input_pdf: str, output_dir: str = "./tmp", page: int = 0) -> str:
    """
    Convert a PDF floor plan to a synthetic DXF file using local tools only.

    Parameters
    ----------
    input_pdf : str
        Path to the input PDF file.
    output_dir : str
        Directory where the output DXF will be written.
    page : int
        0-based page index to render (default: 0 = first page).

    Returns
    -------
    str
        Absolute path to the generated DXF file.

    Raises
    ------
    RuntimeError
        If the PDF cannot be rendered by any local method.
    """
    input_path = Path(input_pdf).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    dxf_out = output_path / f"{stem}_from_pdf.dxf"

    logger.info(f"[PDF->DXF] Input: {input_path}")

    # Step 1: Render PDF page to bitmap
    image = _render_pdf_to_image(str(input_path), page)

    if image is None:
        raise RuntimeError(
            f"[PDF->DXF] Could not render PDF '{input_path}' with any local method.\n"
            "  Install pypdfium2 (python -m pip install pypdfium2) or\n"
            "  Install pdf2image + Poppler (https://poppler.freedesktop.org/) and add to PATH."
        )

    h, w = image.shape[:2]
    logger.info(f"[PDF->DXF] Rendered page {page}: {w}x{h} px")

    is_raster = _detect_rasterized(image)
    if is_raster:
        logger.warning(
            "[PDF->DXF] Detected rasterized/scanned PDF (non-vector). "
            "Proceeding with best-effort image processing. "
            "Conversion quality may be lower than a CAD-exported PDF."
        )
    else:
        logger.info("[PDF->DXF] Detected vector-style PDF -- high-quality conversion expected.")

    # Step 2: Image -> edge map -> line segments
    lines = _extract_line_segments(image)
    logger.info(f"[PDF->DXF] Extracted {len(lines)} line segments")

    if len(lines) == 0:
        raise RuntimeError(
            "[PDF->DXF] No line geometry detected in the PDF. "
            "The floor plan may be too light, a photo, or an unrecognized format."
        )

    # Step 3: Scale pixel coords to real-world DXF units
    # Normalise so the longest dimension = ~50,000 DXF units (50m at 1=mm)
    dxf_scale = 50_000.0 / max(w, h)

    # Step 4: Write DXF
    doc = ezdxf.new(dxfversion="R2013")
    doc.header["$INSUNITS"] = 4   # 4 = millimetres
    msp = doc.modelspace()

    doc.layers.new(name="FLOOR_PLAN", dxfattribs={"color": 7})

    for x1, y1, x2, y2 in lines:
        # Flip Y: image Y increases downward, DXF Y increases upward
        dx1 = float(x1) * dxf_scale
        dy1 = float(h - y1) * dxf_scale
        dx2 = float(x2) * dxf_scale
        dy2 = float(h - y2) * dxf_scale

        msp.add_line(
            start=(dx1, dy1, 0),
            end=(dx2, dy2, 0),
            dxfattribs={"layer": "FLOOR_PLAN"},
        )

    doc.saveas(str(dxf_out))
    logger.info(f"[PDF->DXF] DXF saved: {dxf_out}  ({len(lines)} entities)")

    return str(dxf_out)


# =============================================================================
# PDF RENDERING (multi-strategy, local only)
# =============================================================================

def _render_pdf_to_image(pdf_path: str, page: int = 0) -> Optional[np.ndarray]:
    """
    Render a PDF page to a numpy RGB image using local tools.

    Tries strategies in order:
      1. pypdfium2   -- pure Python, zero system deps, fastest
      2. pdf2image   -- requires local Poppler binary (free, open-source)

    Returns numpy uint8 BGR image, or None if all strategies fail.
    """
    img = _try_pypdfium2(pdf_path, page)
    if img is not None:
        logger.debug("[PDF->DXF] Strategy 1 (pypdfium2) succeeded")
        return img

    img = _try_pdf2image(pdf_path, page)
    if img is not None:
        logger.debug("[PDF->DXF] Strategy 2 (pdf2image) succeeded")
        return img

    logger.error("[PDF->DXF] All PDF rendering strategies failed")
    return None


def _try_pypdfium2(pdf_path: str, page: int) -> Optional[np.ndarray]:
    """Render via pypdfium2 (https://github.com/pypdfium2-team/pypdfium2)."""
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]

        doc = pdfium.PdfDocument(pdf_path)
        if page >= len(doc):
            page = 0
        pg = doc[page]

        # Compute scale factor
        w_pt, h_pt = pg.get_size()
        scale = PDF_RENDER_DPI / 72.0  # PDF points are 72 DPI

        # Cap to max dimension
        long_side = max(w_pt * scale, h_pt * scale)
        if long_side > PDF_MAX_PX:
            scale *= PDF_MAX_PX / long_side

        bitmap = pg.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()

        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    except ImportError:
        logger.debug("[PDF->DXF] pypdfium2 not installed")
        return None
    except Exception as e:
        logger.debug(f"[PDF->DXF] pypdfium2 failed: {e}")
        return None


def _try_pdf2image(pdf_path: str, page: int) -> Optional[np.ndarray]:
    """Render via pdf2image + local Poppler."""
    try:
        from pdf2image import convert_from_path  # type: ignore[import-not-found]

        images = convert_from_path(
            pdf_path,
            dpi=PDF_RENDER_DPI,
            first_page=page + 1,
            last_page=page + 1,
        )
        if not images:
            return None

        pil_img = images[0]
        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        # Cap dimensions
        h, w = img_rgb.shape[:2]
        if max(h, w) > PDF_MAX_PX:
            scale = PDF_MAX_PX / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    except ImportError:
        logger.debug("[PDF->DXF] pdf2image not installed")
        return None
    except Exception as e:
        logger.debug(f"[PDF->DXF] pdf2image failed: {e}")
        return None


# =============================================================================
# IMAGE PROCESSING -- LINE SEGMENT EXTRACTION
# =============================================================================

def _detect_rasterized(image: np.ndarray) -> bool:
    """
    Heuristic: is this a scanned PDF vs. a clean vector PDF render?
    Vector renders have very crisp edges and near-white backgrounds.
    Scanned PDFs have JPEG artifacts, noise, and non-white backgrounds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Background uniformity: in vector PDFs, most pixels are exactly white
    white_fraction = np.sum(gray > 240) / gray.size
    if white_fraction < 0.70:
        return True  # non-white background -> likely scanned

    # Noise level via Laplacian variance (high = sharp/clean, low = blurry/noisy)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return True  # blurry -> likely a photo/scan

    return False


def _extract_line_segments(image: np.ndarray) -> list:
    """
    Extract geometric line segments from a floor plan image.

    Pipeline:
      1. Grayscale + adaptive threshold (handles varying brightness)
      2. Morphological closing (connect broken wall segments)
      3. Canny edge detection
      4. HoughLinesP (probabilistic Hough transform)
      5. Filter and snap lines to horizontal/vertical

    Returns list of (x1, y1, x2, y2) in pixel coordinates.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pre-filter: force all light-grey pixels (guides, dimension lines,
    # staircase hatch, furniture) to pure white BEFORE thresholding so
    # the Hough transform never sees them.
    gray[gray > 180] = 255

    # Hard-crop the bottom 18% to remove the title block / legend text
    # before Hough runs — prevents text rows from becoming fake wall lines.
    h_crop = int(h * 0.82)
    gray = gray[:h_crop, :]

    # 1. Adaptive threshold: robust to brightness variation in scanned docs
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=8
    )

    # 2. Morphological closing: connect small gaps in wall lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Canny edges
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)

    # 4. Adaptive Hough parameters based on image size.
    # min_length at 2% of the shorter side captures walls while skipping tiny stubs.
    min_length = max(HOUGH_MIN_LENGTH, int(min(w, h) * 0.02))

    raw = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_length,
        maxLineGap=HOUGH_MAX_GAP,
    )

    if raw is None:
        return []

    lines = [(int(x[0]), int(x[1]), int(x[2]), int(x[3])) for x in raw.reshape(-1, 4)]

    # 5. Filter and snap to H/V
    lines = _filter_lines(lines, min_length=min_length)

    return lines


def _filter_lines(
    lines: list,
    min_length: int = 30,
    angle_snap_deg: float = 8.0,
) -> list:
    """
    Filter Hough lines to near-horizontal and near-vertical segments only.

    Floor plans from clean CAD-exported PDFs are almost entirely orthogonal
    (walls are H/V).  Diagonal noise comes from:
      - Door/window swing arcs approximated as line segments
      - Staircase cross-hatch marks
      - Roof outline lines (long but not structural)
      - Dimension leader lines and text character edges

    Keeping a generous snap_deg (8°) ensures slightly-skewed walls from
    scan tilt or imprecise CAD export are correctly captured as H/V.

    Note: floor plans with genuinely angled structural walls (e.g., diagonal
    terrace walls) are better handled by the raster fallback path, which
    performs direct pixel-to-grid mapping rather than line detection.
    """
    filtered = []
    snap_rad = math.radians(angle_snap_deg)

    for x1, y1, x2, y2 in lines:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue

        angle = math.atan2(abs(y2 - y1), abs(x2 - x1))  # 0=horizontal, pi/2=vertical

        # Snap near-horizontal lines → pure horizontal
        if angle < snap_rad:
            y_avg = (y1 + y2) // 2
            filtered.append((x1, y_avg, x2, y_avg))

        # Snap near-vertical lines → pure vertical
        elif angle > (math.pi / 2 - snap_rad):
            x_avg = (x1 + x2) // 2
            filtered.append((x_avg, y1, x_avg, y2))

        # All diagonals dropped — they are arc/hatch/roof/text noise in vector PDFs.
        # For floor plans with structural diagonal walls, the raster fallback path
        # (triggered when free% > 88.5) produces better results than Hough.

    return filtered


# =============================================================================
# UTILITY
# =============================================================================

def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
        return len(pdfium.PdfDocument(pdf_path))
    except Exception:
        pass

    try:
        from pdf2image import pdfinfo_from_path  # type: ignore[import-not-found]
        info = pdfinfo_from_path(pdf_path)
        return int(info.get("Pages", 1))
    except Exception:
        return 1
