"""
raster_to_grid.py
─────────────────
Direct raster-image → occupancy-grid converter (zero DXF, zero vector data needed).

Used as a fallback when a PDF floor plan contains diagonal structural walls or other
geometry that Hough line detection misses (i.e. vector free% > 88.5%).

Pipeline (KEY INSIGHT — auto-crop before resize):
  1. Load image (from file path or pre-rendered numpy array)
  2. Crop bottom 20% (title block / legend)
  3. Threshold at 200 → capture all drawn elements (walls + text + dims)
  4. AUTO-CROP to floor plan content bounding box  ← critical: removes blank page margins
  5. CC filter: remove tiny blobs (individual text characters)
  6. MORPH_CLOSE: seal micro-gaps in wall lines
  7. Pre-dilation: thicken thin walls so they survive downsampling
  8. Resize to target grid dimensions (mapper dims if provided, else max_dim)
  9. Re-threshold at 25% coverage — preserves thin interior walls
  10. Return binary occupancy grid (0 = free, 1 = obstacle)

WHY auto-crop matters:
  A4 portrait PDF renders at 1653×2339 px @ 200 DPI, but the building is only
  ~1200×800 px inside it. Without cropping, each interior wall (≈5 px at 1:200 scale)
  becomes < 1 cell after downsampling to 500 cells — completely invisible.
  After auto-crop, the building fills the entire grid and walls become 2-4 cells wide.

Zero paid APIs. Zero cloud services. Pure OpenCV.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MAX_DIM = 500   # target grid longest dimension if no explicit dims given


def _mask_text_regions(img_bgr: np.ndarray) -> np.ndarray:
    """Detect and mask alphanumeric text with white to prevent them becoming obstacles."""
    try:
        import pytesseract
        data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
        masked = img_bgr.copy()
        n_boxes = len(data['level'])
        count = 0
        for i in range(n_boxes):
            if float(data['conf'][i]) > 0.0:
                text = data['text'][i].strip()
                if len(text) > 0:
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    pad = 5
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end   = min(masked.shape[1], x + w + pad)
                    y_end   = min(masked.shape[0], y + h + pad)
                    cv2.rectangle(masked, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)
                    count += 1
        if count > 0:
            logger.info(f"[RasterGrid] Masked {count} text/dimension regions via Tesseract.")
        return masked
    except ImportError:
        logger.debug("[RasterGrid] pytesseract not installed, skipping text masking.")
        return img_bgr
    except Exception as e:
        logger.debug(f"[RasterGrid] Tesseract execution failed: {e}")
        return img_bgr


def image_to_occupancy_grid(
    source: Union[str, np.ndarray],
    max_dim: int = DEFAULT_MAX_DIM,
    real_world_m: Optional[float] = None,
    invert: bool = False,
    target_rows: Optional[int] = None,
    target_cols: Optional[int] = None,
) -> tuple:
    """
    Convert a floor plan image to a binary occupancy grid.

    Parameters
    ----------
    source : str or np.ndarray
        File path to an image (PNG/JPG/BMP) or a pre-loaded BGR numpy array.
    max_dim : int
        Target maximum grid dimension if target_rows/target_cols are not given.
    real_world_m : float or None
        Optional known real-world extent (longest side in metres).
    invert : bool
        If True, invert the grayscale image first (dark background / light walls).
    target_rows : int or None
        Explicit target grid height in cells (from vector-path ScaleMapper).
    target_cols : int or None
        Explicit target grid width in cells (from vector-path ScaleMapper).

    Returns
    -------
    tuple : (grid, meta)
        grid — np.ndarray uint8, shape (H, W), 0=free 1=obstacle
        meta — dict with keys: rows, cols, px_per_cell, cell_size_m (if known)
    """
    # ── 1. Load grayscale ─────────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        gray_img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise FileNotFoundError(f"[RasterGrid] Cannot read image: {source}")
    else:
        if len(source.shape) == 3:
            gray_img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = source.copy()

    if invert:
        gray_img = cv2.bitwise_not(gray_img)

    orig_h, orig_w = gray_img.shape[:2]
    logger.debug(f"[RasterGrid] Input image: {orig_w}×{orig_h} px")

    # ── 2. Bottom crop (title block / legend strip) ────────────────────────────
    # Remove the bottom 20% which typically contains "Plan du Rez de Chaussée",
    # scale bar, and dimension text that would pollute the grid as obstacles.
    h, w = gray_img.shape
    gray_cropped = gray_img[0 : int(h * 0.80), :]

    # ── 3. Threshold: capture all drawn elements ───────────────────────────────
    # 200 threshold: anything darker than very-light-grey = content.
    # Strict enough to avoid lightly-shaded floor fills becoming walls.
    _, binary = cv2.threshold(gray_cropped, 200, 255, cv2.THRESH_BINARY_INV)

    # ── 4. AUTO-CROP to floor plan content bounding box ───────────────────────
    # THE critical step. A4 portrait PDF = 1653×2339 px @ 200 DPI, but the
    # building itself is only ~1200×800 px inside it. Without this crop, every
    # interior wall (≈5 px) becomes < 1 cell after resize → invisible.
    # After auto-crop the building fills the full grid → walls 2-4 cells wide.
    row_has_content = np.any(binary > 0, axis=1)
    col_has_content = np.any(binary > 0, axis=0)

    crop_h, crop_w = binary.shape   # defaults if crop fails

    if row_has_content.any() and col_has_content.any():
        r_min = int(np.argmax(row_has_content))
        r_max = int(len(row_has_content) - np.argmax(row_has_content[::-1]) - 1)
        c_min = int(np.argmax(col_has_content))
        c_max = int(len(col_has_content) - np.argmax(col_has_content[::-1]) - 1)

        # 4% padding on each side — ensures perimeter walls aren't clipped
        pad_r = max(5, int((r_max - r_min) * 0.04))
        pad_c = max(5, int((c_max - c_min) * 0.04))
        r_min = max(0, r_min - pad_r)
        r_max = min(binary.shape[0] - 1, r_max + pad_r)
        c_min = max(0, c_min - pad_c)
        c_max = min(binary.shape[1] - 1, c_max + pad_c)

        binary   = binary[r_min : r_max + 1, c_min : c_max + 1]
        crop_h, crop_w = binary.shape
        logger.info(
            f"[RasterGrid] Auto-cropped: {crop_w}×{crop_h} px "
            f"(was {w}×{gray_cropped.shape[0]} px — blank margins removed)"
        )
    else:
        logger.warning("[RasterGrid] No content found — check image.")

    # ── 5. Remove text / labels (small isolated blobs) ────────────────────────
    # Individual text characters ≈ 100-500 px². Wall segments >> 1000 px².
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 1200:
            binary[labels == i] = 0

    # ── 5b. Top-trim: remove dimension annotation at the top ──────────────────
    # The "2.545" (or similar) dimension arrow sits at the very top of the
    # floor plan drawing, above all structural walls. After auto-crop it lands
    # in the top ~5% of the cropped image. Slicing it away is the most reliable
    # fix — the dimension annotation is never part of the navigable floor plan.
    top_trim = max(1, int(crop_h * 0.05))
    binary = binary[top_trim:, :]
    crop_h = binary.shape[0]
    logger.info(f"[RasterGrid] Top-trimmed {top_trim} px (dimension annotation removed)")

    # ── 5c. Remove any remaining full-width leader / annotation lines ─────────
    # Some dimension annotations survive as a thin horizontal line spanning > 60%
    # of the image width but < 12 px tall (the line itself without the text).
    num_dim, labels_dim, stats_dim, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    dim_removed = 0
    for i in range(1, num_dim):
        w_bb = stats_dim[i, cv2.CC_STAT_WIDTH]
        h_bb = stats_dim[i, cv2.CC_STAT_HEIGHT]
        if w_bb > crop_w * 0.60 and h_bb < 12:
            binary[labels_dim == i] = 0
            dim_removed += 1
    if dim_removed:
        logger.info(f"[RasterGrid] Removed {dim_removed} full-width annotation line(s)")

    # ── 6. Seal micro-gaps and thicken thin walls ──────────────────────────────
    # MORPH_CLOSE joins nearby wall fragments into continuous bands.
    kernel_close = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Pre-dilation: ensures thin interior walls (4-8 px at 200 DPI 1:200 scale)
    # survive the resize. 4 px → 6 px → after 0.25× scale → 1.5 cells ✅
    kernel_dilate = np.ones((2, 2), np.uint8)
    final_grid = cv2.dilate(binary, kernel_dilate, iterations=1)



    # ── 7. Resize to target grid dimensions ────────────────────────────────────
    # Prefer explicit mapper dimensions so coordinate systems align with the
    # vector path. Fall back to max_dim scaling if not provided.
    if target_rows and target_cols:
        grid_h, grid_w = int(target_rows), int(target_cols)
        logger.info(
            f"[RasterGrid] Resizing to mapper target: {grid_w}×{grid_h} cells"
        )
    else:
        scale  = max_dim / max(crop_h, crop_w)
        grid_w = max(2, int(crop_w * scale))
        grid_h = max(2, int(crop_h * scale))

    small = cv2.resize(final_grid, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # Re-threshold at 25% coverage: wall covering ≥25% of cell area = obstacle.
    # Standard 50% (127) threshold loses thin single-cell-wide interior walls.
    _, grid_bin = cv2.threshold(small, 63, 255, cv2.THRESH_BINARY)

    # ── 8. Convert to 0/1 occupancy ───────────────────────────────────────────
    grid = (grid_bin > 0).astype(np.uint8)

    # ── 9. Border safety ──────────────────────────────────────────────────────
    grid[0, :]  = 1
    grid[-1, :] = 1
    grid[:, 0]  = 1
    grid[:, -1] = 1

    free_pct = 100 * (grid == 0).sum() / grid.size
    logger.info(
        f"[RasterGrid] Grid: {grid_h}×{grid_w}  "
        f"Free: {free_pct:.1f}%  Obstacle: {100-free_pct:.1f}%"
    )

    if free_pct < 5:
        logger.warning(
            "[RasterGrid] Grid is >95% obstacle — image may be inverted. "
            "Try re-running with invert=True."
        )
    elif free_pct > 95:
        logger.warning(
            "[RasterGrid] Grid is <5% obstacle — walls may not be detected."
        )

    px_per_cell = max(crop_h, crop_w) / max(grid_h, grid_w)
    meta = {
        "rows":        grid_h,
        "cols":        grid_w,
        "orig_h":      orig_h,
        "orig_w":      orig_w,
        "px_per_cell": px_per_cell,
        "cell_size_m": (real_world_m / max(grid_h, grid_w)) if real_world_m else None,
        "free_pct":    free_pct,
    }

    return grid, meta


def pdf_image_to_occupancy_grid(
    pdf_path: str,
    page: int = 0,
    max_dim: int = DEFAULT_MAX_DIM,
    dpi: int = 200,
    target_rows: Optional[int] = None,
    target_cols: Optional[int] = None,
) -> tuple:
    """
    Render a PDF page directly to an occupancy grid (skips DXF entirely).

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF.
    page : int
        0-based page index.
    max_dim : int
        Target grid longest dimension (used if target_rows/cols not given).
    dpi : int
        Render resolution. 200 DPI recommended for clean PDF exports.
    target_rows : int or None
        Explicit target rows from ScaleMapper (preferred over max_dim).
    target_cols : int or None
        Explicit target cols from ScaleMapper (preferred over max_dim).

    Returns
    -------
    tuple : (grid, meta) — same as image_to_occupancy_grid
    """
    img_bgr = _render_pdf_page(pdf_path, page, dpi)
    if img_bgr is None:
        raise RuntimeError(
            f"[RasterGrid] Could not render PDF '{pdf_path}' — "
            "install pypdfium2: python -m pip install pypdfium2"
        )
    logger.info(
        f"[RasterGrid] PDF rendered: {img_bgr.shape[1]}×{img_bgr.shape[0]} px @ {dpi}dpi"
    )
    return image_to_occupancy_grid(
        img_bgr,
        max_dim=max_dim,
        target_rows=target_rows,
        target_cols=target_cols,
    )


def _render_pdf_page(pdf_path: str, page: int, dpi: int) -> Optional[np.ndarray]:
    """Render one PDF page to a BGR numpy image using pypdfium2 or pdf2image."""
    # Strategy 1: pypdfium2 (zero system deps)
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
        doc    = pdfium.PdfDocument(pdf_path)
        pg     = doc[min(page, len(doc) - 1)]
        scale  = dpi / 72.0
        bitmap = pg.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[RasterGrid] pypdfium2 failed: {e}")

    # Strategy 2: pdf2image + Poppler
    try:
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
        imgs = convert_from_path(pdf_path, dpi=dpi, first_page=page + 1, last_page=page + 1)
        if imgs:
            img_rgb = np.array(imgs[0].convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[RasterGrid] pdf2image failed: {e}")

    return None


def save_raster_grid_image(grid: np.ndarray, output_path: str):
    """Save a 0/1 occupancy grid as a black-and-white PNG image."""
    img = ((1 - grid) * 255).astype(np.uint8)   # 0=free→white, 1=obstacle→black
    cv2.imwrite(output_path, img)
    logger.info(f"[RasterGrid] Grid image saved: {output_path}")
