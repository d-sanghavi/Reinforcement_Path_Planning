"""
scale_utils.py
──────────────
Bidirectional coordinate mapping between:
  - DXF drawing coordinates (real-world units: mm, m, inches, etc.)
  - Grid cell indices (row, col) in the occupancy grid

Also handles:
  - DXF unit detection and meters-per-unit conversion factor
  - Grid resolution determination (user-configurable cell size in real-world units)
  - Coordinate round-tripping for path export

Usage:
    from cad_ingestion.scale_utils import ScaleMapper
    mapper = ScaleMapper(parsed_dxf, cell_size_m=0.1)
    grid_coord = mapper.dxf_to_grid(dxf_x, dxf_y)
    dxf_coord  = mapper.grid_to_dxf(row, col)
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Meters per DXF unit (from $INSUNITS code) ────────────────────────────────
METERS_PER_UNIT = {
    0: 1.0,           # unitless — assume meters
    1: 0.0254,        # inches
    2: 0.3048,        # feet
    3: 1609.344,      # miles
    4: 0.001,         # millimeters
    5: 0.01,          # centimeters
    6: 1.0,           # meters
    7: 1000.0,        # kilometers
    8: 2.54e-8,       # microinches
    9: 2.54e-5,       # mils
    10: 0.9144,       # yards
    14: 0.1,          # decimeters
}

DEFAULT_CELL_SIZE_M = 0.1  # 10 cm per grid cell


class ScaleMapper:
    """
    Handles all coordinate transforms between DXF space and grid space.

    DXF space: floating-point (x, y) in drawing units
    Grid space: integer (row, col) — row 0 is the TOP of the grid (image convention)

    Parameters
    ----------
    drawing_bounds : BoundingBox
        The overall bounding box of the DXF drawing (from dxf_parser).
    units_code : int
        $INSUNITS code from the DXF header.
    cell_size_m : float
        Desired grid cell size in meters (e.g., 0.1 for 10 cm).
    """

    def __init__(self, drawing_bounds, units_code: int, cell_size_m: float = DEFAULT_CELL_SIZE_M):
        self.bounds = drawing_bounds
        self.units_code = units_code
        self.cell_size_m = cell_size_m

        # Meters per drawing unit
        self.m_per_unit = METERS_PER_UNIT.get(units_code, 1.0)
        if units_code not in METERS_PER_UNIT:
            logger.warning(
                f"Unknown $INSUNITS code {units_code}, assuming 1 unit = 1 meter"
            )

        # Grid cell size in DXF units
        self.cell_size_dxf = cell_size_m / self.m_per_unit

        # Grid dimensions
        width_dxf = drawing_bounds.max_x - drawing_bounds.min_x
        height_dxf = drawing_bounds.max_y - drawing_bounds.min_y

        self.grid_cols = max(1, math.ceil(width_dxf / self.cell_size_dxf))
        self.grid_rows = max(1, math.ceil(height_dxf / self.cell_size_dxf))

        logger.info(
            f"[ScaleMapper] units_code={units_code} ({self.m_per_unit} m/unit), "
            f"cell_size={cell_size_m}m = {self.cell_size_dxf:.4f} DXF units, "
            f"grid={self.grid_rows}×{self.grid_cols} cells "
            f"covering {width_dxf*self.m_per_unit:.1f}×{height_dxf*self.m_per_unit:.1f} m"
        )

    # ── Coordinate Transforms ─────────────────────────────────────────────────

    def dxf_to_grid(self, dxf_x: float, dxf_y: float) -> tuple:
        """
        Convert DXF (x, y) to grid (row, col).
        Row 0 = top of grid (y_max in DXF space).
        """
        col = (dxf_x - self.bounds.min_x) / self.cell_size_dxf
        row = (self.bounds.max_y - dxf_y) / self.cell_size_dxf  # flip Y axis

        col = max(0, min(self.grid_cols - 1, int(col)))
        row = max(0, min(self.grid_rows - 1, int(row)))
        return (row, col)

    def grid_to_dxf(self, row: int, col: int) -> tuple:
        """
        Convert grid (row, col) to DXF (x, y) at cell CENTER.
        """
        dxf_x = self.bounds.min_x + (col + 0.5) * self.cell_size_dxf
        dxf_y = self.bounds.max_y - (row + 0.5) * self.cell_size_dxf
        return (dxf_x, dxf_y)

    def grid_to_real_world(self, row: int, col: int) -> tuple:
        """
        Convert grid (row, col) to real-world meters from drawing origin.
        """
        dxf_x, dxf_y = self.grid_to_dxf(row, col)
        real_x = (dxf_x - self.bounds.min_x) * self.m_per_unit
        real_y = (dxf_y - self.bounds.min_y) * self.m_per_unit
        return (real_x, real_y)

    def path_to_real_world(self, path: list) -> list:
        """
        Convert a list of (row, col) cells to real-world (x, y) meters.
        Returns list of (x_m, y_m) tuples.
        """
        return [self.grid_to_real_world(r, c) for r, c in path]

    def path_to_dxf_coords(self, path: list) -> list:
        """
        Convert a list of (row, col) cells to DXF coordinate pairs.
        Returns list of (dxf_x, dxf_y) tuples.
        """
        return [self.grid_to_dxf(r, c) for r, c in path]

    def path_length_meters(self, path: list) -> float:
        """
        Compute the total length of a grid path in real-world meters.
        """
        if len(path) < 2:
            return 0.0
        real_coords = self.path_to_real_world(path)
        total = 0.0
        for i in range(len(real_coords) - 1):
            dx = real_coords[i + 1][0] - real_coords[i][0]
            dy = real_coords[i + 1][1] - real_coords[i][1]
            total += math.hypot(dx, dy)
        return total

    def dxf_bbox_to_grid_bbox(self, min_x, min_y, max_x, max_y) -> tuple:
        """
        Convert DXF bounding box to grid cell bounding box.
        Returns (row_min, col_min, row_max, col_max).
        """
        r1, c1 = self.dxf_to_grid(min_x, max_y)  # top-left in grid
        r2, c2 = self.dxf_to_grid(max_x, min_y)  # bottom-right in grid
        return (
            max(0, min(r1, r2)),
            max(0, min(c1, c2)),
            min(self.grid_rows - 1, max(r1, r2)),
            min(self.grid_cols - 1, max(c1, c2)),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save mapping config to JSON for later use."""
        data = {
            "units_code": self.units_code,
            "m_per_unit": self.m_per_unit,
            "cell_size_m": self.cell_size_m,
            "cell_size_dxf": self.cell_size_dxf,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "bounds": {
                "min_x": self.bounds.min_x,
                "min_y": self.bounds.min_y,
                "max_x": self.bounds.max_x,
                "max_y": self.bounds.max_y,
            },
        }
        Path(filepath).write_text(json.dumps(data, indent=2))
        logger.info(f"[ScaleMapper] Mapping saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, BoundingBoxClass=None):
        """Restore a ScaleMapper from a saved JSON file."""
        data = json.loads(Path(filepath).read_text())

        # Reconstruct a simple bbox-like object
        class _BB:
            def __init__(self, d):
                self.min_x = d["min_x"]
                self.min_y = d["min_y"]
                self.max_x = d["max_x"]
                self.max_y = d["max_y"]

        bounds = _BB(data["bounds"])
        mapper = cls(bounds, data["units_code"], data["cell_size_m"])
        return mapper

    @property
    def unit_label(self) -> str:
        labels = {
            0: "units", 1: "in", 2: "ft", 3: "mi",
            4: "mm", 5: "cm", 6: "m", 7: "km",
            8: "µin", 9: "mils", 10: "yd", 14: "dm",
        }
        return labels.get(self.units_code, "units")


def auto_cell_size(
    drawing_bounds,
    units_code: int,
    target_max_dim: int = 500,
    default_cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> float:
    """
    Automatically determine cell size so the grid doesn't exceed
    `target_max_dim` cells in either dimension.

    Returns the recommended cell_size_m.
    """
    m_per_unit = METERS_PER_UNIT.get(units_code, 1.0)
    width_m = (drawing_bounds.max_x - drawing_bounds.min_x) * m_per_unit
    height_m = (drawing_bounds.max_y - drawing_bounds.min_y) * m_per_unit

    max_dim_m = max(width_m, height_m)
    if max_dim_m <= 0:
        return default_cell_size_m

    # Target: no more than target_max_dim cells on any side
    min_cell_size = max_dim_m / target_max_dim
    # Use the larger of the minimum required and the user default
    recommended = max(min_cell_size, default_cell_size_m)

    logger.info(
        f"[AutoCellSize] Drawing {width_m:.1f}×{height_m:.1f}m → "
        f"cell_size={recommended:.3f}m → "
        f"grid≈{int(height_m/recommended)}×{int(width_m/recommended)} cells"
    )
    return recommended
