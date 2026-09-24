"""
export_coordinates.py
─────────────────────
Convert a grid-cell path to real-world coordinates and export as JSON.

Output format (coordinates.json):
{
  "source_file": "floor_plan.dxf",
  "units": "meters",
  "total_length_m": 12.45,
  "path_cells": [[0,5], [1,5], ...],
  "path_real_world": [[x0,y0], [x1,y1], ...],
  "start_grid": [row, col],
  "goal_grid": [row, col],
  "start_real_world": [x, y],
  "goal_real_world": [x, y],
  "waypoints_count": 42
}
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_coordinates(
    path: list,
    mapper,
    output_path: str = "Results/coordinates.json",
    source_file: str = "unknown",
) -> dict:
    """
    Convert grid path to real-world coordinates and write JSON.

    Parameters
    ----------
    path : list of (row, col) tuples
    mapper : ScaleMapper from scale_utils.py
    output_path : str
    source_file : str

    Returns
    -------
    dict : the coordinate data dict (also written to file)
    """
    if not path:
        logger.warning("[CoordExport] Empty path — nothing to export")
        return {}

    # Convert each cell to real-world (meters from drawing origin)
    real_coords = mapper.path_to_real_world(path)
    dxf_coords = mapper.path_to_dxf_coords(path)
    total_length_m = mapper.path_length_meters(path)

    start_real = mapper.grid_to_real_world(path[0][0], path[0][1])
    goal_real = mapper.grid_to_real_world(path[-1][0], path[-1][1])

    data = {
        "source_file": source_file,
        "units": "meters",
        "total_length_m": round(total_length_m, 4),
        "waypoints_count": len(path),
        "start_grid": list(path[0]),
        "goal_grid": list(path[-1]),
        "start_real_world_m": [round(x, 4) for x in start_real],
        "goal_real_world_m": [round(x, 4) for x in goal_real],
        "path_cells": [list(p) for p in path],
        "path_real_world_m": [[round(x, 4), round(y, 4)] for x, y in real_coords],
        "path_dxf_coords": [[round(x, 4), round(y, 4)] for x, y in dxf_coords],
        "grid_cell_size_m": round(mapper.cell_size_m, 4),
        "coordinate_system": "origin=drawing_min_corner, Y_up=DXF_Y_up",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(
        f"[CoordExport] Saved {len(path)} waypoints, "
        f"length={total_length_m:.2f}m → {output_path}"
    )

    return data
