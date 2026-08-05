"""
build_grid.py
─────────────
Convert symbol classifications + DXF geometry into a binary occupancy grid.

Occupancy grid values:
  0 = FREE / traversable (open floor, door openings, archways)
  1 = OBSTACLE (walls, columns, furniture, stairs)

Key semantic rule:
  Door symbols mark their swing arc area as FREE — this is the critical
  reasoning step that distinguishes doors from walls in the traversability map.

Output:
  - NumPy uint8 grid array
  - PNG preview (white=free, black=obstacle)
  - coordinate mapping JSON via ScaleMapper
"""

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from symbol_model.symbol_kb.block_names import OBSTACLE_CLASSES, TRAVERSABLE_CLASSES, SKIP_CLASSES

logger = logging.getLogger(__name__)

# ── Grid cell values ──────────────────────────────────────────────────────────
FREE = 0
OBSTACLE = 1

# ── Safety margin around obstacles (cells) ───────────────────────────────────
DEFAULT_SAFETY_MARGIN = 1  # 1 cell dilation around walls/obstacles


class GridBuilder:
    """
    Builds a binary occupancy grid from classified DXF entities.

    Parameters
    ----------
    parsed_dxf : ParsedDXF
        Parsed DXF with geometry.
    classifications : dict
        group_id → classification dict from infer_symbols.py
    mapper : ScaleMapper
        Coordinate mapper from scale_utils.py
    safety_margin : int
        Number of cells to dilate around obstacles (robot safety margin).
    """

    def __init__(self, parsed_dxf, classifications: dict, mapper, safety_margin: int = DEFAULT_SAFETY_MARGIN):
        self.parsed_dxf = parsed_dxf
        self.classifications = classifications
        self.mapper = mapper
        self.safety_margin = safety_margin

        self.rows = mapper.grid_rows
        self.cols = mapper.grid_cols

        # Initialize grid: everything is FREE; obstacles are painted on top
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Boolean mask tracking cells forcibly set as door/opening traversable
        # (used by LiveTrainingViz for amber door highlight)
        self.door_cells = np.zeros((self.rows, self.cols), dtype=bool)

    def build(self) -> np.ndarray:
        """
        Execute the full grid construction pipeline.

        Returns
        -------
        np.ndarray
            Binary occupancy grid (rows × cols), uint8, 0=FREE 1=OBSTACLE.
        """
        logger.info(f"[GridBuilder] Building {self.rows}×{self.cols} grid...")

        # Build group lookup by group_id
        group_by_id = {g.group_id: g for g in self.parsed_dxf.entity_groups}

        # Separate groups by class
        obstacle_groups = []
        door_groups = []
        skip_groups = []

        for gid_str, cls_data in self.classifications.items():
            gid = int(gid_str)
            symbol_class = cls_data["class"]
            group = group_by_id.get(gid)
            if group is None:
                continue

            if symbol_class in SKIP_CLASSES:
                skip_groups.append(group)
            elif symbol_class in TRAVERSABLE_CLASSES:
                door_groups.append(group)
            elif symbol_class in OBSTACLE_CLASSES:
                obstacle_groups.append(group)
            # window, column, unknown → treat as obstacles
            elif symbol_class in ("window", "column", "unknown"):
                obstacle_groups.append(group)

        logger.info(
            f"[GridBuilder] {len(obstacle_groups)} obstacle groups, "
            f"{len(door_groups)} door groups, "
            f"{len(skip_groups)} annotation groups (skipped)"
        )

        # Step 1: Paint obstacle geometry
        for group in obstacle_groups:
            self._rasterize_group_as_obstacle(group)

        # Step 2: Apply raw geometry rasterization for unclassified walls
        self._rasterize_raw_walls()

        # Step 2b: Force text/annotation bounding boxes to FREE (removes stray dimension lines)
        for group in skip_groups:
            self._mask_annotation_group(group)

        # Step 3: Apply safety margin dilation (thickens walls for robot clearance)
        if self.safety_margin > 0:
            kernel_size = 2 * self.safety_margin + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.grid = cv2.dilate(self.grid, kernel, iterations=1)

        # ── EXPLICIT OPENCV LOGIC FROM USER ────────────────────────────────────
        # 1. Text & Noise Eraser (Run BEFORE morphology so text doesn't merge into walls)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.grid, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 300:  
                self.grid[labels == i] = 0

        # 2. Filter Thin Lines (Now that text is gone, safely kill diagonal/hatch lines)
        kernel = np.ones((3, 3), np.uint8)
        self.grid = cv2.morphologyEx(self.grid, cv2.MORPH_OPEN, kernel)
        # ───────────────────────────────────────────────────────────────────────

        # Step 4: Force ALL door/opening cells to FREE (overrides dilation & morphology)
        #         This is the key step: every detected door/opening MUST be walkable.
        for group in door_groups:
            self._force_traversable_pass(group)

        # Step 5: Ensure grid border is obstacle (prevents boundary exploits)
        self.grid[0, :] = OBSTACLE
        self.grid[-1, :] = OBSTACLE
        self.grid[:, 0] = OBSTACLE
        self.grid[:, -1] = OBSTACLE

        # ── Detect door_cells: FREE cells flanked by OBSTACLE on both sides ───────────
        # A doorway gap is a FREE cell that has walls on both its left+right
        # OR both its top+bottom. Pure vectorised: no loops, no distance transform.
        obs = (self.grid == OBSTACLE)
        # Horizontal doorway: obstacle immediately to the left AND to the right
        h_door = (np.roll(obs, 1, axis=1) & np.roll(obs, -1, axis=1) & ~obs)
        # Vertical doorway: obstacle immediately above AND below
        v_door = (np.roll(obs, 1, axis=0) & np.roll(obs, -1, axis=0) & ~obs)
        self.door_cells = h_door | v_door
        door_count = int(self.door_cells.sum())
        if door_count:
            logger.info(f"[GridBuilder] Door gaps detected: {door_count} cells marked FREE (amber)")
        # ───────────────────────────────────────────────────────────────────────

        # ── Auto-connect isolated rooms (rooms with no door gap in wall lines) ──
        self._connect_isolated_rooms()
        # ───────────────────────────────────────────────────────────────────────
        free_pct = 100 * np.sum(self.grid == FREE) / self.grid.size
        logger.info(
            f"[GridBuilder] Grid complete. "
            f"Free: {free_pct:.1f}%, "
            f"Obstacle: {100-free_pct:.1f}%"
        )

        if free_pct < 5:
            logger.warning(
                "[GridBuilder] Less than 5% of grid is free — "
                "the floor plan may not have been parsed correctly"
            )

        return self.grid

    # ───────────────────────────────────────────────────────────────────────
    def _connect_isolated_rooms(self):
        """
        Post-processing: find isolated free regions (rooms with no door gap in
        the wall line data) and create the minimum-cost wall opening to connect
        them to the main floor plan.

        Algorithm:
          1. Label all connected FREE regions.
          2. The largest region = main interior (hall + already-connected rooms).
          3. For each isolated room (size >= 80 cells and >= 2% of main):
             a. Compute distance-to-isolated and distance-to-main for every cell.
             b. Find the obstacle cell with the minimum sum of both distances
                (the “cheapest” bridge point).
             c. Open a 5-cell gap there and mark cells as door_cells (amber).
          4. Recompute connected labels after each bridge so later rooms connect
             to the already-expanded main region.

        Works universally for any floor plan regardless of how doors are drawn.
        """
        n_connected = 0

        for _ in range(20):   # safety cap — at most 20 rooms to connect per plan
            free_map = (self.grid == FREE).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_map)

            if num_labels <= 2:   # background + one region — nothing to connect
                break

            # Sort interior regions by size (descending); skip background (label 0)
            region_info = sorted(
                [(lbl, stats[lbl, cv2.CC_STAT_AREA]) for lbl in range(1, num_labels)],
                key=lambda x: -x[1],
            )
            main_label, main_size = region_info[0]

            # Find the first isolated region large enough to be a real room
            # but NOT so large it's the exterior or main connected area.
            # Typical largest room: ~50 m² = 5000 cells at 10 cm/cell.
            # Any region > 20% of grid is almost certainly exterior space, skip.
            max_room_cells = int(self.rows * self.cols * 0.20)
            # Centroid bounds: real rooms are in the inner 80% of the grid
            row_lo, row_hi = int(self.rows * 0.10), int(self.rows * 0.90)
            col_lo, col_hi = int(self.cols * 0.10), int(self.cols * 0.90)

            isolated_label = None
            isolated_area = 0
            for lbl, area in region_info[1:]:
                if area < 80 or area < main_size * 0.02:
                    continue   # too small — noise
                if area > max_room_cells:
                    continue   # too large — exterior space or large connected zone
                # Centroid must be inside the building bounds (inner 80%)
                cy = int(centroids[lbl][1])  # centroid row
                cx = int(centroids[lbl][0])  # centroid col
                if not (row_lo <= cy <= row_hi and col_lo <= cx <= col_hi):
                    continue   # centroid near border — exterior region
                isolated_label = lbl
                isolated_area = area
                break

            if isolated_label is None:
                break   # no more rooms to connect

            # Distance transforms
            isolated_mask_inv = cv2.bitwise_not(
                (labels == isolated_label).astype(np.uint8) * 255
            )
            main_mask_inv = cv2.bitwise_not(
                (labels == main_label).astype(np.uint8) * 255
            )
            dist_from_iso  = cv2.distanceTransform(isolated_mask_inv, cv2.DIST_L2, 5)
            dist_from_main = cv2.distanceTransform(main_mask_inv,     cv2.DIST_L2, 5)

            # Only consider obstacle cells that are NOT on the 3-cell border
            obs = (self.grid == OBSTACLE)
            inner = np.zeros_like(obs)
            inner[3:-3, 3:-3] = True
            valid_bridge = obs & inner

            bridge_cost = dist_from_iso + dist_from_main
            bridge_cost[~valid_bridge] = np.inf

            if np.all(np.isinf(bridge_cost)):
                break

            # Optimal bridge point
            br, bc = np.unravel_index(np.argmin(bridge_cost), bridge_cost.shape)

            # Choose gap orientation (H or V) based on lower cost
            h_cost = (dist_from_iso [br, max(0, bc-3)] +
                      dist_from_main[br, min(self.cols-1, bc+3)])
            v_cost = (dist_from_iso [max(0, br-3), bc] +
                      dist_from_main[min(self.rows-1, br+3), bc])

            GAP = 5   # ~50 cm door width at 10 cm/cell
            if h_cost <= v_cost:
                for dc in range(-1, GAP):
                    nc = bc + dc
                    if 1 <= nc < self.cols - 1:
                        self.grid[br, nc]      = FREE
                        self.door_cells[br, nc] = True
            else:
                for dr in range(-1, GAP):
                    nr = br + dr
                    if 1 <= nr < self.rows - 1:
                        self.grid[nr, bc]      = FREE
                        self.door_cells[nr, bc] = True

            n_connected += 1
            logger.info(
                f"[GridBuilder] Auto-door: connected isolated room "
                f"(area={isolated_area}) at grid ({br},{bc})"
            )

        if n_connected:
            logger.info(f"[GridBuilder] Auto-door: opened {n_connected} room(s) total")

    def _rasterize_group_as_obstacle(self, group):
        """Paint all entity primitives in a group as obstacles on the grid."""
        for entity in group.entities:
            self._rasterize_entity(entity, value=OBSTACLE)

    def _rasterize_entity(self, entity, value: int):
        """Rasterize a single entity onto the grid."""
        etype = entity.entity_type

        if etype == "LINE":
            if len(entity.points) >= 2:
                self._draw_line_on_grid(entity.points[0], entity.points[1], value)

        elif etype in ("LWPOLYLINE", "POLYLINE"):
            pts = entity.points
            for i in range(len(pts) - 1):
                self._draw_line_on_grid(pts[i], pts[i + 1], value)
            if entity.extra.get("is_closed", False) and len(pts) > 2:
                self._draw_line_on_grid(pts[-1], pts[0], value)

        elif etype == "ARC":
            self._draw_arc_on_grid(entity, value)

        elif etype == "CIRCLE":
            self._draw_circle_on_grid(entity, value)

        elif etype == "SPLINE":
            pts = entity.points
            for i in range(len(pts) - 1):
                self._draw_line_on_grid(pts[i], pts[i + 1], value)

        # Fill bounding box for INSERT / generic entities
        elif etype in ("INSERT", "SOLID", "TRACE"):
            bbox = entity.bbox
            r_min, c_min, r_max, c_max = self.mapper.dxf_bbox_to_grid_bbox(
                bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
            )
            self.grid[r_min:r_max+1, c_min:c_max+1] = value

    def _draw_line_on_grid(self, pt1: tuple, pt2: tuple, value: int):
        """Bresenham line drawing on the grid."""
        r1, c1 = self.mapper.dxf_to_grid(pt1[0], pt1[1])
        r2, c2 = self.mapper.dxf_to_grid(pt2[0], pt2[1])

        # Use OpenCV line to draw on grid
        temp = np.zeros((self.rows, self.cols), dtype=np.uint8)
        cv2.line(temp, (c1, r1), (c2, r2), 255, 1)
        self.grid[temp > 0] = value

    def _draw_arc_on_grid(self, entity, value: int):
        """Rasterize an arc onto the grid."""
        center_dxf = entity.extra.get("center", (0, 0))
        radius_dxf = entity.extra.get("radius", 0)
        start_a = entity.extra.get("start_angle", 0)
        end_a = entity.extra.get("end_angle", 360)
        swept = entity.extra.get("swept_angle", 360)

        # Sample arc as polyline
        n_pts = max(8, int(swept / 5))
        pts = []
        for i in range(n_pts + 1):
            a = math.radians(start_a + swept * i / n_pts)
            x = center_dxf[0] + radius_dxf * math.cos(a)
            y = center_dxf[1] + radius_dxf * math.sin(a)
            pts.append((x, y))

        for i in range(len(pts) - 1):
            self._draw_line_on_grid(pts[i], pts[i + 1], value)

    def _draw_circle_on_grid(self, entity, value: int):
        """Rasterize a circle onto the grid."""
        center_dxf = entity.extra.get("center", (0, 0))
        radius_dxf = entity.extra.get("radius", 0)

        cr, cc = self.mapper.dxf_to_grid(center_dxf[0], center_dxf[1])
        # Radius in grid cells
        r_cells = max(1, int(radius_dxf / self.mapper.cell_size_dxf))

        temp = np.zeros((self.rows, self.cols), dtype=np.uint8)
        cv2.circle(temp, (cc, cr), r_cells, 255, -1)  # filled circle
        self.grid[temp > 0] = value

    def _rasterize_raw_walls(self):
        """
        Rasterize unclassified LINE entities that likely represent walls.
        This catches walls that weren't classified as part of a group.
        """
        classified_handles = set()
        for cls_data in self.classifications.values():
            classified_handles.update(cls_data.get("entities", []))

        for prim in self.parsed_dxf.primitives:
            if prim.handle in classified_handles:
                continue
            # Long lines not yet classified → treat as wall
            if prim.entity_type == "LINE":
                length = prim.extra.get("length", 0)
                # Only paint lines longer than 2 grid cells
                min_length = self.mapper.cell_size_dxf * 2
                if length >= min_length:
                    self._rasterize_entity(prim, value=OBSTACLE)

    def _force_traversable_pass(self, group, clearance: int = 2):
        """
        Forcibly mark ALL cells associated with a door/opening group as FREE.

        This is stronger than the old _mark_door_as_free approach:
        - Clears every single entity in the group (lines, arcs, inserts)
        - Applies a configurable clearance padding to guarantee the path
          can actually enter and exit the doorway
        - Tracks cleared cells in self.door_cells for visualization

        Parameters
        ----------
        group : EntityGroup
            A door or opening entity group.
        clearance : int
            Extra cell padding around the door geometry (default: 2 cells).
        """
        # Collect all grid cells touched by any entity in this group
        touched_cells = set()

        for entity in group.entities:
            etype = entity.entity_type

            if etype == "LINE" and len(entity.points) >= 2:
                r1, c1 = self.mapper.dxf_to_grid(entity.points[0][0], entity.points[0][1])
                r2, c2 = self.mapper.dxf_to_grid(entity.points[1][0], entity.points[1][1])
                touched_cells.update(self._cells_on_line(r1, c1, r2, c2))

            elif etype in ("LWPOLYLINE", "POLYLINE"):
                pts = entity.points
                for i in range(len(pts) - 1):
                    r1, c1 = self.mapper.dxf_to_grid(pts[i][0], pts[i][1])
                    r2, c2 = self.mapper.dxf_to_grid(pts[i+1][0], pts[i+1][1])
                    touched_cells.update(self._cells_on_line(r1, c1, r2, c2))

            elif etype == "ARC":
                # Arc = door swing; clear its full bounding box
                bbox = entity.bbox
                r_min, c_min, r_max, c_max = self.mapper.dxf_bbox_to_grid_bbox(
                    bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
                )
                for rr in range(r_min, r_max + 1):
                    for cc in range(c_min, c_max + 1):
                        touched_cells.add((rr, cc))

            else:
                # Fallback: use entity bounding box
                bbox = entity.bbox
                r_min, c_min, r_max, c_max = self.mapper.dxf_bbox_to_grid_bbox(
                    bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
                )
                for rr in range(r_min, r_max + 1):
                    for cc in range(c_min, c_max + 1):
                        touched_cells.add((rr, cc))

        if not touched_cells:
            # Fallback: use group bounding box
            bbox = group.bbox
            r_min, c_min, r_max, c_max = self.mapper.dxf_bbox_to_grid_bbox(
                bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
            )
            for rr in range(r_min, r_max + 1):
                for cc in range(c_min, c_max + 1):
                    touched_cells.add((rr, cc))

        # Expand with clearance and apply FREE
        for r, c in list(touched_cells):
            r_lo = max(1, r - clearance)
            r_hi = min(self.rows - 2, r + clearance)
            c_lo = max(1, c - clearance)
            c_hi = min(self.cols - 2, c + clearance)
            self.grid[r_lo:r_hi+1, c_lo:c_hi+1] = FREE
            self.door_cells[r_lo:r_hi+1, c_lo:c_hi+1] = True

    def _mask_annotation_group(self, group):
        """Force text/dimension bounding boxes to FREE to avoid them acting as obstacles."""
        for entity in group.entities:
            if entity.entity_type in ("TEXT", "MTEXT", "DIMENSION"):
                bbox = entity.bbox
                r_min, c_min, r_max, c_max = self.mapper.dxf_bbox_to_grid_bbox(
                    bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
                )
                # Ensure we don't overwrite the very edge boundaries
                r_min, r_max = max(1, r_min), min(self.rows - 2, r_max)
                c_min, c_max = max(1, c_min), min(self.cols - 2, c_max)
                self.grid[r_min:r_max+1, c_min:c_max+1] = FREE

    def _cells_on_line(self, r1: int, c1: int, r2: int, c2: int) -> list:
        """Return list of (row, col) cells on a Bresenham line."""
        temp = np.zeros((self.rows, self.cols), dtype=np.uint8)
        cv2.line(temp, (c1, r1), (c2, r2), 1, 1)
        positions = np.argwhere(temp > 0)
        return [(int(r), int(c)) for r, c in positions]

    def _mark_door_as_free(self, group):
        """
        Legacy method — kept for compatibility.
        Now simply calls _force_traversable_pass.
        """
        self._force_traversable_pass(group)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_grid_image(grid: np.ndarray, output_path: str):
    """
    Save the occupancy grid as a crisp 2-color PNG.
    FREE (0) → white, OBSTACLE (1) → black.
    """
    # Scale to 0-255
    img = ((1 - grid) * 255).astype(np.uint8)
    # Upscale for visibility
    scale = max(1, min(4, 800 // max(grid.shape)))
    if scale > 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, img)
    logger.info(f"[GridBuilder] Grid image saved: {output_path}")


def save_grid_numpy(grid: np.ndarray, output_path: str):
    """Save grid as NumPy .npy file for fast reloading."""
    np.save(output_path, grid)
    logger.info(f"[GridBuilder] Grid array saved: {output_path}")


def load_grid_numpy(input_path: str) -> np.ndarray:
    """Load a previously saved grid."""
    return np.load(input_path)


def build_occupancy_grid(
    parsed_dxf,
    classifications: dict,
    mapper,
    output_dir: str = "Results",
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
) -> tuple:
    """
    Convenience function: build grid + save image + save npy.

    Returns
    -------
    tuple : (grid, door_cells)
        grid      — binary occupancy grid (uint8, 0=FREE 1=OBSTACLE)
        door_cells — boolean mask of forcibly-cleared door cells
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = GridBuilder(parsed_dxf, classifications, mapper, safety_margin)
    grid = builder.build()
    door_cells = builder.door_cells

    save_grid_image(grid, str(output_dir / "occupancy_grid.png"))
    save_grid_numpy(grid, str(output_dir / "occupancy_grid.npy"))
    np.save(str(output_dir / "door_cells.npy"), door_cells)

    return grid, door_cells
