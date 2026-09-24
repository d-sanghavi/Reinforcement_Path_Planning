"""
dxf_parser.py
─────────────
Parse a validated DXF file into structured entity groups using ezdxf.

Extracts:
  - Raw primitives (LINE, ARC, LWPOLYLINE, CIRCLE, SPLINE)
  - Block references (INSERT) recursively resolved
  - Geometric metadata (bounding boxes, centroids, angles)
  - Entity candidate groups (for symbol recognition input)

The output `ParsedDXF` object is the single contract between the ingestion
layer and the symbol recognition layer.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import ezdxf
from ezdxf import recover
from ezdxf.math import Vec3

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return float("inf")
        return self.width / self.height


@dataclass
class PrimitiveEntity:
    """A single DXF geometric entity with extracted properties."""
    handle: str
    entity_type: str          # LINE, ARC, LWPOLYLINE, CIRCLE, SPLINE, INSERT, etc.
    layer: str
    bbox: BoundingBox
    points: list              # Key geometry points [(x,y), ...]
    extra: dict = field(default_factory=dict)  # entity-type-specific data


@dataclass
class EntityGroup:
    """
    A cluster of spatially related PrimitiveEntity objects that
    likely form a single floor plan symbol (e.g., door + arc).
    """
    group_id: int
    entities: list            # list of PrimitiveEntity
    bbox: BoundingBox
    centroid: tuple           # (cx, cy)
    primary_type: str         # dominant entity type in the group
    block_name: Optional[str] = None  # if group originated from an INSERT block


@dataclass
class ParsedDXF:
    """Complete parsed representation of a DXF floor plan."""
    source_file: str
    dxf_version: str
    units: str
    units_code: int
    drawing_bounds: BoundingBox
    primitives: list          # all PrimitiveEntity objects
    entity_groups: list       # EntityGroup clusters for symbol recognition
    entity_counts: dict       # type → count summary
    warnings: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_dxf(dxf_path: str) -> ParsedDXF:
    """
    Parse a DXF file and return a structured ParsedDXF object.

    Parameters
    ----------
    dxf_path : str
        Path to a validated .dxf file.

    Returns
    -------
    ParsedDXF
        Fully parsed floor plan representation.
    """
    dxf_path = Path(dxf_path)
    logger.info(f"[DXF Parser] Parsing: {dxf_path}")

    warnings = []

    # Load document
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        logger.warning(f"Direct read failed, attempting recovery: {e}")
        try:
            doc, audit = recover.readfile(str(dxf_path))
            warnings.append(f"DXF recovered with {len(audit.errors)} repair operations")
        except Exception as e2:
            raise RuntimeError(f"Cannot parse DXF file: {e2}")

    msp = doc.modelspace()
    units_code = doc.header.get("$INSUNITS", 0)
    units_str = _insunits_to_str(units_code)
    dxf_version = doc.dxfversion

    logger.info(f"[DXF Parser] Version={dxf_version}, Units={units_str}")

    # ── Extract all primitives ─────────────────────────────────────────────────
    primitives: list = []
    _extract_entities(msp, doc, primitives, parent_block=None)

    # ── Entity count summary ───────────────────────────────────────────────────
    entity_counts: dict = {}
    for prim in primitives:
        entity_counts[prim.entity_type] = entity_counts.get(prim.entity_type, 0) + 1

    logger.info(f"[DXF Parser] Entity counts: {entity_counts}")

    if not primitives:
        warnings.append("WARNING: No parseable entities found in modelspace")

    # ── Overall drawing bounds ─────────────────────────────────────────────────
    drawing_bounds = _compute_overall_bbox(primitives)

    # ── Group nearby entities into candidate symbol clusters ───────────────────
    entity_groups = _cluster_entities(primitives, drawing_bounds)
    logger.info(f"[DXF Parser] Formed {len(entity_groups)} entity groups")

    return ParsedDXF(
        source_file=str(dxf_path),
        dxf_version=dxf_version,
        units=units_str,
        units_code=units_code,
        drawing_bounds=drawing_bounds,
        primitives=primitives,
        entity_groups=entity_groups,
        entity_counts=entity_counts,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION (recursive for block references)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_entities(
    space,
    doc,
    out: list,
    parent_block: Optional[str],
    depth: int = 0,
):
    """Recursively extract entities from modelspace and block definitions."""
    if depth > 8:  # guard against circular block references
        return

    for entity in space:
        etype = entity.dxftype()

        if etype == "INSERT":
            # Recursively resolve block reference
            block_name = entity.dxf.name
            try:
                block_def = doc.blocks[block_name]
                # Re-call with the block definition's modelspace
                sub_primitives: list = []
                _extract_entities(
                    block_def, doc, sub_primitives, parent_block=block_name, depth=depth + 1
                )
                # Translate sub-primitives by the INSERT insertion point
                insert_pt = entity.dxf.insert
                scale_x = getattr(entity.dxf, "xscale", 1.0)
                scale_y = getattr(entity.dxf, "yscale", 1.0)
                rotation = getattr(entity.dxf, "rotation", 0.0)
                for prim in sub_primitives:
                    prim = _transform_primitive(prim, insert_pt, scale_x, scale_y, rotation)
                    prim.extra["block_name"] = block_name
                    out.append(prim)
            except (ezdxf.DXFKeyError, KeyError):
                logger.debug(f"Block '{block_name}' definition not found, skipping")
            # Also record the INSERT itself as a primitive
            prim = _parse_insert(entity, parent_block)
            if prim:
                out.append(prim)

        elif etype == "LINE":
            prim = _parse_line(entity)
            if prim:
                out.append(prim)

        elif etype == "ARC":
            prim = _parse_arc(entity)
            if prim:
                out.append(prim)

        elif etype in ("LWPOLYLINE", "POLYLINE"):
            prim = _parse_polyline(entity, etype)
            if prim:
                out.append(prim)

        elif etype == "CIRCLE":
            prim = _parse_circle(entity)
            if prim:
                out.append(prim)

        elif etype == "SPLINE":
            prim = _parse_spline(entity)
            if prim:
                out.append(prim)

        elif etype in ("TEXT", "MTEXT"):
            prim = _parse_text(entity, etype)
            if prim:
                out.append(prim)

        elif etype in ("DIMENSION", "LEADER", "HATCH"):
            # Record but mark as annotation
            prim = _parse_annotation(entity, etype)
            if prim:
                out.append(prim)

        # Other entity types (SOLID, TRACE, etc.) — record basic bbox
        else:
            prim = _parse_generic(entity, etype)
            if prim:
                out.append(prim)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY-SPECIFIC PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_line(entity) -> Optional[PrimitiveEntity]:
    try:
        s = entity.dxf.start
        e = entity.dxf.end
        pts = [(s.x, s.y), (e.x, e.y)]
        bbox = _pts_to_bbox(pts)
        length = math.hypot(e.x - s.x, e.y - s.y)
        angle = math.degrees(math.atan2(e.y - s.y, e.x - s.x)) % 180
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type="LINE",
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=pts,
            extra={"length": length, "angle": angle},
        )
    except Exception as ex:
        logger.debug(f"LINE parse error: {ex}")
        return None


def _parse_arc(entity) -> Optional[PrimitiveEntity]:
    try:
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle

        # Normalize end > start
        if end_angle < start_angle:
            end_angle += 360

        swept = end_angle - start_angle

        # Sample arc points for bbox
        n_pts = max(8, int(swept / 10))
        pts = []
        for i in range(n_pts + 1):
            a = math.radians(start_angle + swept * i / n_pts)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

        bbox = _pts_to_bbox(pts)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type="ARC",
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=pts,
            extra={
                "center": (cx, cy),
                "radius": r,
                "start_angle": start_angle,
                "end_angle": end_angle % 360,
                "swept_angle": swept,
            },
        )
    except Exception as ex:
        logger.debug(f"ARC parse error: {ex}")
        return None


def _parse_polyline(entity, etype: str) -> Optional[PrimitiveEntity]:
    try:
        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
        else:  # POLYLINE
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]

        if not pts:
            return None

        is_closed = entity.is_closed if hasattr(entity, "is_closed") else False
        bbox = _pts_to_bbox(pts)

        # Compute perimeter
        perimeter = 0.0
        for i in range(len(pts) - 1):
            perimeter += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if is_closed and len(pts) > 2:
            perimeter += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])

        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type="LWPOLYLINE",
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=pts,
            extra={"is_closed": is_closed, "vertex_count": len(pts), "perimeter": perimeter},
        )
    except Exception as ex:
        logger.debug(f"POLYLINE parse error: {ex}")
        return None


def _parse_circle(entity) -> Optional[PrimitiveEntity]:
    try:
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        pts = [
            (cx - r, cy - r), (cx + r, cy - r),
            (cx + r, cy + r), (cx - r, cy + r),
        ]
        bbox = _pts_to_bbox(pts)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type="CIRCLE",
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=[(cx, cy)],
            extra={"center": (cx, cy), "radius": r, "area": math.pi * r * r},
        )
    except Exception as ex:
        logger.debug(f"CIRCLE parse error: {ex}")
        return None


def _parse_spline(entity) -> Optional[PrimitiveEntity]:
    try:
        pts = [(p[0], p[1]) for p in entity.control_points]
        if not pts:
            return None
        bbox = _pts_to_bbox(pts)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type="SPLINE",
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=pts,
            extra={"degree": entity.dxf.degree},
        )
    except Exception as ex:
        logger.debug(f"SPLINE parse error: {ex}")
        return None


def _parse_insert(entity, parent_block: Optional[str]) -> Optional[PrimitiveEntity]:
    try:
        ip = entity.dxf.insert
        block_name = entity.dxf.name
        pts = [(ip.x, ip.y)]
        bbox = BoundingBox(ip.x - 0.1, ip.y - 0.1, ip.x + 0.1, ip.y + 0.1)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type="INSERT",
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=pts,
            extra={
                "block_name": block_name,
                "rotation": getattr(entity.dxf, "rotation", 0.0),
                "xscale": getattr(entity.dxf, "xscale", 1.0),
                "yscale": getattr(entity.dxf, "yscale", 1.0),
            },
        )
    except Exception as ex:
        logger.debug(f"INSERT parse error: {ex}")
        return None


def _parse_text(entity, etype: str) -> Optional[PrimitiveEntity]:
    try:
        if etype == "TEXT":
            ip = entity.dxf.insert
            pts = [(ip.x, ip.y)]
            text_str = entity.dxf.text
        else:  # MTEXT
            ip = entity.dxf.insert
            pts = [(ip.x, ip.y)]
            text_str = entity.plain_text() if hasattr(entity, "plain_text") else ""

        bbox = BoundingBox(ip.x, ip.y, ip.x + 1, ip.y + 1)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type=etype,
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=pts,
            extra={"text": text_str[:100]},
        )
    except Exception as ex:
        logger.debug(f"TEXT parse error: {ex}")
        return None


def _parse_annotation(entity, etype: str) -> Optional[PrimitiveEntity]:
    try:
        # Use a dummy bbox at origin — annotations are filtered by class
        bbox = BoundingBox(0, 0, 0, 0)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type=etype,
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=[],
            extra={"annotation": True},
        )
    except Exception:
        return None


def _parse_generic(entity, etype: str) -> Optional[PrimitiveEntity]:
    """Fallback parser for unrecognized entity types."""
    try:
        bbox = BoundingBox(0, 0, 0, 0)
        return PrimitiveEntity(
            handle=entity.dxf.handle or "",
            entity_type=etype,
            layer=entity.dxf.layer or "0",
            bbox=bbox,
            points=[],
            extra={},
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY CLUSTERING (proximity-based grouping for symbol recognition)
# ═══════════════════════════════════════════════════════════════════════════════

def _cluster_entities(primitives: list, drawing_bounds: BoundingBox) -> list:
    """
    Group spatially nearby entities into candidate symbol clusters using
    a simple grid-cell proximity approach (faster than DBSCAN for large files).

    Entities within ~5% of the drawing diagonal of each other are grouped.
    """
    if not primitives:
        return []

    diag = math.hypot(drawing_bounds.width, drawing_bounds.height)
    cluster_radius = max(diag * 0.03, 0.5)  # 3% of drawing diagonal, min 0.5 units

    # Simple union-find clustering by centroid proximity
    centroids = [p.bbox.center for p in primitives]
    n = len(primitives)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Only cluster non-annotation entities that are close together
    geometric_types = {"LINE", "ARC", "LWPOLYLINE", "CIRCLE", "SPLINE", "INSERT"}
    for i in range(n):
        if primitives[i].entity_type not in geometric_types:
            continue
        cx_i, cy_i = centroids[i]
        for j in range(i + 1, n):
            if primitives[j].entity_type not in geometric_types:
                continue
            cx_j, cy_j = centroids[j]
            dist = math.hypot(cx_j - cx_i, cy_j - cy_i)
            if dist < cluster_radius:
                union(i, j)

    # Collect groups
    groups_map: dict = {}
    for i, prim in enumerate(primitives):
        root = find(i)
        groups_map.setdefault(root, []).append(prim)

    entity_groups = []
    for gid, (root, members) in enumerate(groups_map.items()):
        group_bbox = _merge_bboxes([m.bbox for m in members])
        cx = (group_bbox.min_x + group_bbox.max_x) / 2
        cy = (group_bbox.min_y + group_bbox.max_y) / 2

        # Find dominant entity type
        type_counts: dict = {}
        block_name = None
        for m in members:
            type_counts[m.entity_type] = type_counts.get(m.entity_type, 0) + 1
            if m.entity_type == "INSERT" and "block_name" in m.extra:
                block_name = m.extra["block_name"]

        primary_type = max(type_counts, key=type_counts.get)

        entity_groups.append(EntityGroup(
            group_id=gid,
            entities=members,
            bbox=group_bbox,
            centroid=(cx, cy),
            primary_type=primary_type,
            block_name=block_name,
        ))

    return entity_groups


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _pts_to_bbox(pts: list) -> BoundingBox:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def _compute_overall_bbox(primitives: list) -> BoundingBox:
    if not primitives:
        return BoundingBox(0, 0, 100, 100)
    all_pts = []
    for p in primitives:
        all_pts.extend(p.points)
    if not all_pts:
        return BoundingBox(0, 0, 100, 100)
    return _pts_to_bbox(all_pts)


def _merge_bboxes(bboxes: list) -> BoundingBox:
    return BoundingBox(
        min(b.min_x for b in bboxes),
        min(b.min_y for b in bboxes),
        max(b.max_x for b in bboxes),
        max(b.max_y for b in bboxes),
    )


def _transform_primitive(
    prim: PrimitiveEntity,
    insert_pt,
    scale_x: float,
    scale_y: float,
    rotation_deg: float,
) -> PrimitiveEntity:
    """Apply INSERT transformation (translate + scale + rotate) to a primitive."""
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def transform_pt(px, py):
        # Scale
        px *= scale_x
        py *= scale_y
        # Rotate
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        # Translate
        return (rx + insert_pt.x, ry + insert_pt.y)

    new_pts = [transform_pt(px, py) for px, py in prim.points]
    new_bbox = _pts_to_bbox(new_pts) if new_pts else prim.bbox

    return PrimitiveEntity(
        handle=prim.handle,
        entity_type=prim.entity_type,
        layer=prim.layer,
        bbox=new_bbox,
        points=new_pts,
        extra=prim.extra.copy(),
    )


def _insunits_to_str(code: int) -> str:
    UNITS = {
        0: "unitless", 1: "inches", 2: "feet", 3: "miles",
        4: "millimeters", 5: "centimeters", 6: "meters",
        7: "kilometers", 8: "microinches", 9: "mils",
        10: "yards", 14: "decimeters",
    }
    return UNITS.get(code, f"unknown({code})")
