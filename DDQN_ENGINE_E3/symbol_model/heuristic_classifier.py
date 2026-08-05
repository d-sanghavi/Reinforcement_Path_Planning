"""
heuristic_classifier.py
────────────────────────
Tier-1 symbol classifier — deterministic, rule-based, no ML required.

Uses geometric signatures to classify DXF entity groups:
  - Block name lookup (confidence 1.0)
  - Layer name heuristics (confidence 0.8)
  - Geometric shape analysis (confidence 0.6–0.85)

This classifier is always run first. The CNN (Tier-2) is only invoked for
groups where Tier-1 confidence falls below a threshold.

Classification rules based on AutoCAD floor plan conventions:
  WALL   — long LINE / LWPOLYLINE, often parallel pairs
  DOOR   — ARC with ~90° sweep adjacent to a LINE endpoint
  WINDOW — short parallel perpendicular tick marks on a wall
  STAIRS — N parallel lines of equal length, equally spaced
  COLUMN — small closed square/circle (filled or hatched)
  FURNITURE_OBSTACLE — closed polyline with furniture-like aspect ratio
  ANNOTATION — TEXT, DIMENSION, LEADER, HATCH on annotation layers
"""

import logging
import math
from typing import Optional

from symbol_model.symbol_kb.block_names import (
    classify_by_block_name,
    classify_by_layer,
    is_annotation_layer,
    OBSTACLE_CLASSES,
)

logger = logging.getLogger(__name__)

# ── Classification confidence thresholds ─────────────────────────────────────
CONF_BLOCK_NAME = 1.00   # exact block name match
CONF_LAYER      = 0.80   # layer name heuristic
CONF_GEOMETRY   = 0.70   # geometric shape rule
CONF_FALLBACK   = 0.40   # best-guess from entity type alone

# ── Door arc sweep range (degrees) ───────────────────────────────────────────
DOOR_ARC_MIN_SWEEP = 75.0
DOOR_ARC_MAX_SWEEP = 105.0

# ── Wall geometry thresholds ──────────────────────────────────────────────────
WALL_MIN_LENGTH_RATIO = 5.0   # length/width >= 5 → likely a wall segment
WALL_MAX_ASPECT = 0.15        # bbox aspect ratio (height/width or w/h) <= 0.15

# ── Column geometry thresholds ────────────────────────────────────────────────
COLUMN_MAX_ASPECT_DEVIATION = 0.3  # max deviation from square (aspect ≈ 1.0)
COLUMN_MAX_AREA_FRACTION = 0.005   # column occupies < 0.5% of drawing area


class ClassificationResult:
    __slots__ = ("symbol_class", "confidence", "method", "group_id")

    def __init__(self, symbol_class: str, confidence: float, method: str, group_id: int = -1):
        self.symbol_class = symbol_class
        self.confidence = confidence
        self.method = method
        self.group_id = group_id

    def __repr__(self):
        return (
            f"ClassificationResult(class={self.symbol_class!r}, "
            f"conf={self.confidence:.2f}, method={self.method!r})"
        )


class HeuristicClassifier:
    """
    Rule-based symbol classifier for DXF entity groups.

    Parameters
    ----------
    drawing_area : float
        Total drawing area in DXF units² — used to normalize size thresholds.
    """

    def __init__(self, drawing_area: float = 1e6):
        self.drawing_area = max(drawing_area, 1.0)

    def classify_group(self, group) -> ClassificationResult:
        """
        Classify an EntityGroup from dxf_parser.py.

        Returns a ClassificationResult with class + confidence.
        """
        gid = group.group_id

        # ── 1. Block name lookup (highest priority) ────────────────────────
        if group.block_name:
            cls = classify_by_block_name(group.block_name)
            if cls:
                return ClassificationResult(cls, CONF_BLOCK_NAME, "block_name", gid)

        # ── 2. Collect entity type inventory ──────────────────────────────
        entity_types = {}
        layers = set()
        for ent in group.entities:
            entity_types[ent.entity_type] = entity_types.get(ent.entity_type, 0) + 1
            if ent.layer:
                layers.add(ent.layer)

        # ── 3. Annotation detection ────────────────────────────────────────
        # Entity type based
        if entity_types.get("TEXT", 0) + entity_types.get("MTEXT", 0) > 0:
            if "LINE" not in entity_types and "ARC" not in entity_types:
                return ClassificationResult("annotation", CONF_GEOMETRY, "text_entity", gid)

        if entity_types.get("DIMENSION", 0) > 0:
            return ClassificationResult("annotation", CONF_GEOMETRY, "dimension_entity", gid)

        # Layer-based annotation
        for layer in layers:
            if is_annotation_layer(layer):
                return ClassificationResult("annotation", CONF_LAYER, "annotation_layer", gid)

        # ── 4. Layer name classification ───────────────────────────────────
        for layer in layers:
            cls = classify_by_layer(layer)
            if cls:
                return ClassificationResult(cls, CONF_LAYER, f"layer:{layer}", gid)

        # ── 5. Geometric analysis ──────────────────────────────────────────

        # --- Door: ARC with ~90° sweep ----------------------------------------
        arcs = [e for e in group.entities if e.entity_type == "ARC"]
        for arc in arcs:
            swept = arc.extra.get("swept_angle", 0)
            if DOOR_ARC_MIN_SWEEP <= swept <= DOOR_ARC_MAX_SWEEP:
                return ClassificationResult("door", CONF_GEOMETRY, "door_arc_90deg", gid)

        # --- Wall: long thin rectangle or single long line --------------------
        lines = [e for e in group.entities if e.entity_type == "LINE"]
        long_lines = [l for l in lines if l.extra.get("length", 0) > 0]
        if long_lines and not arcs:
            # Check if the group bbox is very elongated
            bbox = group.bbox
            width = bbox.width
            height = bbox.height
            if width > 0 and height > 0:
                aspect = max(width, height) / min(width, height)
                if aspect >= WALL_MIN_LENGTH_RATIO:
                    return ClassificationResult("wall", CONF_GEOMETRY, "elongated_bbox", gid)

        # Check for LWPOLYLINE walls (long thin polylines)
        polylines = [e for e in group.entities if e.entity_type == "LWPOLYLINE"]
        for poly in polylines:
            bbox = poly.bbox
            if bbox.width > 0 and bbox.height > 0:
                aspect = max(bbox.width, bbox.height) / min(bbox.width, bbox.height)
                is_closed = poly.extra.get("is_closed", False)
                verts = poly.extra.get("vertex_count", 0)

                if aspect >= WALL_MIN_LENGTH_RATIO and not is_closed:
                    return ClassificationResult("wall", CONF_GEOMETRY, "polyline_wall", gid)

                # Closed rectangular polyline
                if is_closed and verts in (4, 5):
                    # Check if it's column-sized or furniture-sized
                    area_frac = (bbox.area / self.drawing_area)
                    aspect_from_sq = abs(bbox.aspect_ratio - 1.0)

                    if (area_frac < COLUMN_MAX_AREA_FRACTION and
                            aspect_from_sq < COLUMN_MAX_ASPECT_DEVIATION):
                        return ClassificationResult("column", CONF_GEOMETRY, "small_square_poly", gid)

                    if area_frac < 0.01:
                        return ClassificationResult("furniture_obstacle", 0.6, "closed_poly_obstacle", gid)

        # --- Stairs: multiple parallel equal-length lines --------------------
        if len(lines) >= 3 and not arcs:
            lengths = [l.extra.get("length", 0) for l in lines]
            angles = [l.extra.get("angle", 0) for l in lines]
            if lengths:
                mean_len = sum(lengths) / len(lengths)
                len_variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
                mean_ang = sum(angles) / len(angles)
                ang_variance = sum((x - mean_ang) ** 2 for x in angles) / len(angles)

                # Low variance in length and angle → parallel equal lines → stairs
                if len_variance < (mean_len * 0.1) ** 2 and ang_variance < 100:
                    if len(lines) >= 4:
                        return ClassificationResult("stairs", CONF_GEOMETRY, "parallel_equal_lines", gid)

        # --- Circle: column or opening ----------------------------------------
        circles = [e for e in group.entities if e.entity_type == "CIRCLE"]
        for circle in circles:
            r = circle.extra.get("radius", 0)
            area_frac = (math.pi * r * r) / self.drawing_area
            if area_frac < COLUMN_MAX_AREA_FRACTION * 2:
                return ClassificationResult("column", CONF_GEOMETRY, "circle_column", gid)

        # ── 6. Entity type fallback ────────────────────────────────────────
        dominant_type = max(entity_types, key=entity_types.get) if entity_types else "unknown"

        if dominant_type in ("TEXT", "MTEXT", "DIMENSION", "LEADER"):
            return ClassificationResult("annotation", CONF_FALLBACK, "entity_type_fallback", gid)

        if dominant_type == "HATCH":
            # Hatch on a small area → furniture/obstacle
            return ClassificationResult("furniture_obstacle", CONF_FALLBACK, "hatch_fallback", gid)

        if dominant_type in ("LINE", "LWPOLYLINE", "POLYLINE"):
            # Generic geometric → assume wall (most common floor plan element)
            return ClassificationResult("wall", CONF_FALLBACK * 0.8, "line_fallback", gid)

        return ClassificationResult("unknown", 0.1, "no_match", gid)

    def classify_all(self, entity_groups: list) -> dict:
        """
        Classify all entity groups from ParsedDXF.entity_groups.

        Returns
        -------
        dict : group_id → ClassificationResult
        """
        results = {}
        for group in entity_groups:
            result = self.classify_group(group)
            results[group.group_id] = result
            logger.debug(f"Group {group.group_id}: {result}")

        # Summary
        class_counts: dict = {}
        for r in results.values():
            class_counts[r.symbol_class] = class_counts.get(r.symbol_class, 0) + 1
        logger.info(f"[HeuristicClassifier] Summary: {class_counts}")

        return results

    def confidence_summary(self, results: dict) -> dict:
        """
        Return summary statistics about classification confidence.
        """
        confidences = [r.confidence for r in results.values()]
        if not confidences:
            return {}

        high = sum(1 for c in confidences if c >= 0.75)
        medium = sum(1 for c in confidences if 0.5 <= c < 0.75)
        low = sum(1 for c in confidences if c < 0.5)

        return {
            "total": len(confidences),
            "high_confidence": high,
            "medium_confidence": medium,
            "low_confidence_flagged": low,
            "mean_confidence": sum(confidences) / len(confidences),
            "needs_cnn_review": [
                gid for gid, r in results.items() if r.confidence < 0.5
            ],
        }
