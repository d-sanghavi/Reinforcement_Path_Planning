"""
infer_symbols.py
─────────────────
Unified symbol inference: Tier-1 heuristic → Tier-2 CNN fallback.

Pipeline:
  1. Run HeuristicClassifier on ALL entity groups → fast, deterministic
  2. Identify low-confidence groups (confidence < threshold)
  3. Run SymbolCNN on low-confidence groups → refine classifications
  4. Merge results and cache to parsed_symbols.json

Usage:
    from symbol_model.infer_symbols import classify_all_symbols
    results = classify_all_symbols(parsed_dxf, weights_path="models/symbol_cnn.pth")
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Groups below this threshold get CNN review
CNN_CONFIDENCE_THRESHOLD = 0.55


def classify_all_symbols(
    parsed_dxf,
    weights_path: str = "models/symbol_cnn.pth",
    cache_path: Optional[str] = None,
    force_reclassify: bool = False,
) -> dict:
    """
    Classify all entity groups in a parsed DXF file.

    Parameters
    ----------
    parsed_dxf : ParsedDXF
        Output from cad_ingestion.dxf_parser.parse_dxf()
    weights_path : str
        Path to pre-trained SymbolCNN weights (optional — falls back to Tier-1 only)
    cache_path : str, optional
        If provided, cache classification results to this JSON file.
        On re-runs, load from cache if it exists (unless force_reclassify=True).
    force_reclassify : bool
        If True, ignore any cached results.

    Returns
    -------
    dict : group_id → {
        "class": str,
        "confidence": float,
        "method": str,
        "entities": [handle, ...],
        "bbox": {...}
    }
    """
    # ── Load from cache if available ──────────────────────────────────────────
    if cache_path and Path(cache_path).exists() and not force_reclassify:
        logger.info(f"[SymbolInfer] Loading cached classifications from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    # ── Step 1: Tier-1 Heuristic Classification ───────────────────────────────
    from symbol_model.heuristic_classifier import HeuristicClassifier

    drawing_area = (
        parsed_dxf.drawing_bounds.width * parsed_dxf.drawing_bounds.height
    )
    classifier = HeuristicClassifier(drawing_area=drawing_area)
    heuristic_results = classifier.classify_all(parsed_dxf.entity_groups)

    conf_summary = classifier.confidence_summary(heuristic_results)
    logger.info(
        f"[SymbolInfer] Tier-1 results: "
        f"{conf_summary.get('high_confidence', 0)} high-conf, "
        f"{conf_summary.get('low_confidence_flagged', 0)} flagged for CNN review"
    )

    # ── Step 2: Tier-2 CNN Classification for low-confidence groups ───────────
    low_conf_ids = conf_summary.get("needs_cnn_review", [])
    cnn_model = None

    if low_conf_ids:
        try:
            from symbol_model.cnn_model import load_model, predict_group
            cnn_model = load_model(weights_path)
        except ImportError:
            logger.warning("[SymbolInfer] PyTorch not available — running Tier-1 only")
        except Exception as e:
            logger.warning(f"[SymbolInfer] CNN load error: {e} — running Tier-1 only")

    cnn_upgrades = 0
    if cnn_model:
        from symbol_model.cnn_model import predict_group
        group_by_id = {g.group_id: g for g in parsed_dxf.entity_groups}

        for gid in low_conf_ids:
            group = group_by_id.get(gid)
            if group is None:
                continue
            try:
                cnn_class, cnn_conf = predict_group(cnn_model, group)
                tier1_result = heuristic_results[gid]

                # Only accept CNN result if it has higher confidence
                if cnn_conf > tier1_result.confidence:
                    tier1_result.symbol_class = cnn_class
                    tier1_result.confidence = cnn_conf
                    tier1_result.method = f"cnn:{tier1_result.method}"
                    cnn_upgrades += 1
            except Exception as e:
                logger.debug(f"[SymbolInfer] CNN inference error for group {gid}: {e}")

        logger.info(f"[SymbolInfer] CNN upgraded {cnn_upgrades}/{len(low_conf_ids)} low-conf groups")

    # ── Step 3: Build output dict ─────────────────────────────────────────────
    group_by_id = {g.group_id: g for g in parsed_dxf.entity_groups}
    output = {}

    for gid, result in heuristic_results.items():
        group = group_by_id.get(gid)
        bbox = group.bbox if group else None

        output[str(gid)] = {
            "class": result.symbol_class,
            "confidence": round(result.confidence, 4),
            "method": result.method,
            "entities": [e.handle for e in (group.entities if group else [])],
            "bbox": {
                "min_x": bbox.min_x if bbox else 0,
                "min_y": bbox.min_y if bbox else 0,
                "max_x": bbox.max_x if bbox else 0,
                "max_y": bbox.max_y if bbox else 0,
            } if bbox else {},
            "block_name": group.block_name if group else None,
        }

    # ── Step 4: Cache results ─────────────────────────────────────────────────
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"[SymbolInfer] Classifications cached to {cache_path}")

    # Print summary
    class_counts: dict = {}
    for v in output.values():
        c = v["class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    logger.info(f"[SymbolInfer] Final classification summary: {class_counts}")

    return output


def get_classified_entities_by_class(classifications: dict, target_class: str) -> list:
    """
    Filter classification results for a specific symbol class.

    Returns list of classification dicts with the target class.
    """
    return [
        {"group_id": int(gid), **data}
        for gid, data in classifications.items()
        if data["class"] == target_class
    ]


def get_confidence_report(classifications: dict) -> dict:
    """Return a human-readable confidence breakdown."""
    confidences = [v["confidence"] for v in classifications.values()]
    if not confidences:
        return {}

    methods = {}
    for v in classifications.values():
        m = v["method"].split(":")[0]  # get primary method
        methods[m] = methods.get(m, 0) + 1

    return {
        "total_groups": len(confidences),
        "high_confidence_>=0.75": sum(1 for c in confidences if c >= 0.75),
        "medium_confidence_0.5-0.75": sum(1 for c in confidences if 0.5 <= c < 0.75),
        "low_confidence_<0.5": sum(1 for c in confidences if c < 0.5),
        "mean_confidence": round(sum(confidences) / len(confidences), 3),
        "min_confidence": round(min(confidences), 3),
        "classification_methods": methods,
    }
