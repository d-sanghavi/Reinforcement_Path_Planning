"""
block_names.py
──────────────
Lookup tables: CAD block name patterns → symbol class.

These are compiled from common AutoCAD Architecture, ARCAT, RCP block libraries,
and general CAD convention, to enable fast name-based symbol classification
without any ML inference (Tier-1 classification, confidence = 1.0).

Symbol Classes:
  wall                — solid wall segment
  door                — swing/slide/folding door
  window              — glazed wall opening
  opening             — archway / no-door opening
  stairs              — staircase
  column              — structural pillar/column
  furniture_obstacle  — fixed furniture (chairs, tables, beds, etc.)
  annotation          — dimensions, text, title block, north arrow
  equipment           — plumbing, HVAC, electrical fixtures
  unknown             — unrecognized
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Block name pattern → symbol class
# Matching is case-insensitive prefix/substring matching.
# Entries are checked in order — more specific patterns first.
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_NAME_PATTERNS = [
    # ── Doors ────────────────────────────────────────────────────────────────
    ("door",              "door"),
    ("dr-",               "door"),
    ("dr_",               "door"),
    ("swing",             "door"),
    ("hinged",            "door"),
    ("sliding-door",      "door"),
    ("slid-dr",           "door"),
    ("folding-door",      "door"),
    ("bifold",            "door"),
    ("double-door",       "door"),
    ("dbl-dr",            "door"),
    ("revolving",         "door"),

    # ── Windows ──────────────────────────────────────────────────────────────
    ("window",            "window"),
    ("wnd",               "window"),
    ("win-",              "window"),
    ("win_",              "window"),
    ("casement",          "window"),
    ("awning",            "window"),
    ("skylight",          "window"),
    ("glazing",           "window"),

    # ── Stairs ───────────────────────────────────────────────────────────────
    ("stair",             "stairs"),
    ("strs",              "stairs"),
    ("step",              "stairs"),
    ("ladder",            "stairs"),
    ("escalator",         "stairs"),

    # ── Columns / Pillars ────────────────────────────────────────────────────
    ("column",            "column"),
    ("col-",              "column"),
    ("pillar",            "column"),
    ("pier",              "column"),
    ("post",              "column"),

    # ── Furniture / Obstacles ─────────────────────────────────────────────────
    ("chair",             "furniture_obstacle"),
    ("seat",              "furniture_obstacle"),
    ("sofa",              "furniture_obstacle"),
    ("couch",             "furniture_obstacle"),
    ("table",             "furniture_obstacle"),
    ("desk",              "furniture_obstacle"),
    ("workstation",       "furniture_obstacle"),
    ("counter",           "furniture_obstacle"),
    ("cabinet",           "furniture_obstacle"),
    ("shelv",             "furniture_obstacle"),
    ("shelf",             "furniture_obstacle"),
    ("bed",               "furniture_obstacle"),
    ("dresser",           "furniture_obstacle"),
    ("wardrobe",          "furniture_obstacle"),
    ("bookcase",          "furniture_obstacle"),

    # ── Plumbing / Bathroom Fixtures (obstacles) ──────────────────────────────
    ("toilet",            "furniture_obstacle"),
    ("wc",                "furniture_obstacle"),
    ("lavatory",          "furniture_obstacle"),
    ("bathtub",           "furniture_obstacle"),
    ("bath",              "furniture_obstacle"),
    ("shower",            "furniture_obstacle"),
    ("sink",              "furniture_obstacle"),
    ("basin",             "furniture_obstacle"),
    ("urinal",            "furniture_obstacle"),

    # ── Kitchen Fixtures ──────────────────────────────────────────────────────
    ("kitchen",           "furniture_obstacle"),
    ("cooktop",           "furniture_obstacle"),
    ("stove",             "furniture_obstacle"),
    ("oven",              "furniture_obstacle"),
    ("refrigerator",      "furniture_obstacle"),
    ("fridge",            "furniture_obstacle"),
    ("dishwasher",        "furniture_obstacle"),
    ("microwave",         "furniture_obstacle"),

    # ── HVAC / Mechanical ─────────────────────────────────────────────────────
    ("hvac",              "equipment"),
    ("duct",              "equipment"),
    ("vent",              "equipment"),
    ("grille",            "equipment"),
    ("diffuser",          "equipment"),
    ("radiator",          "equipment"),
    ("boiler",            "equipment"),
    ("ac-unit",           "equipment"),
    ("aircon",            "equipment"),

    # ── Electrical ────────────────────────────────────────────────────────────
    ("outlet",            "annotation"),
    ("switch",            "annotation"),
    ("panel",             "annotation"),
    ("elec",              "annotation"),
    ("socket",            "annotation"),

    # ── Annotations / Title Block ─────────────────────────────────────────────
    ("dim",               "annotation"),
    ("dimension",         "annotation"),
    ("text",              "annotation"),
    ("anno",              "annotation"),
    ("label",             "annotation"),
    ("title",             "annotation"),
    ("border",            "annotation"),
    ("titleblock",        "annotation"),
    ("north",             "annotation"),
    ("compass",           "annotation"),
    ("scalebar",          "annotation"),
    ("scale-bar",         "annotation"),
    ("revision",          "annotation"),
    ("cloud",             "annotation"),
    ("leader",            "annotation"),

    # ── Openings / Archways ───────────────────────────────────────────────────
    ("arch",              "opening"),
    ("opening",           "opening"),
    ("passage",           "opening"),
    ("portal",            "opening"),
    ("cased",             "opening"),
]

# Lower-cased set of block name substrings that strongly indicate annotations
ANNOTATION_LAYER_KEYWORDS = {
    "defpoints", "dims", "dimensions", "notes", "hatching",
    "title", "border", "anno", "text", "leade", "refplan",
}

# Layer names that typically carry walls
WALL_LAYER_KEYWORDS = {
    "wall", "walls", "a-wall", "a_wall", "boundary", "partition",
    "structure", "structural", "exterior", "interior",
}

# Layer names for doors
DOOR_LAYER_KEYWORDS = {
    "door", "doors", "a-door", "a_door", "opening", "a-glaz",
}

# Layer names for windows
WINDOW_LAYER_KEYWORDS = {
    "window", "windows", "a-wind", "a_wind", "glazing", "a-glaz",
}


# ─────────────────────────────────────────────────────────────────────────────
# Lookup Functions
# ─────────────────────────────────────────────────────────────────────────────

def classify_by_block_name(block_name: str) -> Optional[str]:
    """
    Match a DXF INSERT block name against the known pattern library.

    Returns the symbol class string, or None if no match found.
    Confidence is implicitly 1.0 for any match (deterministic).
    """
    if not block_name:
        return None

    name_lower = block_name.lower().strip()

    for pattern, symbol_class in BLOCK_NAME_PATTERNS:
        if pattern in name_lower:
            return symbol_class

    return None


def classify_by_layer(layer_name: str) -> Optional[str]:
    """
    Heuristic classification based on DXF layer name.
    Returns symbol class or None.
    """
    if not layer_name:
        return None

    layer_lower = layer_name.lower().strip()

    for kw in WALL_LAYER_KEYWORDS:
        if kw in layer_lower:
            return "wall"

    for kw in DOOR_LAYER_KEYWORDS:
        if kw in layer_lower:
            return "door"

    for kw in WINDOW_LAYER_KEYWORDS:
        if kw in layer_lower:
            return "window"

    for kw in ANNOTATION_LAYER_KEYWORDS:
        if kw in layer_lower:
            return "annotation"

    return None


def is_annotation_layer(layer_name: str) -> bool:
    """Return True if the layer is likely an annotation/non-geometric layer."""
    if not layer_name:
        return False
    layer_lower = layer_name.lower()
    return any(kw in layer_lower for kw in ANNOTATION_LAYER_KEYWORDS)


ALL_SYMBOL_CLASSES = [
    "wall",
    "door",
    "window",
    "opening",
    "stairs",
    "column",
    "furniture_obstacle",
    "annotation",
    "equipment",
    "unknown",
]

OBSTACLE_CLASSES = {"wall", "column", "furniture_obstacle", "stairs", "equipment"}
TRAVERSABLE_CLASSES = {"door", "opening"}
SKIP_CLASSES = {"annotation"}
