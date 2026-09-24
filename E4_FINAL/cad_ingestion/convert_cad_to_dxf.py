"""
convert_cad_to_dxf.py
─────────────────────
Multi-format CAD file normalizer. Accepts DXF, DWG, and any proprietary CAD
format and produces a clean DXF R2013 ASCII file suitable for ezdxf parsing.

Conversion cascade (all local, zero external APIs):
  1. ezdxf direct read   — fastest, works on valid DXF files
  2. ezdxf recover mode  — handles malformed/corrupt DXF files
  3. ODA File Converter  — handles DWG/DWT/DXB (free local binary)
  4. LibreCAD CLI        — fallback for ODA-unsupported variants
  5. FreeCAD Python API  — final fallback (if FreeCAD installed)

Usage:
    from cad_ingestion.convert_cad_to_dxf import normalize_to_dxf
    dxf_path = normalize_to_dxf("floor_plan.dwg", output_dir="./tmp")
"""

import os
import sys
import subprocess
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional

import ezdxf
from ezdxf import recover

logger = logging.getLogger(__name__)

# ── DXF version target for maximum ezdxf compatibility ──────────────────────
TARGET_DXF_VERSION = "R2013"

# ── Common install locations for ODA File Converter (Windows / Linux / macOS) ─
ODA_SEARCH_PATHS = [
    r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
    r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
    "/usr/bin/ODAFileConverter",
    "/usr/local/bin/ODAFileConverter",
    "/opt/ODAFileConverter/ODAFileConverter",
]

# ── Common install locations for LibreCAD CLI ────────────────────────────────
LIBRECAD_SEARCH_PATHS = [
    r"C:\Program Files\LibreCAD\LibreCAD.exe",
    r"C:\Program Files (x86)\LibreCAD\LibreCAD.exe",
    "/usr/bin/librecad",
    "/usr/local/bin/librecad",
    "/Applications/LibreCAD.app/Contents/MacOS/LibreCAD",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_to_dxf(input_path: str, output_dir: str = "./tmp") -> str:
    """
    Accept ANY CAD file format and return the path to a valid DXF file.

    Parameters
    ----------
    input_path : str
        Path to input file (.pdf, .dxf, .dwg, .cad, .dxt, .dxb, .dwt, etc.)
    output_dir : str
        Directory where converted DXF will be written (created if absent).

    Returns
    -------
    str
        Absolute path to the normalized DXF file.

    Raises
    ------
    RuntimeError
        If all conversion strategies fail. Error message includes specific
        failure details from each attempted converter.
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_path.suffix.lower()
    stem = input_path.stem
    output_dxf = output_dir / f"{stem}_normalized.dxf"

    logger.info(f"[CAD Ingestion] Input: {input_path}  Extension: {ext}")

    errors = []

    # ── Strategy 0: PDF → DXF (local, zero paid APIs) ───────────────────────
    if ext == ".pdf":
        try:
            from cad_ingestion.pdf_to_grid import pdf_to_dxf
            result = pdf_to_dxf(str(input_path), output_dir=str(output_dir))
            logger.info(f"[Strategy 0] PDF→DXF conversion success: {result}")
            # Now normalize the resulting DXF through strategy 1/2
            pdf_dxf_path = Path(result)
            dxf_result = _try_ezdxf_direct(pdf_dxf_path, output_dxf)
            if dxf_result:
                return str(dxf_result)
            return str(result)  # return raw DXF if normalization fails
        except Exception as e:
            errors.append(f"PDF-to-DXF: {e}")
            logger.warning(f"[Strategy 0] PDF conversion failed: {e}")

    # ── Strategy 1: ezdxf direct read (works if already a valid DXF) ─────────
    if ext in (".dxf", ".dxt"):
        result = _try_ezdxf_direct(input_path, output_dxf)
        if result:
            return str(result)
        errors.append("ezdxf-direct: failed to parse as valid DXF")

        # ── Strategy 2: ezdxf recover (malformed/old DXF) ───────────────────
        result = _try_ezdxf_recover(input_path, output_dxf)
        if result:
            return str(result)
        errors.append("ezdxf-recover: recovery mode also failed")

    # ── Strategy 3: ODA File Converter (handles DWG, DWT, DXB, old DXF) ─────
    result = _try_oda_converter(input_path, output_dir, output_dxf)
    if result:
        return str(result)
    errors.append("ODA-converter: not found or conversion failed")

    # ── Strategy 4: LibreCAD headless CLI ────────────────────────────────────
    result = _try_librecad(input_path, output_dxf)
    if result:
        return str(result)
    errors.append("LibreCAD-CLI: not found or conversion failed")

    # ── Strategy 5: FreeCAD Python API ───────────────────────────────────────
    result = _try_freecad(input_path, output_dxf)
    if result:
        return str(result)
    errors.append("FreeCAD-API: not found or conversion failed")

    # ── All strategies failed ─────────────────────────────────────────────────
    error_detail = "\n  → ".join(errors)
    raise RuntimeError(
        f"Could not convert '{input_path}' to DXF.\n"
        f"All conversion strategies failed:\n  → {error_detail}\n\n"
        f"Recommended fix:\n"
        f"  • For DWG files: install ODA File Converter (free) from "
        f"https://www.opendesign.com/guestfiles/oda_file_converter\n"
        f"  • For DXF files: verify the file opens in AutoCAD/LibreCAD"
    )


def validate_dxf(dxf_path: str) -> dict:
    """
    Open and validate a DXF file. Returns a sanity report dict.

    Returns
    -------
    dict with keys:
        valid (bool), entity_counts (dict), units (str), version (str),
        warnings (list[str])
    """
    dxf_path = Path(dxf_path)
    report = {
        "valid": False,
        "entity_counts": {},
        "units": "unknown",
        "version": "unknown",
        "warnings": [],
    }

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as e:
        try:
            doc, audit = recover.readfile(str(dxf_path))
            report["warnings"].append(f"Recovered with {len(audit.errors)} errors")
        except Exception as e2:
            report["warnings"].append(str(e2))
            return report

    msp = doc.modelspace()

    # Count entities by type
    counts: dict = {}
    for entity in msp:
        etype = entity.dxftype()
        counts[etype] = counts.get(etype, 0) + 1

    # Recurse into block references
    for insert in msp.query("INSERT"):
        block_name = insert.dxf.name
        counts.setdefault("INSERT_BLOCKS", {})[block_name] = (
            counts.get("INSERT_BLOCKS", {}).get(block_name, 0) + 1
        )

    report["entity_counts"] = counts
    report["version"] = doc.dxfversion

    # Extract units from header
    try:
        insunits = doc.header.get("$INSUNITS", 0)
        report["units"] = _insunits_to_str(insunits)
    except Exception:
        report["warnings"].append("Could not read $INSUNITS header variable")

    report["valid"] = True
    logger.info(f"[DXF Validation] {dxf_path.name}: {report}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE CONVERSION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def _try_ezdxf_direct(input_path: Path, output_dxf: Path) -> Optional[Path]:
    """Attempt to read DXF with ezdxf and re-save as R2013."""
    try:
        doc = ezdxf.readfile(str(input_path))
        doc.saveas(str(output_dxf))          # saveas() takes path; save() does not
        logger.info(f"[Strategy 1] ezdxf direct read success: {output_dxf}")
        return output_dxf
    except Exception as e:
        logger.debug(f"[Strategy 1] ezdxf direct failed: {e}")
        return None


def _try_ezdxf_recover(input_path: Path, output_dxf: Path) -> Optional[Path]:
    """Use ezdxf recover mode for malformed DXF files."""
    try:
        doc, audit = recover.readfile(str(input_path))
        if audit.errors:
            logger.warning(
                f"[Strategy 2] DXF recovered with {len(audit.errors)} errors — "
                "some geometry may be lost"
            )
        doc.saveas(str(output_dxf))          # saveas() takes path; save() does not
        logger.info(f"[Strategy 2] ezdxf recover success: {output_dxf}")
        return output_dxf
    except Exception as e:
        logger.debug(f"[Strategy 2] ezdxf recover failed: {e}")
        return None


def _try_oda_converter(
    input_path: Path, output_dir: Path, output_dxf: Path
) -> Optional[Path]:
    """
    Use ODA File Converter CLI (free, local).
    CLI signature: ODAFileConverter <input_dir> <output_dir> <out_version>
                                    <out_format> <recurse> <audit> [filter]
    """
    oda_exe = _find_executable(ODA_SEARCH_PATHS, "ODAFileConverter")
    if not oda_exe:
        logger.debug("[Strategy 3] ODA File Converter not found in PATH or search paths")
        return None

    try:
        tmp_input_dir = tempfile.mkdtemp(prefix="oda_input_")
        tmp_output_dir = tempfile.mkdtemp(prefix="oda_output_")

        # ODA requires the file in its own directory
        shutil.copy2(str(input_path), tmp_input_dir)

        cmd = [
            str(oda_exe),
            tmp_input_dir,
            tmp_output_dir,
            TARGET_DXF_VERSION,
            "DXF",          # output format
            "0",            # recurse subdirs: no
            "1",            # audit: yes
            f"*.{input_path.suffix[1:]}",  # filter by extension
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        logger.debug(f"[ODA] stdout: {result.stdout}  stderr: {result.stderr}")

        # Find the converted file
        converted = list(Path(tmp_output_dir).glob("*.dxf"))
        if converted:
            shutil.copy2(str(converted[0]), str(output_dxf))
            logger.info(f"[Strategy 3] ODA conversion success: {output_dxf}")
            return output_dxf
        else:
            logger.debug("[Strategy 3] ODA ran but produced no DXF output")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("[Strategy 3] ODA File Converter timed out (>120s)")
        return None
    except Exception as e:
        logger.debug(f"[Strategy 3] ODA conversion error: {e}")
        return None


def _try_librecad(input_path: Path, output_dxf: Path) -> Optional[Path]:
    """
    Use LibreCAD headless CLI to convert to DXF.
    LibreCAD CLI: librecad dxf2dxf -o <output> <input>
    """
    librecad_exe = _find_executable(LIBRECAD_SEARCH_PATHS, "librecad")
    if not librecad_exe:
        logger.debug("[Strategy 4] LibreCAD not found")
        return None

    try:
        cmd = [
            str(librecad_exe),
            "dxf2dxf",
            "-o", str(output_dxf),
            str(input_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if output_dxf.exists() and output_dxf.stat().st_size > 0:
            logger.info(f"[Strategy 4] LibreCAD conversion success: {output_dxf}")
            return output_dxf
        else:
            logger.debug(
                f"[Strategy 4] LibreCAD failed: {result.stderr}"
            )
            return None
    except Exception as e:
        logger.debug(f"[Strategy 4] LibreCAD error: {e}")
        return None


def _try_freecad(input_path: Path, output_dxf: Path) -> Optional[Path]:
    """
    Use FreeCAD's Python API via subprocess to import and export DXF.
    FreeCAD must be installed and importable.
    FreeCAD and importDXF are optional runtime dependencies — not resolvable
    by the static analyser, hence the type: ignore directives.
    """
    try:
        freecad_paths = [
            r"C:\Program Files\FreeCAD 0.21\bin",
            r"C:\Program Files\FreeCAD 0.20\bin",
            "/usr/lib/freecad/lib",
            "/usr/local/lib/freecad/lib",
        ]
        for fp in freecad_paths:
            if Path(fp).exists() and fp not in sys.path:
                sys.path.insert(0, fp)

        import FreeCAD     # type: ignore[import-not-found]  # optional runtime dep
        import importDXF   # type: ignore[import-not-found]  # optional runtime dep

        doc = FreeCAD.newDocument()                           # type: ignore[attr-defined]
        importDXF.insert(str(input_path), doc.Name)           # type: ignore[attr-defined]
        importDXF.export(doc.Objects, str(output_dxf))        # type: ignore[attr-defined]
        if output_dxf.exists():
            logger.info(f"[Strategy 5] FreeCAD conversion success: {output_dxf}")
            return output_dxf
        return None
    except ImportError:
        logger.debug("[Strategy 5] FreeCAD not available")
        return None
    except Exception as e:
        logger.debug(f"[Strategy 5] FreeCAD error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _find_executable(search_paths: list, name: str) -> Optional[Path]:
    """Check PATH first, then known install directories."""
    # Check system PATH
    which = shutil.which(name)
    if which:
        return Path(which)

    # Check known install locations
    for p in search_paths:
        candidate = Path(p)
        if candidate.exists():
            return candidate

    return None


def _insunits_to_str(code: int) -> str:
    """Convert DXF $INSUNITS integer code to human-readable unit string."""
    UNITS = {
        0: "unitless",
        1: "inches",
        2: "feet",
        3: "miles",
        4: "millimeters",
        5: "centimeters",
        6: "meters",
        7: "kilometers",
        8: "microinches",
        9: "mils",
        10: "yards",
        11: "angstroms",
        12: "nanometers",
        13: "microns",
        14: "decimeters",
        15: "decameters",
        16: "hectometers",
        17: "gigameters",
        18: "astronomical_units",
        19: "light_years",
        20: "parsecs",
    }
    return UNITS.get(code, f"unknown({code})")
