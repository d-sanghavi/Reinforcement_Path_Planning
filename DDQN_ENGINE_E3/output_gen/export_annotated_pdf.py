"""
export_annotated_pdf.py
───────────────────────
Generate a high-resolution B/W PDF of the floor plan with the
computed path overlaid in red.

Pipeline:
  1. Render the DXF to a matplotlib figure (B/W — walls black, background white)
     using ezdxf's drawing add-on (matplotlib backend)
  2. Overlay the DDQN/A* path as a red polyline using DXF real-world coordinates
  3. Mark start (green dot) and goal (red star) on the overlay
  4. Export as a single PDF via matplotlib's PDF backend

Output: annotated_floor_plan.pdf
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

logger = logging.getLogger(__name__)


def export_annotated_pdf(
    dxf_path: str,
    path_dxf_coords: list,
    mapper,
    output_path: str = "Results/annotated_floor_plan.pdf",
    path_color: str = "#e63946",  # vivid red
    path_linewidth: float = 2.5,
    dpi: int = 200,
) -> str:
    """
    Generate the annotated floor plan PDF.

    Parameters
    ----------
    dxf_path : str
        Path to the source DXF file.
    path_dxf_coords : list
        List of (dxf_x, dxf_y) tuples (from export_coordinates.py).
    mapper : ScaleMapper
        Used to get start/goal DXF coordinates.
    output_path : str
        Output PDF file path.
    path_color : str
        Matplotlib color for path line.
    path_linewidth : float
        Path line width in points.
    dpi : int
        Output resolution.

    Returns
    -------
    str : absolute path to the generated PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[PDFExport] Generating annotated PDF: {output_path}")

    # ── Render DXF to figure ──────────────────────────────────────────────────
    fig, ax = _render_dxf_to_figure(dxf_path, dpi=dpi)

    if fig is None:
        # Fallback: render the occupancy grid instead
        logger.warning("[PDFExport] DXF rendering failed, using grid fallback")
        fig, ax = _fallback_grid_figure(mapper)

    # ── Overlay the path ──────────────────────────────────────────────────────
    if len(path_dxf_coords) >= 2:
        xs = [c[0] for c in path_dxf_coords]
        ys = [c[1] for c in path_dxf_coords]

        ax.plot(
            xs, ys,
            color=path_color,
            linewidth=path_linewidth,
            linestyle="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=10,
            label="Optimal Path (DDQN/A*)",
        )

        # Start marker
        ax.plot(
            xs[0], ys[0],
            marker="o", markersize=10,
            color="#2dc653", markeredgecolor="white", markeredgewidth=1.5,
            zorder=11, label="Start",
        )

        # Goal marker
        ax.plot(
            xs[-1], ys[-1],
            marker="*", markersize=14,
            color="#e63946", markeredgecolor="white", markeredgewidth=1.0,
            zorder=11, label="Goal",
        )

        # Add direction arrows along path
        n_arrows = min(8, len(xs) // 4)
        if n_arrows > 0:
            step = max(1, len(xs) // n_arrows)
            for i in range(0, len(xs) - step, step):
                ax.annotate(
                    "",
                    xy=(xs[i + step], ys[i + step]),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=path_color,
                        lw=1.5,
                    ),
                    zorder=9,
                )

    # ── Legend ────────────────────────────────────────────────────────────────
    path_patch = mpatches.Patch(color=path_color, label="Optimal Path")
    start_patch = mpatches.Patch(color="#2dc653", label="Start Point")
    goal_patch = mpatches.Patch(color="#e63946", label="Goal Point")

    ax.legend(
        handles=[path_patch, start_patch, goal_patch],
        loc="upper right",
        framealpha=0.85,
        fontsize=8,
        facecolor="white",
        edgecolor="#cccccc",
    )

    # Title
    dxf_name = Path(dxf_path).stem if dxf_path else "Floor Plan"
    ax.set_title(
        f"Floor Plan Path — {dxf_name}\n"
        f"Path Length: {len(path_dxf_coords)} waypoints",
        fontsize=10, pad=8,
    )

    # ── Save PDF ──────────────────────────────────────────────────────────────
    with PdfPages(str(output_path)) as pdf:
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight", facecolor="white")

        # Add metadata page
        _add_metadata_page(pdf, dxf_path, path_dxf_coords, mapper)

    plt.close(fig)
    logger.info(f"[PDFExport] PDF saved: {output_path}")
    return str(output_path)


def _render_dxf_to_figure(dxf_path: str, dpi: int = 200):
    """
    Render a DXF file to a matplotlib figure using ezdxf's drawing add-on.
    Returns (fig, ax) or (None, None) if rendering fails.
    """
    try:
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
        ax.set_facecolor("white")

        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)

        # Override colors for B/W output
        ctx.set_current_layout(msp)

        Frontend(ctx, out).draw_layout(msp, finalize=True)

        ax.set_aspect("equal")
        ax.tick_params(colors="#888888", labelsize=6)
        ax.set_xlabel("X (DXF units)", fontsize=7)
        ax.set_ylabel("Y (DXF units)", fontsize=7)

        logger.info("[PDFExport] DXF rendered successfully via ezdxf drawing add-on")
        return fig, ax

    except ImportError:
        logger.warning("[PDFExport] ezdxf drawing add-on not available (install ezdxf[draw])")
        return None, None
    except Exception as e:
        logger.warning(f"[PDFExport] DXF render error: {e}")
        return None, None


def _fallback_grid_figure(mapper):
    """
    Fallback: render a simple axes representing the grid coordinate space.
    Used when ezdxf rendering fails.
    """
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.set_facecolor("#f8f8f8")
    ax.set_xlim(mapper.bounds.min_x, mapper.bounds.max_x)
    ax.set_ylim(mapper.bounds.min_y, mapper.bounds.max_y)
    ax.set_aspect("equal")
    ax.set_xlabel("X (DXF units)", fontsize=8)
    ax.set_ylabel("Y (DXF units)", fontsize=8)
    ax.set_title("Floor Plan (DXF rendering unavailable — showing path only)", fontsize=9)
    ax.grid(True, alpha=0.3, color="#cccccc")
    return fig, ax


def _add_metadata_page(pdf, dxf_path: str, path_coords: list, mapper):
    """Add a metadata summary page to the PDF."""
    fig_meta, ax_meta = plt.subplots(figsize=(8.5, 11), facecolor="white")
    ax_meta.axis("off")

    total_length_m = mapper.path_length_meters(
        [tuple(mapper.dxf_to_grid(x, y)) for x, y in path_coords]
    ) if path_coords else 0

    lines = [
        "CAD-to-Grid DDQN Path Planner — Run Report",
        "=" * 60,
        "",
        f"Source File:     {Path(dxf_path).name if dxf_path else 'N/A'}",
        f"DXF Units:       {mapper.unit_label}",
        f"Grid Resolution: {mapper.cell_size_m:.3f} m/cell",
        f"Grid Size:       {mapper.grid_rows} × {mapper.grid_cols} cells",
        "",
        f"Path Waypoints:  {len(path_coords)}",
        f"Path Length:     {total_length_m:.2f} m",
        "",
        "Algorithm:       DDQN (Double Deep Q-Network) + A* baseline",
        "Legend:",
        "  ● Green dot  = Start point",
        "  ★ Red star   = Goal point",
        "  — Red line   = Optimal path (DDQN/A* result)",
    ]

    ax_meta.text(
        0.1, 0.9,
        "\n".join(lines),
        transform=ax_meta.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        color="#222222",
    )

    pdf.savefig(fig_meta, bbox_inches="tight", facecolor="white")
    plt.close(fig_meta)
