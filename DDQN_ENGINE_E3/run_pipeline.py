"""
run_pipeline.py
───────────────
Main entry point for the CAD-to-Grid DDQN Path Planner v2.

5-Step Pipeline:
  Step 1: CLI prompt for file path (PDF / DXF / DWG) + local conversion
  Step 2: Grid generation with ALL doors forced open / traversable
  Step 3: Interactive UI to select Start and Goal points
  Step 4: Pygame live visualization window opens; DDQN trains in real time
  Step 5: Final path saved + metrics dashboard generated

Usage:
  python run_pipeline.py                          # prompts for file path
  python run_pipeline.py --input floor_plan.dxf  # non-interactive input
  python run_pipeline.py --input plan.pdf --no-live-viz  # headless mode
  python run_pipeline.py --input plan.dxf --start 10 15 --goal 80 90
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows to prevent UnicodeEncodeError
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Setup path so imports work when run from project root ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Color-coded console logging ───────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)


def cprint(msg: str, color: str = "cyan", bold: bool = False):
    """Color-print a pipeline status message."""
    if HAS_COLOR:
        c = getattr(Fore, color.upper(), "")
        b = "\033[1m" if bold else ""
        print(f"{b}{c}  {msg}{Style.RESET_ALL}")
    else:
        print(f"  {msg}")


def banner(stage: int, title: str):
    line = "═" * 64
    cprint(f"\n{line}", "blue")
    cprint(f"  Step {stage}: {title}", "cyan", bold=True)
    cprint(f"{line}\n", "blue")


def _pick_floor_plan_file() -> str:
    """
    Open a native OS file-browser dialog so the user can 'upload' (select)
    their floor plan file. Returns the selected path string, or "" if cancelled.

    Supported: .pdf  .dxf  .dwg  .dxb  .dwt  .dxt

    Uses Tkinter's askopenfilename (always installed with Python on Windows).
    Falls back to a plain terminal prompt if a display is not available.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()           # hide the blank root window
        root.lift()
        root.attributes("-topmost", True)   # bring dialog to front

        filetypes = [
            ("All Floor Plans",      "*.pdf *.dxf *.dwg *.dxb *.dwt *.dxt"),
            ("PDF Floor Plans",      "*.pdf"),
            ("DXF/DWG CAD Files",    "*.dxf *.dwg *.dxb *.dwt *.dxt"),
            ("All Files",            "*.*"),
        ]

        selected = filedialog.askopenfilename(
            title="Select Floor Plan File (PDF / DXF / DWG)",
            filetypes=filetypes,
            initialdir=str(Path.cwd()),
        )
        root.destroy()
        return selected  # "" if user cancels

    except Exception:
        # Headless / no display fallback: plain terminal prompt
        cprint("  (File browser unavailable — using text input)", "yellow")
        cprint("  Supported formats: .pdf  .dxf  .dwg  .dxb  .dwt  .dxt", "cyan")
        val = input("  ▶ Enter full path to floor plan file: ").strip().strip('"').strip("'")
        return val


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    t_pipeline_start = time.perf_counter()
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Clean up previous run outputs ──────────────────────────────────────────
    old_files = [
        "annotated_floor_plan.pdf", "coordinates.json", "door_cells.npy", 
        "metrics_dashboard.png", "metrics.json", "occupancy_grid.npy", 
        "occupancy_grid.png", "parsed_symbols.json", "scale_mapping.json"
    ]
    for fname in old_files:
        fpath = results_dir / fname
        if fpath.exists():
            try:
                fpath.unlink()
            except Exception:
                pass


    cprint(f"\n{'═'*66}", "magenta", bold=True)
    cprint(f"  CAD-to-Grid DDQN Path Planner  v2.1", "magenta", bold=True)
    cprint(f"{'═'*66}\n", "magenta", bold=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: FILE INPUT + FORMAT NORMALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    banner(1, "File Input & Format Normalization")

    # Prompt user for input if not provided via CLI
    input_file = args.input
    if not input_file:
        cprint("  No input file specified via --input", "yellow")
        cprint("  Opening file browser — select your floor plan...", "cyan")
        input_file = _pick_floor_plan_file()
        if not input_file:
            cprint("✗ No file selected. Exiting.", "red", bold=True)
            sys.exit(1)
        cprint(f"  Selected: {input_file}", "green")

    input_path = Path(input_file)
    if not input_path.exists():
        cprint(f"✗ File not found: {input_path}", "red", bold=True)
        sys.exit(1)

    ext = input_path.suffix.lower()
    cprint(f"  Input: {input_path}  [{ext.upper()} format]", "cyan")
    cprint(f"  Output directory: {results_dir.resolve()}", "cyan")

    from cad_ingestion.convert_cad_to_dxf import normalize_to_dxf, validate_dxf

    tmp_dir = results_dir / "tmp"
    try:
        t0 = time.perf_counter()
        dxf_path = normalize_to_dxf(str(input_path), output_dir=str(tmp_dir))
        elapsed = (time.perf_counter() - t0) * 1000
        cprint(f"✓ Normalized to DXF in {elapsed:.0f}ms → {Path(dxf_path).name}", "green")
    except (FileNotFoundError, RuntimeError) as e:
        cprint(f"✗ FATAL: {e}", "red", bold=True)
        sys.exit(1)

    # Validate
    report = validate_dxf(dxf_path)
    if not report["valid"]:
        cprint(f"✗ DXF validation failed: {report['warnings']}", "red")
        sys.exit(1)

    cprint(f"✓ DXF version: {report['version']},  units: {report['units']}", "green")
    cprint(f"✓ Entities: {report['entity_counts']}", "green")
    for w in report.get("warnings", []):
        cprint(f"  ⚠ {w}", "yellow")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: GRID GENERATION (all doors forced open)
    # ═══════════════════════════════════════════════════════════════════════════
    banner(2, "Grid Generation — Doors Forced Open")

    from cad_ingestion.dxf_parser import parse_dxf
    from cad_ingestion.scale_utils import ScaleMapper, auto_cell_size
    from symbol_model.infer_symbols import classify_all_symbols, get_confidence_report
    from grid_builder.build_grid import build_occupancy_grid

    # Parse DXF
    parsed = parse_dxf(dxf_path)
    n_prims = len(parsed.primitives)
    cprint(f"✓ Parsed: {n_prims} primitives, {len(parsed.entity_groups)} groups", "green")

    # ── Decide: vector path or raster fallback ─────────────────────────────────
    # Triggers when ezdxf finds 0 vector entities (rasterized/scanned floor plan).
    _use_raster = (n_prims == 0)
    if _use_raster:
        cprint("  ⚠ 0 vector entities — floor plan is a raster/scanned image.", "yellow")
        cprint("  ↳ Switching to OpenCV contour-based grid extraction.", "yellow")

    # ── VECTOR PATH ────────────────────────────────────────────────────────────
    if not _use_raster:
        cell_size_m = auto_cell_size(
            parsed.drawing_bounds,
            parsed.units_code,
            target_max_dim=args.max_grid_dim,
            default_cell_size_m=args.cell_size,
        )
        mapper = ScaleMapper(parsed.drawing_bounds, parsed.units_code, cell_size_m)
        mapper.save(str(results_dir / "scale_mapping.json"))
        cprint(f"✓ Grid: {mapper.grid_rows}×{mapper.grid_cols} cells  ({cell_size_m:.3f} m/cell)", "green")

        cache_path = str(results_dir / "parsed_symbols.json")
        classifications = classify_all_symbols(
            parsed,
            weights_path=args.cnn_weights,
            cache_path=cache_path,
            force_reclassify=args.force_reclassify,
        )
        conf = get_confidence_report(classifications)
        cprint(
            f"✓ Classified {conf['total_groups']} symbol groups  "
            f"(high-conf: {conf['high_confidence_>=0.75']}, mean: {conf['mean_confidence']:.2f})",
            "green",
        )
        grid, door_cells = build_occupancy_grid(
            parsed, classifications, mapper,
            output_dir=str(results_dir),
            safety_margin=args.safety_margin,
        )
        # Sanity: nearly-empty vector grid means walls weren't detected → raster.
        # Threshold 88.5%: catches floor plans with diagonal structural walls that
        # Hough misses (89-92% free) while keeping clean orthogonal plans (≤88.5%).
        if 100 * (grid == 0).sum() / grid.size > 88.5:
            cprint(f"  ⚠ Vector grid >88.5%% free — Hough likely missed structural walls.", "yellow")
            cprint(f"  ↳ Switching to OpenCV raster fallback (handles diagonal walls).", "yellow")
            _use_raster = True


    # ── RASTER FALLBACK PATH ───────────────────────────────────────────────────
    if _use_raster:
        import numpy as np
        from cad_ingestion.raster_to_grid import (
            image_to_occupancy_grid, pdf_image_to_occupancy_grid, save_raster_grid_image
        )
        cprint(f"  Running OpenCV image analysis on: {input_path.name}", "cyan")
        try:
            if input_path.suffix.lower() == ".pdf":
                grid, raster_meta = pdf_image_to_occupancy_grid(
                    str(input_path), page=0,
                    max_dim=args.max_grid_dim, dpi=200,
                    target_rows=mapper.grid_rows,   # align with vector-path scale
                    target_cols=mapper.grid_cols,
                )
            else:
                tmp_img = results_dir / "tmp" / f"{input_path.stem}_page0.png"
                src = str(tmp_img) if tmp_img.exists() else str(input_path)
                grid, raster_meta = image_to_occupancy_grid(
                    src, max_dim=args.max_grid_dim,
                    target_rows=mapper.grid_rows,
                    target_cols=mapper.grid_cols,
                )
        except Exception as e:
            cprint(f"✗ Raster fallback failed: {e}", "red", bold=True)
            sys.exit(1)

        door_cells = np.zeros(grid.shape, dtype=bool)
        classifications = {}
        save_raster_grid_image(grid, str(results_dir / "occupancy_grid.png"))
        np.save(str(results_dir / "occupancy_grid.npy"), grid)
        np.save(str(results_dir / "door_cells.npy"), door_cells)
        cell_size_m = raster_meta.get("cell_size_m") or args.cell_size
        # mapper already saved above; raster uses same mapper (same cell_size_m)
        cprint(
            f"✓ Raster grid: {grid.shape[0]}×{grid.shape[1]}  "
            f"Free: {raster_meta['free_pct']:.1f}%  (px/cell={raster_meta['px_per_cell']:.1f})",
            "green",
        )


    free_pct = 100 * (grid == 0).sum() / grid.size
    n_door_cells = int(door_cells.sum())
    cprint(f"✓ Grid built: {grid.shape[0]}×{grid.shape[1]}  |  Free: {free_pct:.1f}%", "green")
    cprint(f"✓ Door cells forced open: {n_door_cells} cells", "green")



    # ═══════════════════════════════════════════════════════════════════════════
    # PREPARE BACKGROUND IMAGE FOR UI
    # ═══════════════════════════════════════════════════════════════════════════
    bg_image_path = None
    tmp_bg = results_dir / "tmp" / f"{input_path.stem}_bg.png"
    if input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        bg_image_path = str(input_path)
    elif input_path.suffix.lower() == ".pdf":
        page_img = results_dir / "tmp" / f"{input_path.stem}_page0.png"
        if not page_img.exists():
            try:
                # pyrefly: ignore [missing-import]
                from pdf2image import convert_from_path
                pages = convert_from_path(str(input_path), dpi=200, first_page=1, last_page=1)
                if pages:
                    pages[0].save(str(page_img), "PNG")
            except Exception:
                pass
        if page_img.exists():
            bg_image_path = str(page_img)
    else:
        # DXF / DWG
        if not tmp_bg.exists():
            try:
                import matplotlib.pyplot as plt
                from ezdxf.addons.drawing import RenderContext, Frontend
                from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                ctx = RenderContext(parsed.doc)
                out = MatplotlibBackend(ax)
                Frontend(ctx, out).draw_layout(parsed.doc.modelspace(), finalize=True)
                fig.savefig(str(tmp_bg), dpi=200, facecolor="white")
                plt.close(fig)
            except Exception as e:
                cprint(f"  [Warning] Could not generate DXF background image: {e}", "yellow")
        if tmp_bg.exists():
            bg_image_path = str(tmp_bg)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: INTERACTIVE GRID REVIEW + POINT SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    banner(3, "Interactive Grid Review & Point Selection")

    from review_ui.grid_review import run_grid_review

    if args.start and args.goal:
        start = tuple(args.start)
        goal  = tuple(args.goal)
        cprint(f"✓ CLI mode — Start: {start}, Goal: {goal}", "cyan")
        if grid[start[0], start[1]] == 1:
            cprint(f"✗ Start point {start} is on an obstacle!", "red")
            sys.exit(1)
        if grid[goal[0], goal[1]] == 1:
            cprint(f"✗ Goal point {goal} is on an obstacle!", "red")
            sys.exit(1)
    else:
        cprint("  Opening interactive grid review UI...", "cyan")
        cprint("  ● Click 'Set Start Point' → click a white cell on the grid", "cyan")
        cprint("  ● Click 'Set Goal Point'  → click another white cell", "cyan")
        cprint("  ● Click 'Proceed to Planning' when ready", "cyan")
        cprint("  ↳ Tip: Amber cells are detected doorways (always walkable)", "yellow")
        try:
            start, goal = run_grid_review(
                grid,
                dxf_filename=input_path.name,
                classifications=classifications,
                bg_image_path=bg_image_path,
            )
        except RuntimeError as e:
            cprint(f"✗ Grid review cancelled: {e}", "red")
            sys.exit(1)

    cprint(f"✓ Start: {start}  |  Goal: {goal}", "green")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: DDQN TRAINING + LIVE VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    banner(4, "DDQN Training + A* Path Planning  [Live Visualization]")

    use_live = not args.no_live_viz
    frame_skip = args.frame_skip

    # Auto-scale frame_skip for large grids
    if use_live and frame_skip is None:
        grid_cells = grid.shape[0] * grid.shape[1]
        if grid_cells > 50_000:
            frame_skip = 20
            cprint(f"  Large grid detected — auto frame_skip={frame_skip}", "yellow")
        elif grid_cells > 10_000:
            frame_skip = 10
        else:
            frame_skip = 5
    elif frame_skip is None:
        frame_skip = 5

    if use_live:
        cprint("  Live Pygame visualization will open now.", "cyan")
        cprint(f"  Frame-skip: {frame_skip}  (render every {frame_skip} agent steps)", "cyan")
    else:
        cprint("  Live visualization disabled (--no-live-viz)", "yellow")

    cprint(f"  DDQN episodes: {args.episodes}  |  Weights: {args.ddqn_weights}", "cyan")

    from rl_agent.run_agent import run_pathfinder

    def ddqn_progress(episode, total_episodes, reward, epsilon, astar_done):
        if episode % max(1, total_episodes // 10) == 0 or episode == total_episodes:
            astar_str = " [A* ✓]" if astar_done else " [A* ...]"
            cprint(
                f"  Ep {episode:>4}/{total_episodes}  |  "
                f"reward={reward:>8.1f}  |  ε={epsilon:.3f}{astar_str}",
                "cyan",
            )

    planning_result = run_pathfinder(
        grid=grid,
        start=start,
        goal=goal,
        n_episodes=args.episodes,
        weights_path=args.ddqn_weights,
        progress_callback=ddqn_progress,
        save_weights=True,
        live_viz=use_live,
        frame_skip=frame_skip,
        door_cells=door_cells,
        bg_image_path=bg_image_path,
    )

    if not planning_result.success:
        cprint("✗ Path planning failed — no valid path found!", "red", bold=True)
        cprint("  Check that start and goal are connected in the grid.", "yellow")
        sys.exit(1)

    cprint(f"✓ Path found: {len(planning_result.path)} cells", "green")
    cprint(
        f"  A* cost: {planning_result.astar_cost:.1f}  |  "
        f"A* time: {planning_result.astar_time_ms:.1f}ms  |  "
        f"DDQN gap: {planning_result.optimality_gap_pct:.1f}%",
        "green",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: OUTPUT GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    banner(5, "Output Generation")

    # 5a. Export coordinates JSON
    from output_gen.export_coordinates import export_coordinates
    coord_data = export_coordinates(
        planning_result.path,
        mapper,
        output_path=str(results_dir / "coordinates.json"),
        source_file=str(input_path),
    )
    cprint(
        f"✓ Coordinates: {coord_data.get('waypoints_count', 0)} waypoints, "
        f"{coord_data.get('total_length_m', 0):.2f} m → coordinates.json",
        "green",
    )

    # 5b. Export annotated PDF
    from output_gen.export_annotated_pdf import export_annotated_pdf
    pdf_path = export_annotated_pdf(
        dxf_path=dxf_path,
        path_dxf_coords=coord_data.get("path_dxf_coords", []),
        mapper=mapper,
        output_path=str(results_dir / "annotated_floor_plan.pdf"),
    )
    cprint(f"✓ Annotated PDF: {pdf_path}", "green")

    # 5c. Metrics dashboard
    from output_gen.metrics_report import generate_metrics_dashboard
    dashboard_path = generate_metrics_dashboard(
        planning_result=planning_result,
        mapper=mapper,
        grid=grid,
        classifications=classifications,
        source_file=str(input_path),
        output_path=str(results_dir / "metrics_dashboard.png"),
        show=not args.no_display,
    )
    cprint(f"✓ Metrics dashboard: {dashboard_path}", "green")

    # 5d. Export JSON metrics report
    import json
    
    real_world_scale = mapper.meters_per_cell if hasattr(mapper, 'meters_per_cell') else 0.1
    def calc_mean_reward(rewards):
        return float(np.mean(rewards)) if rewards else 0.0

    metrics_data = {
        "DDQN": {
            "Success Rate": planning_result.ddqn_success_rate,
            "Average Episode Reward": calc_mean_reward(planning_result.ddqn_episode_rewards),
            "Convergence Speed (Episodes)": planning_result.ddqn_convergence_episode,
            "Path Length (cells)": planning_result.ddqn_path_cost,
            "Path Length in Real World (m)": planning_result.ddqn_path_cost * real_world_scale if planning_result.ddqn_path_cost < float("inf") else float("inf"),
            "Planning Time (ms)": planning_result.ddqn_inference_time_ms,
            "Total Navigation Time (ms)": planning_result.ddqn_inference_time_ms + planning_result.astar_time_ms,
            "Optimality Ratio": planning_result.ddqn_path_cost / max(1, planning_result.astar_cost) if planning_result.astar_cost > 0 else 0,
            "Collision Rate": planning_result.ddqn_collisions / max(1, planning_result.ddqn_path_cost) if planning_result.ddqn_path_cost < float("inf") else 0,
            "Dynamic Obstacle Avoidance Rate": 0.0,
            "Path Smoothness": planning_result.ddqn_smoothness,
            "Replanning Frequency": 0,
            "CPU Peak Usage (%)": planning_result.ddqn_cpu_peak,
            "Memory Peak Usage (MB)": planning_result.ddqn_mem_peak
        },
        "PPOA*": {
            "Success Rate": planning_result.ppoa_success_rate,
            "Average Episode Reward": calc_mean_reward(planning_result.ppoa_episode_rewards),
            "Convergence Speed (Episodes)": planning_result.ppoa_convergence_episode,
            "Path Length (cells)": planning_result.ppoa_path_cost,
            "Path Length in Real World (m)": planning_result.ppoa_path_cost * real_world_scale if planning_result.ppoa_path_cost < float("inf") else float("inf"),
            "Planning Time (ms)": planning_result.ppoa_inference_time_ms,
            "Total Navigation Time (ms)": planning_result.ppoa_inference_time_ms + planning_result.astar_time_ms,
            "Optimality Ratio": planning_result.ppoa_path_cost / max(1, planning_result.astar_cost) if planning_result.astar_cost > 0 else 0,
            "Collision Rate": planning_result.ppoa_collisions / max(1, planning_result.ppoa_path_cost) if planning_result.ppoa_path_cost < float("inf") else 0,
            "Dynamic Obstacle Avoidance Rate": planning_result.ppoa_dyn_avoid_rate,
            "Path Smoothness": planning_result.ppoa_smoothness,
            "Replanning Frequency": planning_result.ppoa_replans,
            "CPU Peak Usage (%)": planning_result.ppoa_cpu_peak,
            "Memory Peak Usage (MB)": planning_result.ppoa_mem_peak
        }
    }
    
    metrics_json_path = results_dir / "metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    cprint(f"✓ Metrics JSON report: {metrics_json_path}", "green")


    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    total_s = time.perf_counter() - t_pipeline_start

    cprint(f"\n{'═'*66}", "magenta", bold=True)
    cprint(f"  Pipeline Complete in {total_s:.1f}s", "magenta", bold=True)
    cprint(f"{'═'*66}", "magenta", bold=True)
    cprint(f"\n  Output files in: {results_dir.resolve()}", "green")
    cprint(f"  ├── occupancy_grid.png         (grid preview with door highlights)", "white")
    cprint(f"  ├── annotated_floor_plan.pdf   (floor plan + red optimal path)", "white")
    cprint(f"  ├── coordinates.json           (real-world path waypoints)", "white")
    cprint(f"  ├── metrics_dashboard.png      (DDQN + A* performance metrics)", "white")
    cprint(f"  ├── metrics.json               (Detailed DDQN/PPOA* metrics data)", "white")
    cprint(f"  ├── parsed_symbols.json        (cached symbol classifications)", "white")
    cprint(f"  ├── door_cells.npy             (door traversability mask)", "white")
    cprint(f"  └── scale_mapping.json         (grid <-> DXF coordinate map)\n", "white")

    cprint(
        f"  Path: {len(planning_result.path)} cells  |  "
        f"{coord_data.get('total_length_m', 0):.2f} m real-world  |  "
        f"A* optimal in {planning_result.astar_time_ms:.1f}ms",
        "green", bold=True,
    )

    return planning_result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="CAD-to-Grid DDQN Path Planner v2.1 — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prompted input (no args):
  python run_pipeline.py

  # PDF floor plan (local conversion, no paid APIs):
  python run_pipeline.py --input floor_plan.pdf

  # DXF with pre-set start/goal (skips UI):
  python run_pipeline.py --input plan.dxf --start 10 15 --goal 80 90

  # Headless / CI mode (no Pygame, no display):
  python run_pipeline.py --input plan.dxf --no-live-viz --no-display

  # Custom frame-skip for large floor plans:
  python run_pipeline.py --input big_plan.dwg --frame-skip 20
        """,
    )

    parser.add_argument(
        "--input", default=None, metavar="FILE",
        help="Input floor plan (.pdf, .dxf, .dwg, .dxb). If omitted, prompts interactively."
    )
    parser.add_argument(
        "--output-dir", default="Results", metavar="DIR",
        help="Output directory for all generated files (default: Results/)"
    )
    parser.add_argument(
        "--cell-size", type=float, default=0.1, metavar="M",
        help="Grid cell size in meters (default: 0.10 m)"
    )
    parser.add_argument(
        "--max-grid-dim", type=int, default=500, metavar="N",
        help="Maximum grid dimension in cells (auto-scales; default: 500)"
    )
    parser.add_argument(
        "--safety-margin", type=int, default=1, metavar="CELLS",
        help="Obstacle dilation in cells for robot safety (default: 1)"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, metavar="N",
        help="DDQN training episodes (default: 100)"
    )
    parser.add_argument(
        "--start", type=int, nargs=2, metavar=("ROW", "COL"),
        help="Start point grid coords — skips interactive UI"
    )
    parser.add_argument(
        "--goal", type=int, nargs=2, metavar=("ROW", "COL"),
        help="Goal point grid coords — skips interactive UI"
    )
    parser.add_argument(
        "--ddqn-weights", default="models/ddqn_best.pth", metavar="PATH",
        help="Path to load/save DDQN weights (default: models/ddqn_best.pth)"
    )
    parser.add_argument(
        "--cnn-weights", default="models/symbol_cnn.pth", metavar="PATH",
        help="Path to Tier-2 CNN weights (optional)"
    )
    parser.add_argument(
        "--force-reclassify", action="store_true",
        help="Force re-running symbol classification (ignore cache)"
    )
    parser.add_argument(
        "--no-live-viz", action="store_true",
        help="Disable Pygame live visualization (headless mode)"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=None, metavar="N",
        help="Live viz: render every N agent steps (auto-scaled if not set)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Suppress interactive display of metrics dashboard"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    run_pipeline(args)