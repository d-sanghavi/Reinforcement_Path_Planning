"""
grid_review.py
──────────────
Interactive Tkinter + matplotlib grid review UI.

Provides:
  1. Visual display of the binary occupancy grid
  2. Toggle cell corrections (click to flip free↔obstacle)
  3. Start point selection (green marker)
  4. Goal point selection (red marker)
  5. Validation: both points must land on free cells
  6. "Proceed to Planning" → returns (start, goal) to the caller

The UI blocks execution (mainloop) until the user clicks "Proceed".
"""

import logging
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")  # Tkinter backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
import numpy as np

logger = logging.getLogger(__name__)

FREE = 0
OBSTACLE = 1


class GridReviewUI:
    """
    Interactive grid review and point selection window.

    Parameters
    ----------
    grid : np.ndarray
        Binary occupancy grid (0=free, 1=obstacle).
    dxf_filename : str
        Source file name (for window title).
    classifications : dict, optional
        Symbol classifications for overlay display.
    """

    def __init__(
        self,
        grid: np.ndarray,
        dxf_filename: str = "Floor Plan",
        classifications: Optional[dict] = None,
        bg_image_path: Optional[str] = None,
    ):
        self.grid = grid.copy()
        self.dxf_filename = dxf_filename
        self.classifications = classifications or {}
        self.bg_image_path = bg_image_path

        self.start_point: Optional[tuple] = None
        self.goal_point:  Optional[tuple] = None

        self.mode = "view"  # "view", "correct", "select_start", "select_goal"
        self._result: Optional[tuple] = None
        self._confirmed = False

        self.root = None
        self.canvas = None
        self.fig = None
        self.ax = None
        self.img_display = None
        self.status_var = None

        # Display grid (mutable for corrections)
        self._display_grid = self.grid.copy().astype(float)

    def show(self) -> tuple:
        """
        Launch the UI. Blocks until user clicks "Proceed to Planning".

        Returns
        -------
        tuple : (start_point, goal_point) as (row, col) tuples.
        Raises RuntimeError if user closes window without confirming.
        """
        self._build_ui()
        self.root.mainloop()

        if not self._confirmed:
            raise RuntimeError("Grid review cancelled by user")

        return self.start_point, self.goal_point

    # ═══════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title(f"CAD-to-Grid Path Planner — {self.dxf_filename}")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1200x750")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#16213e", pady=8)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="🗺  Occupancy Grid Review",
            font=("Segoe UI", 16, "bold"),
            fg="#e94560",
            bg="#16213e",
        ).pack(side=tk.LEFT, padx=16)

        rows, cols = self.grid.shape
        tk.Label(
            header,
            text=f"Grid: {rows}×{cols}  |  Free: {100*np.sum(self.grid==FREE)/self.grid.size:.1f}%  |  Obstacles: {100*np.sum(self.grid==OBSTACLE)/self.grid.size:.1f}%",
            font=("Segoe UI", 10),
            fg="#a0a0c0",
            bg="#16213e",
        ).pack(side=tk.LEFT, padx=16)

        # ── Main layout ───────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True)

        # Left: matplotlib canvas
        canvas_frame = tk.Frame(main, bg="#1a1a2e")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(9, 7), facecolor="#0f0f1a")
        self.ax.set_facecolor("#0f0f1a")
        self.fig.tight_layout(pad=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(canvas_frame, bg="#1a1a2e")
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Right: control panel
        ctrl = tk.Frame(main, bg="#16213e", width=220, padx=12, pady=12)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl.pack_propagate(False)

        self._build_controls(ctrl)

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg="#0f0f1a", pady=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Click 'Select Start Point' then click on the grid.")
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#a0c0a0",
            bg="#0f0f1a",
        ).pack(side=tk.LEFT, padx=10)

        # ── Initial render ────────────────────────────────────────────────────
        self._render_grid()
        self.canvas.mpl_connect("button_press_event", self._on_grid_click)

    def _build_controls(self, parent):
        """Build the right-side control panel."""
        label_style = {"font": ("Segoe UI", 10, "bold"), "fg": "#e0e0f0", "bg": "#16213e", "anchor": "w"}
        btn_style_base = {"font": ("Segoe UI", 10, "bold"), "relief": tk.FLAT, "bd": 0, "cursor": "hand2", "pady": 8}

        # Title
        tk.Label(parent, text="Controls", font=("Segoe UI", 13, "bold"), fg="#e94560", bg="#16213e").pack(fill=tk.X, pady=(0, 12))

        # ── Mode buttons ──────────────────────────────────────────────────────
        tk.Label(parent, text="Grid Interaction Mode", **label_style).pack(fill=tk.X, pady=(4, 2))

        self.mode_var = tk.StringVar(value="view")

        for mode_name, mode_val, color in [
            ("👁  View / Pan / Zoom", "view", "#4a90d9"),
            ("✏  Correct Cells",     "correct", "#f5a623"),
        ]:
            rb = tk.Radiobutton(
                parent, text=mode_name, variable=self.mode_var, value=mode_val,
                font=("Segoe UI", 9), fg="#e0e0f0", bg="#16213e",
                selectcolor="#2a2a4e", activebackground="#16213e",
                command=self._update_mode,
            )
            rb.pack(fill=tk.X, pady=2)

        tk.Frame(parent, bg="#3a3a5e", height=1).pack(fill=tk.X, pady=8)

        # ── Point selection ───────────────────────────────────────────────────
        tk.Label(parent, text="Set Path Points", **label_style).pack(fill=tk.X, pady=(4, 4))

        self.btn_start = tk.Button(
            parent, text="▶  Set Start Point (S)",
            bg="#1a7a1a", fg="white",
            command=lambda: self._set_mode("select_start"),
            **btn_style_base,
        )
        self.btn_start.pack(fill=tk.X, pady=3)

        self.btn_goal = tk.Button(
            parent, text="🏁  Set Goal Point (G)",
            bg="#7a1a1a", fg="white",
            command=lambda: self._set_mode("select_goal"),
            **btn_style_base,
        )
        self.btn_goal.pack(fill=tk.X, pady=3)

        tk.Frame(parent, bg="#3a3a5e", height=1).pack(fill=tk.X, pady=8)

        # ── Point display ─────────────────────────────────────────────────────
        tk.Label(parent, text="Current Selection", **label_style).pack(fill=tk.X, pady=(4, 4))

        self.lbl_start = tk.Label(parent, text="Start: —", font=("Segoe UI", 9), fg="#80ff80", bg="#16213e", anchor="w")
        self.lbl_start.pack(fill=tk.X)
        self.lbl_goal = tk.Label(parent, text="Goal:  —", font=("Segoe UI", 9), fg="#ff8080", bg="#16213e", anchor="w")
        self.lbl_goal.pack(fill=tk.X)

        tk.Frame(parent, bg="#3a3a5e", height=1).pack(fill=tk.X, pady=8)

        # ── Legend ────────────────────────────────────────────────────────────
        tk.Label(parent, text="Legend", **label_style).pack(fill=tk.X, pady=(4, 4))
        for color, text in [
            ("#ffffff", "  Free / Traversable"),
            ("#2a2a2a", "  Obstacle / Wall"),
            ("#00cc00", "  Start Point"),
            ("#cc0000", "  Goal Point"),
            ("#0080ff", "  Planned Path"),
        ]:
            row = tk.Frame(parent, bg="#16213e")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text="■", font=("Segoe UI", 14), fg=color, bg="#16213e", width=2).pack(side=tk.LEFT)
            tk.Label(row, text=text, font=("Segoe UI", 9), fg="#c0c0c0", bg="#16213e").pack(side=tk.LEFT)

        # ── Clear / Reset ─────────────────────────────────────────────────────
        tk.Button(
            parent, text="🔄  Clear Points",
            bg="#3a3a5e", fg="#c0c0c0",
            command=self._clear_points,
            **btn_style_base,
        ).pack(fill=tk.X, pady=(12, 4))

        # ── PROCEED button (big, prominent) ───────────────────────────────────
        self.btn_proceed = tk.Button(
            parent,
            text="⚡  Proceed to Planning",
            bg="#e94560", fg="white",
            font=("Segoe UI", 11, "bold"),
            command=self._on_proceed,
            relief=tk.FLAT, bd=0, cursor="hand2",
            pady=12,
        )
        self.btn_proceed.pack(fill=tk.X, pady=(8, 0))
        self.btn_proceed.config(state=tk.DISABLED)

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDERING
    # ═══════════════════════════════════════════════════════════════════════════

    def _render_grid(self):
        """Redraw the occupancy grid with current markers."""
        self.ax.cla()
        self.ax.set_facecolor("#0f0f1a")

        rows, cols = self._display_grid.shape
        
        # Load background image if provided
        if self.bg_image_path:
            try:
                bg_img = plt.imread(self.bg_image_path)
                self.ax.imshow(bg_img, extent=[0, cols, rows, 0])
            except Exception as e:
                logger.warning(f"[GridUI] Could not load background image: {e}")

        # Build display image (RGBA)
        rgba = np.zeros((rows, cols, 4), dtype=np.float32)
        
        if self.bg_image_path:
            # Alpha overlay mode
            rgba[self._display_grid == OBSTACLE] = [0.12, 0.12, 0.18, 0.6]  # dark semi-transparent
            # FREE remains [0,0,0,0]
        else:
            # Solid mode
            rgba[self._display_grid == FREE] = [0.94, 0.94, 0.96, 1.0]
            rgba[self._display_grid == OBSTACLE] = [0.12, 0.12, 0.18, 1.0]

        # Mark start/goal
        if self.start_point:
            sr, sc = self.start_point
            r_s, r_e = max(0, sr-1), min(rows-1, sr+1)
            c_s, c_e = max(0, sc-1), min(cols-1, sc+1)
            rgba[r_s:r_e+1, c_s:c_e+1] = [0.0, 0.86, 0.31, 1.0]  # green

        if self.goal_point:
            gr, gc = self.goal_point
            r_s, r_e = max(0, gr-1), min(rows-1, gr+1)
            c_s, c_e = max(0, gc-1), min(cols-1, gc+1)
            rgba[r_s:r_e+1, c_s:c_e+1] = [0.86, 0.16, 0.16, 1.0]  # red

        self.ax.imshow(rgba, interpolation="nearest", aspect="auto", extent=[0, cols, rows, 0])
        self.ax.set_title(
            f"Floor Plan Occupancy Grid — {self.dxf_filename}",
            color="#e0e0f0", fontsize=11, pad=8,
        )
        self.ax.tick_params(colors="#606080", labelsize=7)
        self.ax.set_xlabel("Grid Column (X)", color="#606080", fontsize=8)
        self.ax.set_ylabel("Grid Row (Y, flipped)", color="#606080", fontsize=8)

        self.canvas.draw_idle()

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_grid_click(self, event):
        """Handle click on the matplotlib canvas."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        rows, cols = self.grid.shape

        if not (0 <= row < rows and 0 <= col < cols):
            return

        mode = self.mode_var.get()

        if mode == "correct":
            # Toggle cell value
            self._display_grid[row, col] = 1 - self._display_grid[row, col]
            self.grid[row, col] = self._display_grid[row, col]
            self._render_grid()

        elif mode == "select_start":
            if self.grid[row, col] == OBSTACLE:
                self._set_status("⚠  Cannot place start on an obstacle! Click a white (free) cell.")
                return
            self.start_point = (row, col)
            self.lbl_start.config(text=f"Start: row={row}, col={col}")
            self._set_status(f"✓ Start set at ({row}, {col}). Now set the Goal point.")
            self.mode_var.set("view")
            self._render_grid()
            self._check_proceed_enabled()

        elif mode == "select_goal":
            if self.grid[row, col] == OBSTACLE:
                self._set_status("⚠  Cannot place goal on an obstacle! Click a white (free) cell.")
                return
            if (row, col) == self.start_point:
                self._set_status("⚠  Goal must be different from start!")
                return
            self.goal_point = (row, col)
            self.lbl_goal.config(text=f"Goal:  row={row}, col={col}")
            self._set_status(f"✓ Goal set at ({row}, {col}). Click 'Proceed to Planning'.")
            self.mode_var.set("view")
            self._render_grid()
            self._check_proceed_enabled()

    def _update_mode(self):
        mode = self.mode_var.get()
        if mode == "correct":
            self._set_status("Correction mode: click grid cells to toggle free ↔ obstacle.")
        else:
            self._set_status("View mode: use toolbar to pan/zoom. Click 'Set Start/Goal' to place points.")

    def _set_mode(self, mode: str):
        self.mode_var.set(mode)
        msgs = {
            "select_start": "Click a FREE (white) cell to set the Start point.",
            "select_goal": "Click a FREE (white) cell to set the Goal point.",
        }
        self._set_status(msgs.get(mode, ""))

    def _clear_points(self):
        self.start_point = None
        self.goal_point = None
        self.lbl_start.config(text="Start: —")
        self.lbl_goal.config(text="Goal:  —")
        self.btn_proceed.config(state=tk.DISABLED)
        self._render_grid()
        self._set_status("Points cleared. Set Start and Goal to proceed.")

    def _check_proceed_enabled(self):
        if self.start_point and self.goal_point:
            self.btn_proceed.config(state=tk.NORMAL)

    def _on_proceed(self):
        if not self.start_point or not self.goal_point:
            messagebox.showwarning("Missing Points", "Please set both Start and Goal points.")
            return

        # Final validation
        if self.grid[self.start_point[0], self.start_point[1]] == OBSTACLE:
            messagebox.showerror("Invalid Start", f"Start point {self.start_point} is on an obstacle!")
            return
        if self.grid[self.goal_point[0], self.goal_point[1]] == OBSTACLE:
            messagebox.showerror("Invalid Goal", f"Goal point {self.goal_point} is on an obstacle!")
            return

        self._confirmed = True
        logger.info(f"[GridUI] Confirmed — Start: {self.start_point}, Goal: {self.goal_point}")
        self.root.quit()
        self.root.destroy()

    def _on_close(self):
        if messagebox.askyesno("Exit", "Exit without planning?"):
            self._confirmed = False
            self.root.quit()
            self.root.destroy()

    def _set_status(self, msg: str):
        if self.status_var:
            self.status_var.set(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_grid_review(
    grid: np.ndarray,
    dxf_filename: str = "Floor Plan",
    classifications: Optional[dict] = None,
    bg_image_path: Optional[str] = None,
) -> tuple:
    """
    Launch the grid review UI and return (start, goal) points.

    Raises RuntimeError if the user closes the window without confirming.
    """
    ui = GridReviewUI(grid, dxf_filename, classifications, bg_image_path)
    return ui.show()
