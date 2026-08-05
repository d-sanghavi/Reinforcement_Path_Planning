"""
live_viz.py
───────────
Real-time Pygame live training visualization for the DDQN path planner.

Renders a live window showing the agent physically moving through the grid
across all training episodes. Falls back to matplotlib if Pygame is not installed.

Features:
  - Grid rendered once as a texture (fast: no per-frame grid recompute)
  - Agent shown as a colored circle moving in real time
  - Episode trail shown as a fading path (recent steps brighter)
  - A* path overlaid in green once background thread completes
  - HUD: Episode #, Reward, Epsilon, Steps, A* Status
  - Render speed controlled by frame_skip parameter
  - Clean shutdown on window close or training complete

Usage (called from run_agent.py):
    viz = LiveTrainingViz(grid, start, goal, frame_skip=5)
    viz.start()
    for episode in range(n_episodes):
        ...
        viz.step_render(agent_pos, episode, reward, epsilon, done)
    viz.show_final(astar_path)
    viz.close()
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Colors (RGB) ──────────────────────────────────────────────────────────────
C_BACKGROUND  = (13,  13,  26)   # very dark navy
C_FREE        = (220, 220, 228)  # near-white
C_OBSTACLE    = (30,  30,  50)   # dark blue-grey
C_AGENT       = (78,  201, 255)  # bright cyan (agent current position)
C_TRAIL       = (60,  120, 200)  # mid-blue (path trail)
C_START       = (0,   210, 80)   # vivid green
C_GOAL        = (230, 40,  60)   # bright red
C_ASTAR       = (0,   210, 80)   # green (A* optimal path)
C_HUD_BG      = (10,  10,  20)   # HUD panel background
C_HUD_TEXT    = (200, 200, 240)  # HUD main text
C_HUD_ACCENT  = (78,  201, 255)  # HUD highlight
C_DOOR        = (255, 200, 80)   # amber (door cells)

# ── Window sizing ─────────────────────────────────────────────────────────────
MAX_WINDOW_W = 1280
MAX_WINDOW_H = 800
HUD_HEIGHT   = 90   # pixels reserved at top for HUD


class LiveTrainingViz:
    """
    Pygame-based real-time DDQN training visualization.

    Parameters
    ----------
    grid : np.ndarray
        Binary occupancy grid (0=free, 1=obstacle).
    start : tuple
        (row, col) start cell.
    goal : tuple
        (row, col) goal cell.
    frame_skip : int
        Render only every N steps (higher = faster training, less smooth viz).
        Set to 1 for maximum detail, 20+ for large grids.
    door_cells : np.ndarray or None
        Boolean mask of door cells for highlight overlay (optional).
    """

    def __init__(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple,
        frame_skip: int = 5,
        door_cells: Optional[np.ndarray] = None,
        bg_image_path: Optional[str] = None,
    ):
        self.grid = grid.astype(np.uint8)
        self._start_pos = start   # named _start_pos to avoid shadowing the start() method
        self._goal_pos = goal
        self.frame_skip = max(1, frame_skip)
        self.door_cells = door_cells
        self.bg_image_path = bg_image_path
        self.fast_forward = False

        self.rows, self.cols = grid.shape
        self._step_count = 0
        self._closed = False
        self._astar_path: list = []
        self._trail_ddqn: list = []
        self._trail_ppoa: list = []
        self.current_agent = "DDQN"

        # Current HUD state
        self._hud = {
            "episode": 0,
            "total_eps": 0,
            "ddqn_reward": 0.0,
            "ppoa_reward": 0.0,
            "epsilon": 1.0,
            "steps": 0,
            "astar_done": False,
            "astar_cells": 0,
        }

        # Pygame surface references (populated in start())
        self._screen = None
        self._grid_surface = None
        self._cell_w = 1
        self._cell_h = 1
        self._grid_area_h = 0
        self._font_large = None
        self._font_small = None
        self._use_pygame = False
        self._bg_surface = None

        # matplotlib fallback state
        self._mpl_fig = None
        self._mpl_ax = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, total_episodes: int = 300):
        """Open the visualization window. Call before the training loop."""
        self._hud["total_eps"] = total_episodes
        self._use_pygame = self._init_pygame()
        if not self._use_pygame:
            self._init_matplotlib_fallback()

    def close(self):
        """Close the window cleanly."""
        if self._closed:
            return
        self._closed = True
        if self._use_pygame:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
        elif self._mpl_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._mpl_fig)
            except Exception:
                pass

    # ── Per-step render call ──────────────────────────────────────────────────

    def step_render(
        self,
        positions: dict,
        episode: int,
        rewards: dict,
        epsilon: float,
        dones: dict,
    ):
        """
        Call this inside the training step loop.
        Only actually renders every frame_skip steps (or when episode ends).
        """
        if self._closed:
            return

        self._step_count += 1
        
        if "DDQN" in positions and not dones.get("DDQN", False):
            self._trail_ddqn.append(positions["DDQN"])
            if len(self._trail_ddqn) > 200: self._trail_ddqn = self._trail_ddqn[-200:]
            
        if "PPOA" in positions and not dones.get("PPOA", False):
            self._trail_ppoa.append(positions["PPOA"])
            if len(self._trail_ppoa) > 200: self._trail_ppoa = self._trail_ppoa[-200:]

        self._hud.update({
            "episode": episode,
            "ddqn_reward": rewards.get("DDQN", 0.0),
            "ppoa_reward": rewards.get("PPOA", 0.0),
            "epsilon": epsilon,
            "steps": self._step_count,
        })

        done = dones.get("DDQN", False) and dones.get("PPOA", False)
        should_render = (self._step_count % self.frame_skip == 0) or done
        if not should_render:
            return
            
        if self.fast_forward and not done:
            return

        if self._use_pygame:
            self._pygame_render(positions)
        else:
            self._mpl_render(positions)

    def new_episode(self):
        """Call at the start of each new episode to clear the trail."""
        self._trail_ddqn = []
        self._trail_ppoa = []
        self._step_count = 0

    def set_astar_path(self, path: list):
        """Set the A* optimal path for overlay rendering."""
        self._astar_path = path
        self._hud["astar_done"] = True
        self._hud["astar_cells"] = len(path)

    def show_final(self, astar_path: list, hold_secs: float = 3.0):
        """
        Show the final optimal path and hold the window for hold_secs.
        Called after all training episodes complete.
        """
        if self._closed:
            return

        self._astar_path = astar_path
        self._trail_ddqn = []
        self._trail_ppoa = []

        if self._use_pygame:
            self._pygame_render({"DDQN": self._goal_pos, "PPOA": self._goal_pos})
            self._pygame_final_overlay(astar_path)
            time.sleep(hold_secs)
        else:
            self._mpl_render({"DDQN": self._goal_pos, "PPOA": self._goal_pos})
            import matplotlib.pyplot as plt
            plt.pause(hold_secs)

        self.close()

    # =========================================================================
    # PYGAME IMPLEMENTATION
    # =========================================================================

    def _init_pygame(self) -> bool:
        """Initialize Pygame. Returns True on success, False if unavailable."""
        try:
            import pygame
            pygame.init()

            # Compute cell size to fit the grid in the window
            avail_h = MAX_WINDOW_H - HUD_HEIGHT
            cell_h = max(1, avail_h // self.rows)
            cell_w = max(1, MAX_WINDOW_W // self.cols)
            self._cell_w = min(cell_w, cell_h)
            self._cell_h = self._cell_w

            win_w = self.cols * self._cell_w
            win_h = self.rows * self._cell_h + HUD_HEIGHT
            win_w = max(win_w, 400)

            self._screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("CAD-to-Grid DDQN Live Training")

            # Fonts
            self._font_large = pygame.font.SysFont("Consolas", 16, bold=True)
            self._font_small = pygame.font.SysFont("Consolas", 12)

            self._grid_area_h = self.rows * self._cell_h
            
            # Load background image if provided
            if self.bg_image_path:
                try:
                    img = pygame.image.load(self.bg_image_path).convert()
                    self._bg_surface = pygame.transform.smoothscale(img, (win_w, self._grid_area_h))
                except Exception as e:
                    logger.warning(f"[LiveViz] Could not load background image: {e}")

            # Pre-render static grid surface (only once)
            self._grid_surface = self._build_grid_surface(pygame)

            self._screen.fill(C_BACKGROUND)
            if self._bg_surface:
                self._screen.blit(self._bg_surface, (0, HUD_HEIGHT))
            self._screen.blit(self._grid_surface, (0, HUD_HEIGHT))
            pygame.display.flip()

            logger.info(f"[LiveViz] Pygame window opened: {win_w}x{win_h}px, cell={self._cell_w}px")
            return True

        except ImportError:
            logger.warning(
                "[LiveViz] pygame / pygame-ce not installed — falling back to matplotlib.\n"
                "  Install with: python -m pip install pygame-ce  (Python 3.14 compatible)"
            )
            return False
        except Exception as e:
            logger.warning(f"[LiveViz] pygame init failed ({e}) — falling back to matplotlib")
            return False

    def _build_grid_surface(self, pygame):
        """Pre-render the static grid as a Pygame surface."""
        surf_w = self.cols * self._cell_w
        surf_h = self.rows * self._cell_h
        # Use SRCALPHA for transparency support
        surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        
        # If we have a background image, free space is fully transparent. Otherwise, C_FREE.
        bg_color = (0, 0, 0, 0) if self._bg_surface else C_FREE
        surf.fill(bg_color)

        cw, ch = self._cell_w, self._cell_h

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    # Obstacle
                    rect = pygame.Rect(c * cw, r * ch, cw, ch)
                    color = (*C_OBSTACLE, 180) if self._bg_surface else C_OBSTACLE
                    pygame.draw.rect(surf, color, rect)

                # Tint door cells amber
                if self.door_cells is not None and self.door_cells[r, c]:
                    rect = pygame.Rect(c * cw, r * ch, cw, ch)
                    rect = rect.inflate(-max(2, cw//4), -max(2, ch//4))
                    color = (*C_DOOR, 180) if self._bg_surface else C_DOOR
                    pygame.draw.rect(surf, color, rect)

        return surf

    def _pygame_render(self, positions: dict):
        """Full Pygame frame render."""
        import pygame

        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_f, pygame.K_SPACE):
                    self.fast_forward = not self.fast_forward
                    logger.info(f"[LiveViz] Fast-Forward: {'ON' if self.fast_forward else 'OFF'}")

        cw, ch = self._cell_w, self._cell_h

        # 1. Draw static grid
        self._screen.fill(C_BACKGROUND)
        if self._bg_surface:
            self._screen.blit(self._bg_surface, (0, HUD_HEIGHT))
        self._screen.blit(self._grid_surface, (0, HUD_HEIGHT))

        # 2. Draw A* path overlay
        if self._astar_path and len(self._astar_path) > 1:
            points = [(c * cw + cw // 2, r * ch + HUD_HEIGHT + ch // 2) for r, c in self._astar_path]
            pygame.draw.lines(self._screen, (0, 210, 80), False, points, max(2, cw // 2))

        # 3. Draw trail (fading smooth lines)
        for trail, base_color in [(self._trail_ddqn, C_TRAIL), (self._trail_ppoa, (180, 50, 200))]:
            if len(trail) > 1:
                for i in range(len(trail) - 1):
                    tr1, tc1 = trail[i]
                    tr2, tc2 = trail[i+1]
                    alpha = int(80 + 150 * (i / max(1, len(trail))))
                    p1 = (tc1 * cw + cw // 2, tr1 * ch + HUD_HEIGHT + ch // 2)
                    p2 = (tc2 * cw + cw // 2, tr2 * ch + HUD_HEIGHT + ch // 2)
                    pygame.draw.line(self._screen, base_color, p1, p2, max(2, cw // 3))

        # 4. Draw glowing start / goal markers
        import math
        sr, sc = self._start_pos
        gr, gc = self._goal_pos
        radius = max(3, cw // 2)
        glow_radius = radius + 2 + int(2 * math.sin(time.time() * 6))
        
        # Start
        sp = (int(sc * cw + cw // 2), int(sr * ch + HUD_HEIGHT + ch // 2))
        pygame.draw.circle(self._screen, (0, 255, 100), sp, glow_radius, width=1)
        pygame.draw.circle(self._screen, C_START, sp, radius)
        
        # Goal
        gp = (int(gc * cw + cw // 2), int(gr * ch + HUD_HEIGHT + ch // 2))
        pygame.draw.circle(self._screen, (255, 100, 100), gp, glow_radius, width=1)
        pygame.draw.circle(self._screen, C_GOAL, gp, radius)

        # 5. Draw agents
        for agent_type, pos in positions.items():
            ar, ac = pos
            agent_color = C_AGENT if agent_type == "DDQN" else (220, 100, 255)
            pygame.draw.circle(
                self._screen, agent_color,
                (int(ac * cw + cw // 2), int(ar * ch + HUD_HEIGHT + ch // 2)),
                max(2, radius + 1),
            )

        # 6. Draw HUD
        self._draw_hud(pygame)

        pygame.display.flip()

    def _draw_hud(self, pygame):
        """Render the top HUD bar."""
        h = self._hud
        win_w = self._screen.get_width()

        pygame.draw.rect(self._screen, C_HUD_BG, pygame.Rect(0, 0, win_w, HUD_HEIGHT))
        pygame.draw.line(self._screen, C_HUD_ACCENT, (0, HUD_HEIGHT - 1), (win_w, HUD_HEIGHT - 1), 1)

        title = self._font_large.render("DDQN Live Training", True, C_HUD_ACCENT)
        self._screen.blit(title, (10, 8))

        ep_pct = int(100 * h["episode"] / max(1, h["total_eps"]))
        astar_str = f"A* {h['astar_cells']} cells DONE" if h["astar_done"] else "A* computing..."
        astar_color = (0, 210, 80) if h["astar_done"] else (200, 140, 40)

        stats = [
            (f"Ep: {h['episode']:>4}/{h['total_eps']} ({ep_pct}%)", C_HUD_TEXT),
            (f"DDQN R: {h['ddqn_reward']:>5.0f} | PPOA R: {h['ppoa_reward']:>5.0f}", C_HUD_TEXT),
            (f"Eps: {h['epsilon']:.2f}", C_HUD_TEXT),
            (f"Steps: {h['steps']:>4}", C_HUD_TEXT),
            (astar_str, astar_color),
        ]

        # Use slightly varied spacing to fit the DDQN/PPOA combined reward string
        offsets = [10, 200, 480, 600, 720]
        for i, (text, color) in enumerate(stats):
            surf = self._font_small.render(text, True, color)
            x_pos = offsets[i] if i < len(offsets) else offsets[-1] + 120
            self._screen.blit(surf, (x_pos, 38))

        # Progress bar
        bar_y = HUD_HEIGHT - 16
        bar_w = win_w - 20
        pygame.draw.rect(self._screen, (40, 40, 60), pygame.Rect(10, bar_y, bar_w, 8))
        fill_w = int(bar_w * ep_pct / 100)
        if fill_w > 0:
            pygame.draw.rect(self._screen, C_HUD_ACCENT, pygame.Rect(10, bar_y, fill_w, 8))

    def _pygame_final_overlay(self, astar_path: list):
        """Show the final path with a 'Complete!' banner."""
        import pygame

        cw, ch = self._cell_w, self._cell_h

        # Bright green final path
        for r, c in astar_path:
            pygame.draw.rect(
                self._screen,
                (0, 230, 80),
                pygame.Rect(c * cw, r * ch + HUD_HEIGHT, max(2, cw), max(2, ch)),
            )

        # "Training Complete" banner
        banner = self._font_large.render(
            f"Training Complete!  Path: {len(astar_path)} cells (A* Optimal)",
            True, (255, 220, 50),
        )
        bx = (self._screen.get_width() - banner.get_width()) // 2
        self._screen.blit(banner, (bx, HUD_HEIGHT + 20))

        pygame.display.flip()

    # =========================================================================
    # MATPLOTLIB FALLBACK IMPLEMENTATION
    # =========================================================================

    def _init_matplotlib_fallback(self):
        """Initialize matplotlib interactive mode as fallback."""
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            self._mpl_fig, self._mpl_ax = plt.subplots(figsize=(10, 7), facecolor="#0d0d1a")
            self._mpl_ax.set_facecolor("#0d0d1a")
            plt.ion()  # interactive mode
            plt.show(block=False)

            logger.info("[LiveViz] Using matplotlib fallback (slower, no Pygame)")
        except Exception as e:
            logger.error(f"[LiveViz] matplotlib fallback failed: {e}")
            self._closed = True

    def _mpl_render(self, positions: dict):
        """Lightweight matplotlib render update."""
        try:
            import matplotlib.pyplot as plt

            ax = self._mpl_ax
            ax.cla()
            ax.set_facecolor("#0d0d1a")

            # Grid image
            rows, cols = self.grid.shape
            rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
            rgb[self.grid == 0] = [220, 220, 228]
            rgb[self.grid == 1] = [30, 30, 50]

            # A* path
            if self._astar_path:
                for r, c in self._astar_path:
                    rgb[r, c] = [0, 210, 80]

            # Trail
            for r, c in self._trail_ddqn[-50:]:
                rgb[r, c] = [60, 120, 200]
            for r, c in self._trail_ppoa[-50:]:
                rgb[r, c] = [180, 50, 200]

            # Start / goal
            sr, sc = self._start_pos
            gr, gc = self._goal_pos
            rgb[sr, sc] = [0, 210, 80]
            rgb[gr, gc] = [230, 40, 60]

            # Agents
            for agent_type, pos in positions.items():
                ar, ac = pos
                rgb[ar, ac] = [78, 201, 255] if agent_type == "DDQN" else [220, 100, 255]

            ax.imshow(rgb, aspect="auto", interpolation="nearest")
            h = self._hud
            ax.set_title(
                f"Ep {h['episode']}/{h['total_eps']} | "
                f"DDQN R: {h['ddqn_reward']:.0f} | PPOA R: {h['ppoa_reward']:.0f} | ε: {h['epsilon']:.2f}",
                color="#a0c0ff", fontsize=9,
            )
            ax.axis("off")

            self._mpl_fig.canvas.draw_idle()
            plt.pause(0.001)

        except Exception as e:
            logger.debug(f"[LiveViz] matplotlib render error: {e}")
