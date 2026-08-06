"""
metrics_report.py
─────────────────
Compile and display the complete metrics dashboard comparing DDQN vs PPOA*.

Dashboard layout:
  Panel 1: Training Reward Curves (DDQN vs PPOA*)
  Panel 2: Training Loss Curves
  Panel 3: Path Visualization (A* vs DDQN vs PPOA*)
  Panel 4: Detailed 12-point Metrics Comparison Table
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from symbol_model.infer_symbols import get_confidence_report

logger = logging.getLogger(__name__)


def generate_metrics_dashboard(
    planning_result,
    mapper,
    grid: np.ndarray,
    classifications: dict,
    source_file: str = "unknown",
    output_path: str = "Results/metrics_dashboard.png",
    show: bool = True,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pr = planning_result

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 12), facecolor="#0d0d1a")
    fig.suptitle(
        f"CAD-to-Grid Comparative Metrics: DDQN vs. PPOA*\n{Path(source_file).name}",
        fontsize=16, fontweight="bold", color="#e0e0ff", y=0.97,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.25, wspace=0.15,
        left=0.04, right=0.96, top=0.90, bottom=0.05,
        height_ratios=[1, 1.3]
    )

    _BLUE = "#4a9eff"      # DDQN
    _PURPLE = "#b066ff"    # PPOA*
    _GREEN = "#2dc653"     # A*
    _RED = "#e94560"       
    panel_style = dict(facecolor="#12122a")

    # ── Panel 1: Training Reward Curve ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], **panel_style)
    _style_panel(ax1, "Training Progress — Episode Reward", "Episode", "Total Reward")

    if pr.ddqn_episode_rewards:
        eps = list(range(1, len(pr.ddqn_episode_rewards) + 1))
        window = max(1, len(pr.ddqn_episode_rewards) // 10)
        smoothed = np.convolve(pr.ddqn_episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window, len(pr.ddqn_episode_rewards) + 1), smoothed, color=_BLUE, linewidth=2, label="DDQN")
        
    if hasattr(pr, 'ppoa_episode_rewards') and pr.ppoa_episode_rewards:
        eps = list(range(1, len(pr.ppoa_episode_rewards) + 1))
        window = max(1, len(pr.ppoa_episode_rewards) // 10)
        smoothed = np.convolve(pr.ppoa_episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window, len(pr.ppoa_episode_rewards) + 1), smoothed, color=_PURPLE, linewidth=2, label="PPOA*")

    ax1.legend(fontsize=9)

    # ── Panel 2: Training Loss ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], **panel_style)
    _style_panel(ax2, "Network Losses (Smoothed)", "Update Step", "Loss")

    if pr.ddqn_losses:
        window = max(1, len(pr.ddqn_losses) // 20)
        smoothed = np.convolve(pr.ddqn_losses, np.ones(window)/window, mode="valid")
        ax2.semilogy(range(window, len(pr.ddqn_losses)+1), smoothed, color=_BLUE, label="DDQN Huber Loss")
        
    if hasattr(pr, 'ppoa_actor_losses') and pr.ppoa_actor_losses:
        window = max(1, len(pr.ppoa_actor_losses) // 20)
        smoothed_a = np.convolve(pr.ppoa_actor_losses, np.ones(window)/window, mode="valid")
        smoothed_c = np.convolve(pr.ppoa_critic_losses, np.ones(window)/window, mode="valid")
        ax2.semilogy(range(window, len(pr.ppoa_actor_losses)+1), smoothed_a, color=_PURPLE, label="PPOA* Actor Loss")
        ax2.semilogy(range(window, len(pr.ppoa_critic_losses)+1), smoothed_c, color=_RED, label="PPOA* Critic Loss")
        
    ax2.legend(fontsize=9)

    # ── Panel 3: Path Visualization ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], **panel_style)
    _style_panel(ax3, "Path Overlays", "Column", "Row")
    _draw_path_on_grid(ax3, grid, pr.astar_path, pr.start, pr.goal, _GREEN, "A* Optimal")
    if pr.ddqn_path:
        _draw_path_on_grid(ax3, grid, pr.ddqn_path, pr.start, pr.goal, _BLUE, "DDQN", overlay=True)
    if hasattr(pr, 'ppoa_path') and pr.ppoa_path:
        _draw_path_on_grid(ax3, grid, pr.ppoa_path, pr.start, pr.goal, _PURPLE, "PPOA*", overlay=True)

    # ── Panel 4: 12-Point Comparative Table ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1], **panel_style)
    ax4.axis("off")

    def safe_mean(l): return np.mean(l[-10:]) if l else 0.0

    ddqn_mean_r = safe_mean(pr.ddqn_episode_rewards)
    ppo_mean_r = safe_mean(getattr(pr, 'ppoa_episode_rewards', []))
    
    astar_cost = pr.astar_cost if pr.astar_cost > 0 else 1.0

    ddqn_opt = pr.ddqn_path_cost / astar_cost if pr.ddqn_path_cost != float('inf') else float('inf')
    ppo_opt = getattr(pr, 'ppoa_path_cost', float('inf')) / astar_cost
    
    ddqn_cr = (pr.ddqn_collisions / max(1, len(pr.ddqn_path))) * 100 if pr.ddqn_path else 0.0
    ppo_path_len = len(getattr(pr, 'ppoa_path', []))
    ppo_cr = (getattr(pr, 'ppoa_collisions', 0) / max(1, ppo_path_len)) * 100

    table_data = [
        ["Metric",                                  "DDQN",                   "PPOA*"],
        ["────────────────────────────────────",  "────────",               "────────"],
        ["1. Success Rate (Found Goal)",            "100%" if pr.ddqn_path_cost != float("inf") else "0%", "100%" if getattr(pr, "ppoa_path_cost", float("inf")) != float("inf") else "0%"],
        ["2. Avg Episode Reward (last 10)",         f"{ddqn_mean_r:.1f}",     f"{ppo_mean_r:.1f}"],
        ["3. Convergence Speed (Episodes)",         f"{pr.ddqn_convergence_episode}", f"{getattr(pr, 'ppoa_convergence_episode', '-')}"] ,
        ["4. Path Length (Grid Cells)",             f"{int(pr.ddqn_path_cost) if pr.ddqn_path_cost!=float('inf') else '-'}", f"{int(getattr(pr, 'ppoa_path_cost', 0)) if getattr(pr, 'ppoa_path_cost', 0)!=float('inf') else '-'}"] ,
        ["5. Planning / Training Time (s)",         f"{pr.astar_time_ms/1000:.3f}s", f"{pr.astar_time_ms/1000:.3f}s"],
        ["6. Total Navigation Time (s)",            f"{pr.ddqn_inference_time_ms/1000:.3f}s", f"{getattr(pr, 'ppoa_inference_time_ms', 0)/1000:.3f}s"] ,
        ["7. Optimality Ratio (Path/A*)",           f"{ddqn_opt:.2f}",        f"{ppo_opt:.2f}"],
        ["8. Collision Rate (%)",                   f"{ddqn_cr:.1f}%",        f"{ppo_cr:.1f}%"],
        ["9. Dyn. Obstacle Avoidance (%)",          "-",                      f"{getattr(pr, 'ppoa_dyn_avoid_rate', 0.0):.1f}%"],
        ["10. Path Smoothness (Heading Δ)",         f"{pr.ddqn_smoothness:.0f}°", f"{getattr(pr, 'ppoa_smoothness', 0):.0f}°"],
        ["11. Replanning Frequency",                "-",                      f"{getattr(pr, 'ppoa_replans', 0)} times"],
        ["12. CPU Peak Usage (%)",                  f"{pr.ddqn_cpu_peak:.1f}%", f"{getattr(pr, 'ppoa_cpu_peak', 0.0):.1f}%"],
        ["13. Memory Peak (MB)",                    f"{pr.ddqn_mem_peak:.1f}MB", f"{getattr(pr, 'ppoa_mem_peak', 0.0):.1f}MB"],
    ]

    y_start = 0.95
    line_height = 0.055

    for i, row in enumerate(table_data):
        y = y_start - i * line_height
        color_key = "#ffffff" if i == 0 else ("#808090" if i == 1 else "#c0c0d0")
        weight = "bold" if i == 0 else "normal"

        if i < 2:
            ax4.text(0.02, y, row[0], transform=ax4.transAxes, fontsize=11, color=color_key, fontweight=weight, va="top")
            ax4.text(0.55, y, row[1], transform=ax4.transAxes, fontsize=11, color=color_key, fontweight=weight, va="top")
            ax4.text(0.75, y, row[2], transform=ax4.transAxes, fontsize=11, color=color_key, fontweight=weight, va="top")
        else:
            ax4.text(0.02, y, row[0], transform=ax4.transAxes, fontsize=11, color=color_key, fontweight=weight, va="top")
            ax4.text(0.50, y, "--", transform=ax4.transAxes, fontsize=11, color="#808090", fontweight=weight, va="top")
            ax4.text(0.55, y, row[1], transform=ax4.transAxes, fontsize=11, color=_BLUE, fontweight=weight, va="top")
            ax4.text(0.70, y, "--", transform=ax4.transAxes, fontsize=11, color="#808090", fontweight=weight, va="top")
            ax4.text(0.75, y, row[2], transform=ax4.transAxes, fontsize=11, color=_PURPLE, fontweight=weight, va="top")

    ax4.set_title("12-Point Comparative Dashboard", fontsize=12, color="#e0e0ff", pad=8)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    logger.info(f"[Metrics] Comparative Dashboard saved: {output_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass 

    plt.close(fig)
    return str(output_path)


def _style_panel(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor("#12122a")
    ax.set_title(title, fontsize=11, color="#e0e0ff", pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#888888")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#888888")
    ax.tick_params(colors="#666680", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, color="#1e1e3a", linewidth=0.5, alpha=0.7)


def _draw_path_on_grid(ax, grid, path, start, goal, color, label, overlay=False):
    if not overlay:
        rows, cols = grid.shape
        img = np.zeros((rows, cols, 3), dtype=np.uint8)
        img[grid == 0] = [200, 200, 210]  
        img[grid == 1] = [20, 20, 35]     
        ax.imshow(img, aspect="auto", interpolation="nearest")

    if path:
        cols_path = [c for r, c in path]
        rows_path = [r for r, c in path]
        ax.plot(cols_path, rows_path, color=color, linewidth=2, label=label, zorder=5)

        ax.plot(cols_path[0], rows_path[0], "o", color="#2dc653", markersize=8, zorder=6)
        ax.plot(cols_path[-1], rows_path[-1], "*", color="#e63946", markersize=10, zorder=6)

    ax.legend(fontsize=9, loc="upper right")
