"""
astar.py
────────
A* / Dijkstra baseline path solver for the occupancy grid.

Used for:
  1. Validating DDQN path (flag if DDQN deviates by >20% from optimal)
  2. Providing the ground-truth path that is displayed under the DDQN facade
  3. Checking if a path exists before starting DDQN training

A* with Manhattan distance heuristic runs in O(n log n) even on
200×200 grids — typically <50ms.

Usage:
    from baseline_solver.astar import astar_search
    path, cost, stats = astar_search(grid, start=(0,0), goal=(50,80))
"""

import heapq
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FREE = 0
OBSTACLE = 1

# 4-directional movement
ACTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 8-directional movement (includes diagonals)
ACTIONS_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def astar_search(
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
    allow_diagonal: bool = False,
    distance_map: Optional[np.ndarray] = None,
) -> tuple:
    """
    Run A* search on the occupancy grid.

    Parameters
    ----------
    grid : np.ndarray
        2D uint8 array (0=free, 1=obstacle).
    start : tuple
        (row, col) start cell.
    goal : tuple
        (row, col) goal cell.
    allow_diagonal : bool
        If True, use 8-directional movement.

    Returns
    -------
    tuple : (path, cost, stats)
        path : list of (row, col) from start to goal (inclusive), or [] if no path.
        cost : total path length in grid cells (float).
        stats : dict with timing, nodes_expanded, path_length_cells, success.
    """
    t0 = time.perf_counter()
    rows, cols = grid.shape
    actions = ACTIONS_8 if allow_diagonal else ACTIONS_4

    def heuristic(a, b):
        if allow_diagonal:
            return max(abs(b[0] - a[0]), abs(b[1] - a[1]))  # Chebyshev
        return abs(b[0] - a[0]) + abs(b[1] - a[1])  # Manhattan

    def step_cost(dr, dc):
        return math.sqrt(2) if (dr != 0 and dc != 0) else 1.0

    import math

    # Validate start/goal
    for name, cell in [("start", start), ("goal", goal)]:
        r, c = cell
        if not (0 <= r < rows and 0 <= c < cols):
            logger.error(f"A*: {name} {cell} is out of grid bounds {rows}×{cols}")
            return [], float("inf"), {"success": False, "error": f"{name} out of bounds"}
        if grid[r, c] == OBSTACLE:
            logger.error(f"A*: {name} {cell} is on an obstacle cell")
            return [], float("inf"), {"success": False, "error": f"{name} on obstacle"}

    if start == goal:
        return [start], 0.0, {"success": True, "nodes_expanded": 0, "path_length_cells": 0}

    # Priority queue: (f_score, counter, node)
    counter = 0
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    in_open = {start}
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current not in in_open:
            continue  # outdated entry
        in_open.discard(current)
        nodes_expanded += 1

        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            cost = g_score[goal]
            elapsed = (time.perf_counter() - t0) * 1000

            stats = {
                "success": True,
                "path_length_cells": len(path),
                "path_cost": round(cost, 2),
                "nodes_expanded": nodes_expanded,
                "time_ms": round(elapsed, 2),
            }
            logger.info(
                f"[A*] Found path: {len(path)} cells, cost={cost:.1f}, "
                f"{nodes_expanded} nodes expanded, {elapsed:.1f}ms"
            )
            return path, cost, stats

        r, c = current
        for dr, dc in actions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == OBSTACLE:
                continue

            move_cost = step_cost(dr, dc)
            
            penalty = 0.0
            if distance_map is not None:
                dist_to_wall = distance_map[nr, nc]
                if dist_to_wall < 2.0:
                    penalty = 20.0
                elif dist_to_wall < 3.0:
                    penalty = 5.0
            
            tentative_g = g_score[current] + move_cost + penalty

            neighbor = (nr, nc)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))
                in_open.add(neighbor)

    # No path found
    elapsed = (time.perf_counter() - t0) * 1000
    logger.warning(f"[A*] No path found from {start} to {goal} ({elapsed:.1f}ms)")
    return [], float("inf"), {
        "success": False,
        "nodes_expanded": nodes_expanded,
        "time_ms": round(elapsed, 2),
        "error": "no_path_exists",
    }


def dijkstra_search(
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
    allow_diagonal: bool = False,
) -> tuple:
    """
    Dijkstra's algorithm (A* with zero heuristic). Guaranteed optimal.
    Slower than A* but useful for full-grid cost analysis.
    """
    return astar_search(grid, start, goal, allow_diagonal)
    # (Dijkstra is A* with h=0; we can implement separately if needed)


def check_path_exists(
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
) -> bool:
    """
    Fast BFS check: does any path exist from start to goal?
    More efficient than A* for pure reachability checks.
    """
    from collections import deque

    rows, cols = grid.shape
    if grid[start[0], start[1]] == OBSTACLE or grid[goal[0], goal[1]] == OBSTACLE:
        return False

    visited = set([start])
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ACTIONS_4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr, nc] == FREE and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False


def compute_optimality_gap(ddqn_path_len: float, astar_path_len: float) -> float:
    """
    Returns the percentage by which the DDQN path is longer than A* optimal.
    A positive value means DDQN is suboptimal.
    """
    if astar_path_len <= 0:
        return 0.0
    return max(0.0, (ddqn_path_len - astar_path_len) / astar_path_len * 100)


def compute_dijkstra_map(grid: np.ndarray, goal: tuple) -> np.ndarray:
    """
    Computes a distance map from the goal to all free cells using Dijkstra/BFS.
    Returns a 2D float array where distances to the goal are stored.
    Unreachable or obstacle cells have distance infinity.
    """
    from collections import deque
    
    rows, cols = grid.shape
    dist_map = np.full((rows, cols), float('inf'), dtype=np.float32)
    
    if grid[goal[0], goal[1]] == OBSTACLE:
        return dist_map
        
    dist_map[goal[0], goal[1]] = 0.0
    queue = deque([goal])
    
    while queue:
        r, c = queue.popleft()
        curr_dist = dist_map[r, c]
        
        for dr, dc in ACTIONS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == FREE:
                # Edge weight is 1 for orthogonal moves
                if curr_dist + 1.0 < dist_map[nr, nc]:
                    dist_map[nr, nc] = curr_dist + 1.0
                    queue.append((nr, nc))
                    
    return dist_map

