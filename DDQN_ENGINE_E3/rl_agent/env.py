"""
env.py
──────
Custom Gym-style GridWorld environment for DDQN + PPO-A* path planning.

Waypoint Reward Shaping (enabled by default for both algorithms):
  During __init__, the A* optimal path is computed ONCE and subsampled to
  every 10th node (plus the goal) to create a sparse waypoint trail.

  Per-step reward composition:
    REWARD_STEP         = -1       (constant step penalty → shortest path)
    delta * 1.5                   (continuous hot/cold gradient to active WP)
    +5.0                          (waypoint reached bonus, WP_REACH_DIST=1.9)
    REWARD_GOAL         = +100    (terminal: goal reached → episode ends)
    REWARD_WALL_BOUNCE  = -3      (wall hit → position unchanged, episode CONTINUES)
    REWARD_INVALID_MOVE = -10     (boundary hit → position unchanged, episode CONTINUES)

  Episode termination: ONLY on goal_reached OR max_steps exceeded.
  Wall collisions are non-terminal: agent bounces and tries another action.

State representation: local 11×11 observation window around agent + goal direction.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FREE = 0
OBSTACLE = 1

# ── Terminal / step reward constants ─────────────────────────────────────────
REWARD_STEP          = -1
REWARD_OBSTACLE      = -100    # kept for reference (no longer triggers done)
REWARD_GOAL          = +100
REWARD_INVALID_MOVE  = -10    # boundary hit (zero positional change)
REWARD_WALL_BOUNCE   = -3     # wall hit — non-terminal, agent stays in place

# ── Waypoint shaping constants ───────────────────────────────────────────────
# Subsampled waypoints (every 3rd A* node on an inflated costmap) make the
# reward signal center-aligned in corridors — avoids corner-snagging.
WP_REACHED_REWARD  = +5.0   # reached a micro-breadcrumb waypoint
WP_REACH_DIST      = 1.5    # cells — tight radius; path is now wall-safe
WP_DELTA_SCALE     = 1.5    # multiply (prev_dist - curr_dist) each step

# ── Observation window size ───────────────────────────────────────────────────
OBS_WINDOW = 11  # must be odd

# ── Action space ─────────────────────────────────────────────────────────────
ACTIONS = {
    0: (-1,  0),  # UP
    1: ( 1,  0),  # DOWN
    2: ( 0, -1),  # LEFT
    3: ( 0,  1),  # RIGHT
}
N_ACTIONS  = len(ACTIONS)
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class GridWorldEnv:
    """
    Gym-compatible grid navigation environment with A* waypoint reward shaping.

    Parameters
    ----------
    grid : np.ndarray
        Binary occupancy grid (0=free, 1=obstacle).
    start : tuple
        (row, col) starting position.
    goal : tuple
        (row, col) goal position.
    max_steps : int
        Maximum steps per episode before truncation.
    use_potential_shaping : bool
        If True, add Euclidean distance-to-goal shaping to reward.
    use_waypoint_shaping : bool
        If True (default), compute A* path once and use it as a breadcrumb
        trail of waypoints that shape the reward signal every step.
    """

    def __init__(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple,
        max_steps: int = 10000,
        use_potential_shaping: bool = True,
        use_waypoint_shaping: bool = True,
        use_dijkstra_shaping: bool = True,
    ):
        self.grid   = grid.astype(np.uint8)
        self.start  = start
        self.goal   = goal
        self.max_steps = max_steps
        self.use_potential_shaping = use_potential_shaping
        self.use_waypoint_shaping  = use_waypoint_shaping
        self.use_dijkstra_shaping  = use_dijkstra_shaping

        self.rows, self.cols = grid.shape
        self.obs_dim = OBS_WINDOW
        self.half_w  = OBS_WINDOW // 2

        # State dimension: flattened observation window + 2 goal direction channels
        self.state_size  = OBS_WINDOW * OBS_WINDOW + 2
        self.action_size = N_ACTIONS

        # Episode state
        self.agent_pos  = None
        self.steps_taken = 0
        self.path_taken  = []

        # Optimal path length for external reference (set by run_agent)
        self._optimal_length: Optional[float] = None

        # ── Waypoint shaping: compute A* path ONCE in __init__ ───────────────
        # Stored as a SPARSE subsampled list (every 10th node + goal).
        # Per-episode tracking state is reset in reset().
        self._astar_waypoints: list  = []
        self._wp_index:        int   = 0
        self._prev_wp_dist:    float = float("inf")  # dist to active WP last step

        if use_waypoint_shaping and not use_dijkstra_shaping:
            self._compute_waypoints()
            
        # ── Dijkstra shaping: compute full distance map ONCE ───────────────────
        self.dijkstra_map: Optional[np.ndarray] = None
        self._prev_dijkstra_dist: float = float("inf")
        self.max_dijkstra_dist: float = 1.0  # Avoid division by zero
        
        if use_dijkstra_shaping:
            self._compute_dijkstra_map()

        logger.debug(
            f"[GridWorldEnv] Grid: {self.rows}×{self.cols}, "
            f"Start: {start}, Goal: {goal}, MaxSteps: {max_steps}, "
            f"DijkstraShaping: {use_dijkstra_shaping}"
        )

    # ── Waypoint helpers ──────────────────────────────────────────────────────

    def _compute_waypoints(self):
        """
        Generate center-aligned micro-breadcrumb waypoints via Cost Gradient:

        1. Compute a Distance Map from the original grid using distanceTransform.
           This map stores the distance from every cell to the nearest wall.
        2. Run A* on the original grid but pass the distance map. A* applies
           heavy penalties to cells near walls (1 or 2 cells away), forcing
           the path to route down the center of corridors.
        3. Subsample the resulting center-aligned path every 3rd node
           (micro-breadcrumbs) and always append the goal.
        """
        try:
            import cv2
            from baseline_solver.astar import astar_search

            # ── 1. Generate Distance Map ────────────────────────────────────────
            inverted_grid = (1 - self.grid).astype(np.uint8) * 255
            distance_map = cv2.distanceTransform(
                inverted_grid, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )

            # ── 2. Run A* on original grid with Soft Costmap Gradient ─────────
            path, _, stats = astar_search(self.grid, self.start, self.goal, distance_map=distance_map)

            if not path:
                # If distance penalty somehow fails, fallback
                logger.warning(
                    "[GridWorldEnv] A* with distance map found no path — "
                    "retrying without distance map."
                )
                path, _, stats = astar_search(self.grid, self.start, self.goal)

            if path:
                # ── 3. Micro-breadcrumbs: every 3rd node + goal ──────────────
                subsampled = list(path[::3])
                if subsampled[-1] != path[-1]:
                    subsampled.append(path[-1])
                self._astar_waypoints = subsampled
                logger.info(
                    f"[GridWorldEnv] Costmap A*: {len(path)} nodes → "
                    f"{len(subsampled)} micro-breadcrumbs (every 3rd + goal)"
                )
            else:
                self._astar_waypoints = []
                logger.warning(
                    "[GridWorldEnv] A* found no path — waypoint shaping disabled."
                )

        except Exception as exc:
            logger.warning(
                f"[GridWorldEnv] Waypoint A* failed ({exc}) — "
                "waypoint shaping disabled for this run."
            )
            self._astar_waypoints = []

    def _compute_dijkstra_map(self):
        """Compute the Dijkstra distance map from the goal to all free cells."""
        from baseline_solver.astar import compute_dijkstra_map
        self.dijkstra_map = compute_dijkstra_map(self.grid, self.goal)
        start_dist = self.dijkstra_map[self.start[0], self.start[1]]
        if start_dist != float('inf'):
            self.max_dijkstra_dist = start_dist
        else:
            self.max_dijkstra_dist = 1.0

    def _reset_waypoint_tracker(self):
        """Reset per-episode waypoint and dijkstra state."""
        self._wp_index = 0
        if self._astar_waypoints:
            wr, wc = self._astar_waypoints[0]
            self._prev_wp_dist = math.hypot(
                wr - self.start[0], wc - self.start[1]
            )
        else:
            self._prev_wp_dist = float("inf")
            
        if self.use_dijkstra_shaping and self.dijkstra_map is not None:
            self._prev_dijkstra_dist = self.dijkstra_map[self.start[0], self.start[1]]
        else:
            self._prev_dijkstra_dist = float("inf")

    def _waypoint_reward(self, new_pos: tuple) -> float:
        """
        Continuous delta-based waypoint reward (hot/cold mathematical gradient).

        Every step:
          delta = prev_dist_to_WP - curr_dist_to_WP
          reward += delta * WP_DELTA_SCALE (1.5)

        Moving closer → delta > 0 → positive reward (no threshold needed).
        Moving away   → delta < 0 → negative reward (mathematically eliminates camping).
        Waypoint reached (dist < 1.9 cells) → +5.0 bonus, advance to next WP,
          reset prev_dist to dist from new_pos to the new active waypoint.
        """
        if not self.use_waypoint_shaping or not self._astar_waypoints:
            return 0.0

        nr, nc = new_pos
        n_wp   = len(self._astar_waypoints)

        if self._wp_index >= n_wp:
            return 0.0

        wr, wc    = self._astar_waypoints[self._wp_index]
        curr_dist = math.hypot(wr - nr, wc - nc)

        # ── Waypoint reached? ─────────────────────────────────────────────────
        if curr_dist < WP_REACH_DIST and self._wp_index < n_wp - 1:
            # Advance to next sparse waypoint
            self._wp_index += 1
            wr_n, wc_n = self._astar_waypoints[self._wp_index]
            self._prev_wp_dist = math.hypot(wr_n - nr, wc_n - nc)
            return WP_REACHED_REWARD   # +5.0

        # ── Continuous delta reward (hot/cold gradient) ────────────────────────
        # Positive when moving toward WP, negative when moving away.
        # This completely eliminates the camping exploit: any step that does
        # not change position yields delta=0 (no delta reward) but still pays
        # the -10 boundary penalty or -3 bounce penalty.
        delta = self._prev_wp_dist - curr_dist
        self._prev_wp_dist = curr_dist
        return delta * WP_DELTA_SCALE

    def _dijkstra_reward(self, new_pos: tuple) -> float:
        """
        Continuous reward based on Dijkstra distance map.
        Positive when moving closer to the goal, negative when moving away.
        """
        if not self.use_dijkstra_shaping or self.dijkstra_map is None:
            return 0.0
            
        r, c = new_pos
        curr_dist = self.dijkstra_map[r, c]
        if curr_dist == float('inf'):
            return 0.0
            
        delta = self._prev_dijkstra_dist - curr_dist
        self._prev_dijkstra_dist = curr_dist
        
        # Scale appropriately (similar to WP_DELTA_SCALE)
        return delta * 1.5

    # ── Core Gym interface ────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment to start position. Returns initial state."""
        self.agent_pos   = list(self.start)
        self.steps_taken = 0
        self.path_taken  = [tuple(self.start)]
        self._reset_waypoint_tracker()
        return self._get_state()

    def step(self, action: int) -> tuple:
        """
        Execute one action.

        Returns
        -------
        tuple : (next_state, reward, done, info)
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"
        self.steps_taken += 1

        dr, dc = ACTIONS[action]
        old_pos = tuple(self.agent_pos)
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        # ── Boundary check ────────────────────────────────────────────────────
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # Stayed in place, small penalty — no waypoint reward
            return self._get_state(), REWARD_INVALID_MOVE, False, {"event": "boundary"}

        # ── Obstacle check — NON-TERMINAL "Bounce" ────────────────────────────
        # Agent hits a wall: apply moderate penalty, keep position unchanged.
        # Episode does NOT end. Agent must try a different direction next step.
        # Episodes only terminate on: goal reached OR max_steps exceeded.
        if self.grid[nr, nc] == OBSTACLE:
            # Do NOT update prev_wp_dist — position is unchanged, dist unchanged
            return self._get_state(), REWARD_WALL_BOUNCE, False, {"event": "bounce"}

        # ── Valid move ────────────────────────────────────────────────────────
        self.agent_pos = [nr, nc]
        self.path_taken.append((nr, nc))

        # A* waypoint breadcrumb shaping or Dijkstra shaping
        if self.use_dijkstra_shaping:
            wp_reward = self._dijkstra_reward((nr, nc))
        else:
            wp_reward = self._waypoint_reward((nr, nc))

        # Euclidean distance-to-goal potential shaping (optional, legacy)
        shaping = 0.0
        if self.use_potential_shaping and not self.use_dijkstra_shaping:
            old_dist = self._euclidean_to_goal(old_pos)
            new_dist = self._euclidean_to_goal((nr, nc))
            shaping  = 5.0 * (old_dist - new_dist)

        # ── Goal check (terminal) ─────────────────────────────────────────────
        if (nr, nc) == self.goal:
            # Terminal reward: preserved exactly — no waypoint reward added at goal
            reward = REWARD_GOAL + shaping
            return self._get_state(), reward, True, {"event": "goal_reached"}

        # ── Normal step ───────────────────────────────────────────────────────
        reward = REWARD_STEP + shaping + wp_reward
        done   = self.steps_taken >= self.max_steps

        return self._get_state(), reward, done, {
            "event": "step", "steps": self.steps_taken, "wp_reward": wp_reward
        }

    # ── State construction ────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Build the state vector:
          - N×N local grid window centered on agent (padded with obstacles at boundaries)
          - 2 normalized goal direction components (dx, dy)
        """
        r, c   = self.agent_pos
        half   = self.half_w
        obs_size = self.obs_dim

        # Extract window with border padding
        obs = np.ones((obs_size, obs_size), dtype=np.float32)  # default: obstacle
        for dr_off in range(-half, half + 1):
            for dc_off in range(-half, half + 1):
                nr_abs = r + dr_off
                nc_abs = c + dc_off
                if 0 <= nr_abs < self.rows and 0 <= nc_abs < self.cols:
                    obs[dr_off + half, dc_off + half] = float(self.grid[nr_abs, nc_abs])

        # Goal direction (normalized)
        gr, gc   = self.goal
        max_dist = math.hypot(self.rows, self.cols) + 1e-8
        goal_dr  = (gr - r) / max_dist
        goal_dc  = (gc - c) / max_dist

        state = np.concatenate([obs.flatten(), [goal_dr, goal_dc]])
        return state.astype(np.float32)

    def _euclidean_to_goal(self, pos: tuple) -> float:
        return math.hypot(self.goal[0] - pos[0], self.goal[1] - pos[1])

    @property
    def current_pos(self) -> tuple:
        return tuple(self.agent_pos)
        
    @property
    def current_dijkstra_dist(self) -> float:
        if self.use_dijkstra_shaping and self.dijkstra_map is not None:
            return self.dijkstra_map[self.agent_pos[0], self.agent_pos[1]]
        return float('inf')

    def set_optimal_length(self, length: float):
        """Set the A* optimal path length for reward normalization."""
        self._optimal_length = length

    def render_path(self) -> np.ndarray:
        """Return a colored grid image with the path taken marked in blue."""
        import cv2
        img = ((1 - self.grid) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for r, c in self.path_taken:
            img[r, c] = [150, 150, 255]  # blue-ish for path

        # Mark start and goal
        sr, sc = self.start
        gr, gc = self.goal
        img[sr, sc] = [0, 200, 0]    # green = start
        img[gr, gc] = [200, 0, 0]    # red   = goal

        return img


# ─────────────────────────────────────────────────────────────────────────────

class PPOAEnv(GridWorldEnv):
    """
    Environment tailored for PPOA*:
    - Uses the pre-computed A* path for local waypoint targeting.
    - Intermittently spawns dynamic obstacles to force dodging/replanning.
    - Inherits A* waypoint reward shaping from GridWorldEnv (use_waypoint_shaping=True).
      Since the A* path is already provided externally, it is injected directly
      into self._astar_waypoints instead of being recomputed.
    """

    def __init__(
        self,
        grid,
        start,
        goal,
        astar_path: list,
        max_steps: int = 10000,
        lookahead: int = 10,
    ):
        # Disable the parent's A* computation — we'll inject our own path.
        super().__init__(
            grid, start, goal, max_steps,
            use_potential_shaping=False,
            use_waypoint_shaping=True,
        )
        self.astar_path = astar_path
        self.lookahead  = lookahead
        self.path_idx   = 0
        self.dyn_obs    = set()
        self.collisions = 0
        self.replans    = 0

        # Inject pre-computed path as waypoints (overrides any parent A* result)
        self._astar_waypoints = list(astar_path)

    def reset(self):
        # Re-inject waypoints (in case parent reset() clears them)
        self._astar_waypoints = list(self.astar_path)
        self.path_idx   = 0
        self.dyn_obs.clear()
        self.collisions = 0
        self.replans    = 0
        super().reset()          # calls _reset_waypoint_tracker correctly
        return self._get_state()

    def step(self, action: int):
        import random

        # Spawn dynamic obstacle 2% of the time in front of agent
        if random.random() < 0.02 and len(self.dyn_obs) < 3:
            dr, dc  = ACTIONS[action]
            spawn_r = self.agent_pos[0] + dr * 2
            spawn_c = self.agent_pos[1] + dc * 2
            if (
                0 <= spawn_r < self.rows
                and 0 <= spawn_c < self.cols
                and self.grid[spawn_r, spawn_c] == FREE
                and (spawn_r, spawn_c) != self.goal
                and (spawn_r, spawn_c) != tuple(self.agent_pos)
            ):
                self.dyn_obs.add((spawn_r, spawn_c))
                self.replans += 1

        # Randomly remove dynamic obstacles
        if self.dyn_obs and random.random() < 0.05:
            self.dyn_obs.pop()

        # Temporarily inject dynamic obstacles into grid for collision logic
        for dr_r, dr_c in self.dyn_obs:
            self.grid[dr_r, dr_c] = OBSTACLE

        _, reward, done, info = super().step(action)

        # Restore grid
        for dr_r, dr_c in self.dyn_obs:
            self.grid[dr_r, dr_c] = FREE

        if info.get("event") == "collision":
            self.collisions += 1

        return self._get_state(), reward, done, info

    def _get_state(self):
        r, c = self.agent_pos

        # Advance path_idx to closest node slightly ahead
        best_dist = float("inf")
        best_idx  = self.path_idx
        for i in range(self.path_idx, min(len(self.astar_path), self.path_idx + 10)):
            pr, pc = self.astar_path[i]
            d = abs(pr - r) + abs(pc - c)
            if d < best_dist:
                best_dist = d
                best_idx  = i
        self.path_idx = best_idx

        # Local lookahead target for observation direction
        target_idx = min(len(self.astar_path) - 1, self.path_idx + self.lookahead)
        tr, tc = self.astar_path[target_idx]

        half = self.half_w
        obs  = np.ones((self.obs_dim, self.obs_dim), dtype=np.float32)
        for dr_off in range(-half, half + 1):
            for dc_off in range(-half, half + 1):
                nr_abs = r + dr_off
                nc_abs = c + dc_off
                if 0 <= nr_abs < self.rows and 0 <= nc_abs < self.cols:
                    is_obs = (
                        self.grid[nr_abs, nc_abs] == OBSTACLE
                        or (nr_abs, nc_abs) in self.dyn_obs
                    )
                    obs[dr_off + half, dc_off + half] = float(is_obs)

        max_dist = float(self.obs_dim)
        # Vector to local A* lookahead target (not global goal)
        goal_dr = np.clip((tr - r) / max_dist, -1.0, 1.0)
        goal_dc = np.clip((tc - c) / max_dist, -1.0, 1.0)

        state = np.concatenate([obs.flatten(), [goal_dr, goal_dc]])
        return state.astype(np.float32)
