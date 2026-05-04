# env.py — Episode 2 pygame environment, extended for E1 vs E2 comparison.
#
# Extra public method added:
#   draw_dual_routes(ql_route, dqn_route)
#       Draws both agents' best paths on the same grid:
#           Q-Learning  →  BLUE  circles
#           DQN         →  ORANGE circles
#       Then blocks until the user closes the window.

import numpy as np
import pygame
import sys
from collections import deque

# ── Grid / window constants ──────────────────────────────────────────────── #
GRID_SIZE      = 15
CELL_SIZE      = 40
GRID_PIXEL     = GRID_SIZE * CELL_SIZE       # 600 px
SIDEBAR_WIDTH  = 160
WINDOW_WIDTH   = GRID_PIXEL + SIDEBAR_WIDTH
WINDOW_HEIGHT  = GRID_PIXEL
BOTTOM_BAR     = 40
FULL_HEIGHT    = WINDOW_HEIGHT + BOTTOM_BAR

OBSTACLE_SIZE       = 40
COVERAGE_THRESHOLD  = 0.15

# ── Colours ──────────────────────────────────────────────────────────────── #
WHITE      = (255, 255, 255)
GREY       = (180, 180, 180)
DARK_GREY  = (60,  60,  60)
BLACK      = (0,   0,   0)
GREEN      = (50,  200, 50)
RED        = (200, 50,  50)
YELLOW     = (255, 220, 0)
BLUE       = (50,  100, 220)
LIGHT_BLUE = (173, 216, 230)
ORANGE     = (255, 140, 0)
SIDEBAR_BG = (30,  30,  40)

# ── Cell types ───────────────────────────────────────────────────────────── #
EMPTY    = 0
OBSTACLE = 1
START    = 2
GOAL     = 3

# ── Actions ──────────────────────────────────────────────────────────────── #
UP    = 0
DOWN  = 1
RIGHT = 2
LEFT  = 3

# Global best-route dict updated by env.final()
a = {}

OBSTACLE_IMAGE_FILES = [
    "road_closed1.png", "road_closed2.png", "road_closed3.png",
    "tree1.png",        "tree2.png",
    "building1.png",    "building2.png",
    "traffic_lights.png", "pedestrian.png",
    "shop.png",         "bank1.png",        "bank2.png",
]

# ── n_actions exposed so QLearningTable init works identically to E1 ────── #
N_ACTIONS    = 4
ACTION_SPACE = ['up', 'down', 'right', 'left']


class Environment:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WINDOW_WIDTH, FULL_HEIGHT))
        pygame.display.set_caption("E1 vs E2 — Comparison")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)

        # RL compatibility shim
        self.n_actions    = N_ACTIONS
        self.action_space = ACTION_SPACE

        self.grid           = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.cell_image_map = {}
        self.start_pos      = None
        self.goal_pos       = None
        self.agent_pos      = None

        # Route tracking (reset between agents)
        self.episode_route = {}
        self.episode_step  = 0
        self.best_route    = {}
        self.first_goal    = True
        self.shortest      = 0
        self.longest       = 0

        # Placement state
        self.placed_obstacles = []
        self.obstacle_count   = 0
        self.max_obstacles    = 20
        self.placed_markers   = []
        self.selected         = None
        self.dragging         = None
        self.drag_offset      = (0, 0)

        self.status_msg   = ("Select item from sidebar. "
                             "Click canvas to place. Drag to reposition.")
        self.status_color = WHITE

        # Training HUD
        self.current_episode = 0
        self.total_episodes  = 0
        self.current_epsilon = 1.0

        self._load_images()
        self._build_sidebar()

    # ── Image loading ─────────────────────────────────────────────────────── #
    def _load_images(self):
        self.obstacle_canvas_surfaces = []
        self.obstacle_cell_surfaces   = []
        self.sidebar_surfaces         = []
        self.agent_surface            = None
        self.flag_surface             = None

        thumb = (60, 40)
        for fname in OBSTACLE_IMAGE_FILES:
            try:
                img = pygame.image.load(f"images/{fname}").convert_alpha()
                self.obstacle_canvas_surfaces.append(
                    pygame.transform.scale(img, (OBSTACLE_SIZE, OBSTACLE_SIZE)))
                self.obstacle_cell_surfaces.append(
                    pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE)))
                self.sidebar_surfaces.append(
                    pygame.transform.scale(img, thumb))
            except Exception:
                for lst, sz in [
                    (self.obstacle_canvas_surfaces, (OBSTACLE_SIZE, OBSTACLE_SIZE)),
                    (self.obstacle_cell_surfaces,   (CELL_SIZE, CELL_SIZE)),
                    (self.sidebar_surfaces,          thumb),
                ]:
                    s = pygame.Surface(sz, pygame.SRCALPHA)
                    s.fill((0, 191, 255, 200))
                    lst.append(s)

        for fname, attr in [("images/agent1.png", "agent_surface"),
                             ("images/flag.png",   "flag_surface")]:
            try:
                img = pygame.image.load(fname).convert_alpha()
                setattr(self, attr,
                        pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE)))
            except Exception:
                setattr(self, attr, None)

    # ── Sidebar ───────────────────────────────────────────────────────────── #
    def _build_sidebar(self):
        self.sidebar_items = []
        x, y = GRID_PIXEL + 10, 10

        for label, kind in [("START", "start"), ("GOAL", "goal")]:
            r = pygame.Rect(x, y, SIDEBAR_WIDTH - 20, 30)
            self.sidebar_items.append({'label': label, 'rect': r, 'type': kind})
            y += 38

        self.obs_label_y = y
        y += 18
        col_w, col_gap, row_h, row_gap = 60, 20, 40, 10
        start_y = y
        for i in range(len(OBSTACLE_IMAGE_FILES)):
            col = i % 2
            row = i // 2
            r   = pygame.Rect(x + col * (col_w + col_gap),
                              start_y + row * (row_h + row_gap),
                              col_w, row_h)
            self.sidebar_items.append({'label': '', 'rect': r,
                                       'type': 'obstacle', 'idx': i})

        self.done_button_rect = pygame.Rect(x, WINDOW_HEIGHT - 60,
                                            SIDEBAR_WIDTH - 20, 40)

    # ── Phase 1 — placement loop ──────────────────────────────────────────── #
    def run_placement_phase(self):
        running = True
        while running:
            self.clock.tick(30)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self._handle_mousedown(*ev.pos) == 'done':
                        running = False
                if ev.type == pygame.MOUSEMOTION and self.dragging:
                    ox, oy = self.drag_offset
                    self.dragging['rect'].x = ev.pos[0] - ox
                    self.dragging['rect'].y = ev.pos[1] - oy
                    self.dragging['rect'].clamp_ip(
                        pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging = None
            self._draw_placement()

    def _handle_mousedown(self, mx, my):
        if self.done_button_rect.collidepoint(mx, my):
            return 'done' if self._handle_done() else None

        for item in self.sidebar_items:
            if item['rect'].collidepoint(mx, my):
                self.selected = ('start' if item['type'] == 'start' else
                                 'goal'  if item['type'] == 'goal'  else item['idx'])
                return

        if mx < GRID_PIXEL and my < WINDOW_HEIGHT:
            for item in reversed(self.placed_obstacles + self.placed_markers):
                if item['rect'].collidepoint(mx, my):
                    self.dragging    = item
                    self.drag_offset = (mx - item['rect'].x, my - item['rect'].y)
                    return

            if self.selected == 'start':
                self.placed_markers = [m for m in self.placed_markers
                                       if m['type'] != 'start']
                r = pygame.Rect(mx - CELL_SIZE // 2, my - CELL_SIZE // 2,
                                CELL_SIZE, CELL_SIZE)
                r.clamp_ip(pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                self.placed_markers.append({'type': 'start', 'rect': r})
                self.status_msg, self.status_color = "START placed.", GREEN

            elif self.selected == 'goal':
                self.placed_markers = [m for m in self.placed_markers
                                       if m['type'] != 'goal']
                r = pygame.Rect(mx - CELL_SIZE // 2, my - CELL_SIZE // 2,
                                CELL_SIZE, CELL_SIZE)
                r.clamp_ip(pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                self.placed_markers.append({'type': 'goal', 'rect': r})
                self.status_msg, self.status_color = "GOAL placed.", GREEN

            elif isinstance(self.selected, int):
                if self.obstacle_count >= self.max_obstacles:
                    self.status_msg  = f"Max {self.max_obstacles} obstacles reached!"
                    self.status_color = RED
                    return
                idx  = self.selected
                r    = pygame.Rect(mx - OBSTACLE_SIZE // 2,
                                   my - OBSTACLE_SIZE // 2,
                                   OBSTACLE_SIZE, OBSTACLE_SIZE)
                r.clamp_ip(pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                self.placed_obstacles.append(
                    {'img_idx': idx, 'rect': r,
                     'surface': self.obstacle_canvas_surfaces[idx]})
                self.obstacle_count += 1
                self.status_msg   = (f"Obstacle placed "
                                     f"({self.obstacle_count}/{self.max_obstacles}).")
                self.status_color = WHITE
            else:
                self.status_msg   = "Select an item from the sidebar first."
                self.status_color = ORANGE

    def _handle_done(self):
        starts = [m for m in self.placed_markers if m['type'] == 'start']
        goals  = [m for m in self.placed_markers if m['type'] == 'goal']
        if not starts:
            self.status_msg, self.status_color = "Please place START first.", RED
            return False
        if not goals:
            self.status_msg, self.status_color = "Please place GOAL first.", RED
            return False
        self._build_grid_from_canvas(starts[0], goals[0])
        if self.start_pos is None:
            self.status_msg, self.status_color = "START not on canvas.", RED
            return False
        if self.goal_pos is None:
            self.status_msg, self.status_color = "GOAL not on canvas.", RED
            return False
        if self.start_pos == self.goal_pos:
            self.status_msg, self.status_color = "START and GOAL on same cell.", RED
            return False
        if not self._bfs_check():
            self.status_msg, self.status_color = (
                "No valid path! Reposition obstacles/START/GOAL.", RED)
            return False
        self.status_msg, self.status_color = "Path found! Starting training...", GREEN
        self._draw_placement()
        pygame.display.flip()
        pygame.time.wait(800)
        return True

    # ── Grid builder ─────────────────────────────────────────────────────── #
    def _build_grid_from_canvas(self, start_marker, goal_marker):
        self.grid       = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.start_pos  = None
        self.goal_pos   = None
        cell_area       = CELL_SIZE * CELL_SIZE

        for obs in self.placed_obstacles:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    cell_rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE,
                                            CELL_SIZE, CELL_SIZE)
                    inter = cell_rect.clip(obs['rect'])
                    if inter.width > 0 and inter.height > 0:
                        if (inter.width * inter.height) / cell_area >= COVERAGE_THRESHOLD:
                            self.grid[row][col] = OBSTACLE

        for marker, cell_type, attr in [
            (start_marker, START, 'start_pos'),
            (goal_marker,  GOAL,  'goal_pos'),
        ]:
            cx, cy = marker['rect'].centerx, marker['rect'].centery
            if 0 <= cx < GRID_PIXEL and 0 <= cy < WINDOW_HEIGHT:
                r, c = cy // CELL_SIZE, cx // CELL_SIZE
                self.grid[r][c] = cell_type
                setattr(self, attr, (r, c))

    def _bfs_check(self):
        q, vis = deque([self.start_pos]), {self.start_pos}
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal_pos:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                        and (nr, nc) not in vis
                        and self.grid[nr][nc] != OBSTACLE):
                    vis.add((nr, nc))
                    q.append((nr, nc))
        return False

    # ── Training interface ────────────────────────────────────────────────── #
    def reset(self):
        self.agent_pos     = self.start_pos
        self.episode_route = {0: self.start_pos}
        self.episode_step  = 1
        return self.start_pos

    def step(self, action):
        r, c   = self.agent_pos
        nr, nc = r, c
        if   action == UP:    nr -= 1
        elif action == DOWN:  nr += 1
        elif action == RIGHT: nc += 1
        elif action == LEFT:  nc -= 1

        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
            return self.agent_pos, -1, False

        next_pos = (nr, nc)
        cell     = self.grid[nr][nc]

        if cell == OBSTACLE:
            return next_pos, -1, True

        if next_pos == self.goal_pos:
            self.agent_pos = next_pos
            self.episode_route[self.episode_step] = next_pos
            self.episode_step += 1
            if self.first_goal:
                self.best_route = dict(self.episode_route)
                self.shortest   = len(self.episode_route)
                self.longest    = len(self.episode_route)
                self.first_goal = False
            else:
                if len(self.episode_route) < len(self.best_route):
                    self.best_route = dict(self.episode_route)
                    self.shortest   = len(self.episode_route)
                if len(self.episode_route) > self.longest:
                    self.longest    = len(self.episode_route)
            return next_pos, 10, True

        self.agent_pos = next_pos
        self.episode_route[self.episode_step] = next_pos
        self.episode_step += 1
        return next_pos, -0.01, False

    def render(self, episode=0, total=0, epsilon=1.0):
        self.current_episode = episode
        self.total_episodes  = total
        self.current_epsilon = epsilon
        self._draw_training()
        self.clock.tick(0)

    def pump(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

    def final(self):
        global a
        a = dict(self.best_route)
        print(f"  Shortest route : {self.shortest} steps")
        print(f"  Longest route  : {self.longest} steps")
        self._draw_single_route(self.best_route, BLUE, "-- ROUTE FOUND --")

    def update_global_route(self, color=None, title="-- ROUTE FOUND --", show_secs=2):
        """
        Populate final_states() from env.best_route and show the path on the
        pygame canvas for `show_secs` seconds, then auto-dismiss.

        Parameters
        ----------
        color     : pygame colour tuple  (default: BLUE for QL, ORANGE for DQN)
        title     : sidebar heading
        show_secs : how long to display before auto-closing (default 2 s)
        """
        global a
        a = dict(self.best_route)

        if not a:
            return   # no route recorded yet — nothing to draw

        draw_color = color if color is not None else BLUE

        self._draw_base_grid()

        # Draw route dots
        for pos in a.values():
            tr, tc = pos
            pygame.draw.circle(
                self.screen, draw_color,
                (tc * CELL_SIZE + CELL_SIZE // 2,
                 tr * CELL_SIZE + CELL_SIZE // 2), 7)

        # Sidebar
        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        y = 15
        for line in [title,
                     "Shortest:",
                     f"  {self.shortest} steps",
                     "Longest:",
                     f"  {self.longest} steps",
                     "",
                     "Auto-closing..."]:
            self.screen.blit(self.font_sm.render(line, True, WHITE),
                             (GRID_PIXEL + 10, y))
            y += 22

        # Bottom bar
        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(
            self.font_sm.render(
                f"Best route shown. Auto-closing in {show_secs}s "
                f"(or press SPACE/ENTER to skip).",
                True, GREEN),
            (10, WINDOW_HEIGHT + 12))

        pygame.display.flip()

        # Wait up to show_secs seconds; user can press SPACE/ENTER to skip
        start = pygame.time.get_ticks()
        done  = False
        while not done:
            elapsed = (pygame.time.get_ticks() - start) / 1000
            if elapsed >= show_secs:
                done = True
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    done = True   # don't sys.exit — more phases follow
                if ev.type == pygame.KEYDOWN and ev.key in (
                        pygame.K_RETURN, pygame.K_SPACE):
                    done = True
            self.clock.tick(30)

    # ── NEW: show both routes simultaneously ──────────────────────────────── #
    def draw_dual_routes(self, ql_route, dqn_route):
        """
        Draw both agents' best paths on the same grid and block until closed.

        ql_route  — dict {step: (row,col)} from Q-Learning   → BLUE
        dqn_route — dict {step: (row,col)} from DQN          → ORANGE
        """
        self._draw_base_grid()

        # Q-Learning path — BLUE filled circles (radius 7)
        for pos in ql_route.values():
            tr, tc = pos
            pygame.draw.circle(
                self.screen, BLUE,
                (tc * CELL_SIZE + CELL_SIZE // 2,
                 tr * CELL_SIZE + CELL_SIZE // 2), 7)

        # DQN path — ORANGE filled circles (radius 5, drawn on top)
        for pos in dqn_route.values():
            tr, tc = pos
            pygame.draw.circle(
                self.screen, ORANGE,
                (tc * CELL_SIZE + CELL_SIZE // 2,
                 tr * CELL_SIZE + CELL_SIZE // 2), 5)

        # Sidebar legend
        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        y = 15
        for line in ["-- DUAL ROUTE --",
                     "",
                     "Q-Learn (E1):",
                     "  BLUE  ●",
                     f"  {len(ql_route)} steps",
                     "",
                     "DQN (E2):",
                     "  ORANGE ●",
                     f"  {len(dqn_route)} steps"]:
            color = BLUE   if "BLUE"   in line else \
                    ORANGE if "ORANGE" in line else WHITE
            self.screen.blit(
                self.font_sm.render(line, True, color),
                (GRID_PIXEL + 10, y))
            y += 22

        # Bottom bar
        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(
            self.font_sm.render(
                "Blue = Q-Learning  |  Orange = DQN  |  Close window to continue.",
                True, WHITE),
            (10, WINDOW_HEIGHT + 12))

        pygame.display.flip()

        # Block until user closes, then return (do NOT sys.exit — plots still need to run)
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            self.clock.tick(10)
        pygame.quit()

    # ── Internal draw helpers ─────────────────────────────────────────────── #
    def _draw_base_grid(self):
        """Render grid + obstacles onto the screen (no route dots)."""
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE,
                         pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
        for obs in self.placed_obstacles:
            self.screen.blit(obs['surface'], obs['rect'].topleft)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x, y = col * CELL_SIZE, row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                cell = self.grid[row][col]
                if cell == OBSTACLE:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill((255, 50, 50, 100))
                    self.screen.blit(s, (x, y))
                elif cell == START:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif cell == GOAL:
                    if self.flag_surface:
                        self.screen.blit(self.flag_surface, (x, y))
                    else:
                        pygame.draw.rect(self.screen, YELLOW, rect)
                pygame.draw.rect(self.screen, GREY, rect, 1)

    def _draw_single_route(self, route, color, title):
        """Draw one route then block until the window is closed."""
        self._draw_base_grid()
        for pos in route.values():
            tr, tc = pos
            pygame.draw.circle(
                self.screen, color,
                (tc * CELL_SIZE + CELL_SIZE // 2,
                 tr * CELL_SIZE + CELL_SIZE // 2), 6)

        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        y = 15
        for line in [title, "Shortest:",
                     f"  {self.shortest} steps", "Longest:",
                     f"  {self.longest} steps"]:
            self.screen.blit(self.font_sm.render(line, True, WHITE),
                             (GRID_PIXEL + 10, y))
            y += 22

        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(
            self.font_sm.render("Close window to continue.", True, GREEN),
            (10, WINDOW_HEIGHT + 12))
        pygame.display.flip()

        # Block until window is closed or Enter pressed; NEVER sys.exit (more phases follow)
        waiting = True
        while waiting:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    waiting = False          # just stop waiting, don't exit
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    waiting = False
            self.clock.tick(10)

    def _draw_placement(self):
        self.screen.fill(WHITE)
        for obs in self.placed_obstacles:
            self.screen.blit(obs['surface'], obs['rect'].topleft)
        for marker in self.placed_markers:
            r = marker['rect']
            if marker['type'] == 'start':
                pygame.draw.rect(self.screen, GREEN, r, border_radius=6)
                self.screen.blit(self.font_sm.render("S", True, BLACK),
                                 (r.centerx - 5, r.centery - 7))
            else:
                if self.flag_surface:
                    self.screen.blit(
                        pygame.transform.scale(self.flag_surface, (r.width, r.height)),
                        r.topleft)
                else:
                    pygame.draw.rect(self.screen, YELLOW, r, border_radius=6)
                    self.screen.blit(self.font_sm.render("G", True, BLACK),
                                     (r.centerx - 5, r.centery - 7))
        pygame.draw.rect(self.screen, GREY,
                         pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT), 2)
        self._draw_sidebar_placement()
        self._draw_bottom_bar()
        pygame.display.flip()

    def _draw_sidebar_placement(self):
        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        for item in self.sidebar_items:
            is_sel = ((item['type'] == 'start'    and self.selected == 'start') or
                      (item['type'] == 'goal'     and self.selected == 'goal')  or
                      (item['type'] == 'obstacle' and self.selected == item.get('idx')))
            bg = YELLOW if is_sel else DARK_GREY
            pygame.draw.rect(self.screen, bg,   item['rect'], border_radius=4)
            pygame.draw.rect(self.screen, GREY, item['rect'], 1, border_radius=4)
            if item['type'] == 'obstacle':
                self.screen.blit(self.sidebar_surfaces[item['idx']],
                                 item['rect'].topleft)
            else:
                self.screen.blit(
                    self.font_sm.render(item['label'], True,
                                        BLACK if is_sel else WHITE),
                    (item['rect'].x + 5, item['rect'].y + 8))

        self.screen.blit(
            self.font_sm.render(
                f"Obs: {self.obstacle_count}/{self.max_obstacles}", True, GREY),
            (GRID_PIXEL + 10, self.obs_label_y))

        has_both = (any(m['type'] == 'start' for m in self.placed_markers) and
                    any(m['type'] == 'goal'  for m in self.placed_markers))
        done_col = GREEN     if has_both else DARK_GREY
        text_col = BLACK     if has_both else GREY
        pygame.draw.rect(self.screen, done_col, self.done_button_rect, border_radius=6)
        pygame.draw.rect(self.screen, WHITE,    self.done_button_rect, 2, border_radius=6)
        self.screen.blit(self.font_lg.render("DONE", True, text_col),
                         (self.done_button_rect.x + 28, self.done_button_rect.y + 10))

    def _draw_bottom_bar(self):
        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(self.font_sm.render(self.status_msg, True, self.status_color),
                         (10, WINDOW_HEIGHT + 12))

    def _draw_training(self):
        self._draw_base_grid()
        for pos in self.episode_route.values():
            if pos == self.agent_pos:
                continue
            tr, tc = pos
            pygame.draw.circle(
                self.screen, LIGHT_BLUE,
                (tc * CELL_SIZE + CELL_SIZE // 2,
                 tr * CELL_SIZE + CELL_SIZE // 2), 4)
        if self.agent_pos:
            ar, ac = self.agent_pos
            if self.agent_surface:
                self.screen.blit(self.agent_surface, (ac * CELL_SIZE, ar * CELL_SIZE))
            else:
                pygame.draw.circle(
                    self.screen, RED,
                    (ac * CELL_SIZE + CELL_SIZE // 2,
                     ar * CELL_SIZE + CELL_SIZE // 2), 10)

        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        y = 15
        for line in ["-- TRAINING --", "Episode:",
                     f"  {self.current_episode}/{self.total_episodes}",
                     "Epsilon:", f"  {self.current_epsilon:.3f}",
                     "Shortest:", f"  {self.shortest} steps"]:
            self.screen.blit(self.font_sm.render(line, True, WHITE),
                             (GRID_PIXEL + 10, y))
            y += 22

        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(
            self.font_sm.render(
                f"Training... Episode {self.current_episode}/{self.total_episodes}",
                True, WHITE),
            (10, WINDOW_HEIGHT + 12))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        pygame.display.flip()


def final_states():
    """Return the global best-route dict (used by agent_brain.print_q_table)."""
    return a
