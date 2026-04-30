import numpy as np
import pygame
import sys
from collections import deque

# Grid settings
GRID_SIZE      = 15
CELL_SIZE      = 40
GRID_PIXEL     = GRID_SIZE * CELL_SIZE   # 600px
SIDEBAR_WIDTH  = 160
WINDOW_WIDTH   = GRID_PIXEL + SIDEBAR_WIDTH
WINDOW_HEIGHT  = GRID_PIXEL
BOTTOM_BAR     = 40
FULL_HEIGHT    = WINDOW_HEIGHT + BOTTOM_BAR

# Obstacle image size on the free canvas
OBSTACLE_SIZE  = 40

# Coverage threshold — if obstacle covers >= 15% of a cell, it's an obstacle
COVERAGE_THRESHOLD = 0.15

# Colors
WHITE      = (255, 255, 255)
GREY       = (180, 180, 180)
DARK_GREY  = (60, 60, 60)
BLACK      = (0, 0, 0)
GREEN      = (50, 200, 50)
RED        = (200, 50, 50)
YELLOW     = (255, 220, 0)
BLUE       = (50, 100, 220)
LIGHT_BLUE = (173, 216, 230)
SIDEBAR_BG = (30, 30, 40)
ORANGE     = (255, 140, 0)

# Cell types
EMPTY    = 0
OBSTACLE = 1
START    = 2
GOAL     = 3

# Actions
UP    = 0
DOWN  = 1
RIGHT = 2
LEFT  = 3

# Global final route dict
a = {}

OBSTACLE_IMAGE_FILES = [
    "road_closed1.png",
    "road_closed2.png",
    "road_closed3.png",
    "tree1.png",
    "tree2.png",
    "building1.png",
    "building2.png",
    "traffic_lights.png",
    "pedestrian.png",
    "shop.png",
    "bank1.png",
    "bank2.png",
]


class Environment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, FULL_HEIGHT))
        pygame.display.set_caption("DQN Path Planning")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)

        # Grid — computed after DONE
        self.grid         = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.cell_image_map = {}  # {(row,col): (coverage, surface)}

        # Start / goal in grid coords
        self.start_pos = None
        self.goal_pos  = None

        # Agent position during training
        self.agent_pos = None

        # Route tracking
        self.episode_route = {}
        self.episode_step  = 0
        self.best_route    = {}
        self.first_goal    = True
        self.shortest      = 0
        self.longest       = 0

        # Free canvas placed obstacles
        # Each: {'img_idx': int, 'rect': pygame.Rect, 'surface': Surface}
        self.placed_obstacles = []
        self.obstacle_count   = 0
        self.max_obstacles    = 20

        # START / GOAL markers on canvas
        # Each: {'type': 'start'/'goal', 'rect': pygame.Rect}
        self.placed_markers = []

        # Sidebar selection: 'start', 'goal', or int
        self.selected = None

        # Drag state
        self.dragging    = None
        self.drag_offset = (0, 0)

        # Status bar
        self.status_msg   = "Select item from sidebar. Click canvas to place. Drag to reposition."
        self.status_color = WHITE

        # Training overlay info
        self.current_episode = 0
        self.total_episodes  = 0
        self.current_epsilon = 1.0

        self._load_images()
        self._build_sidebar()

    # ------------------------------------------------------------------ #
    #  Image loading                                                       #
    # ------------------------------------------------------------------ #
    def _load_images(self):
        self.obstacle_canvas_surfaces = []
        self.obstacle_cell_surfaces   = []
        self.sidebar_surfaces         = []
        self.agent_surface            = None
        self.flag_surface             = None

        thumb_size = (60, 40)

        for fname in OBSTACLE_IMAGE_FILES:
            try:
                img = pygame.image.load(f"images/{fname}").convert_alpha()
                self.obstacle_canvas_surfaces.append(
                    pygame.transform.scale(img, (OBSTACLE_SIZE, OBSTACLE_SIZE)))
                self.obstacle_cell_surfaces.append(
                    pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE)))
                self.sidebar_surfaces.append(
                    pygame.transform.scale(img, thumb_size))
            except Exception:
                for lst, size in [
                    (self.obstacle_canvas_surfaces, (OBSTACLE_SIZE, OBSTACLE_SIZE)),
                    (self.obstacle_cell_surfaces,   (CELL_SIZE, CELL_SIZE)),
                    (self.sidebar_surfaces,          thumb_size),
                ]:
                    s = pygame.Surface(size, pygame.SRCALPHA)
                    s.fill((0, 191, 255, 200))
                    lst.append(s)

        try:
            img = pygame.image.load("images/agent1.png").convert_alpha()
            self.agent_surface = pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except Exception:
            self.agent_surface = None

        try:
            img = pygame.image.load("images/flag.png").convert_alpha()
            self.flag_surface = pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except Exception:
            self.flag_surface = None

    # ------------------------------------------------------------------ #
    #  Sidebar layout                                                      #
    # ------------------------------------------------------------------ #
    def _build_sidebar(self):
        self.sidebar_items = []
        x = GRID_PIXEL + 10
        y = 10

        r = pygame.Rect(x, y, SIDEBAR_WIDTH - 20, 30)
        self.sidebar_items.append({'label': 'START', 'rect': r, 'type': 'start'})
        y += 38

        r = pygame.Rect(x, y, SIDEBAR_WIDTH - 20, 30)
        self.sidebar_items.append({'label': 'GOAL', 'rect': r, 'type': 'goal'})
        y += 38

        self.obs_label_y = y
        y += 18

        col_w = 60
        col_gap = 20
        row_h = 40
        row_gap = 10

        start_y = y
        for i in range(len(OBSTACLE_IMAGE_FILES)):
            col = i % 2
            row = i // 2
            item_x = x + col * (col_w + col_gap)
            item_y = start_y + row * (row_h + row_gap)
            r = pygame.Rect(item_x, item_y, col_w, row_h)
            self.sidebar_items.append({'label': '', 'rect': r, 'type': 'obstacle', 'idx': i})

        self.done_button_rect = pygame.Rect(x, WINDOW_HEIGHT - 60, SIDEBAR_WIDTH - 20, 40)

    # ------------------------------------------------------------------ #
    #  Phase 1 — Free canvas placement loop                               #
    # ------------------------------------------------------------------ #
    def run_placement_phase(self):
        running = True
        while running:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    result = self._handle_mousedown(mx, my)
                    if result == 'done':
                        running = False

                if event.type == pygame.MOUSEMOTION and self.dragging is not None:
                    ox, oy = self.drag_offset
                    self.dragging['rect'].x = event.pos[0] - ox
                    self.dragging['rect'].y = event.pos[1] - oy
                    self.dragging['rect'].clamp_ip(
                        pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging = None

            self._draw_placement()

    def _handle_mousedown(self, mx, my):
        # Done button
        if self.done_button_rect.collidepoint(mx, my):
            if self._handle_done():
                return 'done'
            return

        # Sidebar click
        for item in self.sidebar_items:
            if item['rect'].collidepoint(mx, my):
                self.selected = 'start' if item['type'] == 'start' else \
                                'goal'  if item['type'] == 'goal'  else item['idx']
                return

        # Canvas click
        if mx < GRID_PIXEL and my < WINDOW_HEIGHT:
            # Try to pick up existing item for dragging
            for item in reversed(self.placed_obstacles):
                if item['rect'].collidepoint(mx, my):
                    self.dragging    = item
                    self.drag_offset = (mx - item['rect'].x, my - item['rect'].y)
                    return
            for item in reversed(self.placed_markers):
                if item['rect'].collidepoint(mx, my):
                    self.dragging    = item
                    self.drag_offset = (mx - item['rect'].x, my - item['rect'].y)
                    return

            # Place new item
            if self.selected == 'start':
                self.placed_markers = [m for m in self.placed_markers if m['type'] != 'start']
                r = pygame.Rect(mx - CELL_SIZE // 2, my - CELL_SIZE // 2, CELL_SIZE, CELL_SIZE)
                r.clamp_ip(pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                self.placed_markers.append({'type': 'start', 'rect': r})
                self.status_msg   = "START placed. Drag to reposition."
                self.status_color = GREEN

            elif self.selected == 'goal':
                self.placed_markers = [m for m in self.placed_markers if m['type'] != 'goal']
                r = pygame.Rect(mx - CELL_SIZE // 2, my - CELL_SIZE // 2, CELL_SIZE, CELL_SIZE)
                r.clamp_ip(pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                self.placed_markers.append({'type': 'goal', 'rect': r})
                self.status_msg   = "GOAL placed. Drag to reposition."
                self.status_color = GREEN

            elif isinstance(self.selected, int):
                if self.obstacle_count >= self.max_obstacles:
                    self.status_msg   = f"Max {self.max_obstacles} obstacles reached!"
                    self.status_color = RED
                    return
                idx  = self.selected
                surf = self.obstacle_canvas_surfaces[idx]
                r    = pygame.Rect(mx - OBSTACLE_SIZE // 2,
                                   my - OBSTACLE_SIZE // 2,
                                   OBSTACLE_SIZE, OBSTACLE_SIZE)
                r.clamp_ip(pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))
                self.placed_obstacles.append({'img_idx': idx, 'rect': r, 'surface': surf})
                self.obstacle_count += 1
                self.status_msg   = f"Obstacle placed ({self.obstacle_count}/{self.max_obstacles}). Drag to reposition."
                self.status_color = WHITE

            else:
                self.status_msg   = "Select an item from the sidebar first."
                self.status_color = ORANGE

    def _handle_done(self):
        start_markers = [m for m in self.placed_markers if m['type'] == 'start']
        goal_markers  = [m for m in self.placed_markers if m['type'] == 'goal']

        if not start_markers:
            self.status_msg   = "Please place START first."
            self.status_color = RED
            return False
        if not goal_markers:
            self.status_msg   = "Please place GOAL first."
            self.status_color = RED
            return False

        self._build_grid_from_canvas(start_markers[0], goal_markers[0])

        if self.start_pos is None:
            self.status_msg   = "START not on canvas. Move it inside the grid area."
            self.status_color = RED
            return False
        if self.goal_pos is None:
            self.status_msg   = "GOAL not on canvas. Move it inside the grid area."
            self.status_color = RED
            return False
        if self.start_pos == self.goal_pos:
            self.status_msg   = "START and GOAL are on the same cell. Move one."
            self.status_color = RED
            return False

        if not self._bfs_check():
            self.status_msg   = "No valid path! Remove obstacles or reposition START/GOAL."
            self.status_color = RED
            return False

        self.status_msg   = "Path found! Starting DQN training..."
        self.status_color = GREEN
        self._draw_placement()
        pygame.display.flip()
        pygame.time.wait(800)
        return True

    # ------------------------------------------------------------------ #
    #  Grid builder — coverage-based                                       #
    # ------------------------------------------------------------------ #
    def _build_grid_from_canvas(self, start_marker, goal_marker):
        self.grid         = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.cell_image_map = {}
        self.start_pos    = None
        self.goal_pos     = None
        cell_area         = CELL_SIZE * CELL_SIZE

        # Mark obstacle cells by coverage
        for obs in self.placed_obstacles:
            obs_rect = obs['rect']
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    cell_rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE,
                                            CELL_SIZE, CELL_SIZE)
                    inter = cell_rect.clip(obs_rect)
                    if inter.width > 0 and inter.height > 0:
                        coverage = (inter.width * inter.height) / cell_area
                        if coverage >= COVERAGE_THRESHOLD:
                            self.grid[row][col] = OBSTACLE

        # START — center point snaps to cell
        scx = start_marker['rect'].centerx
        scy = start_marker['rect'].centery
        if 0 <= scx < GRID_PIXEL and 0 <= scy < WINDOW_HEIGHT:
            sr, sc         = scy // CELL_SIZE, scx // CELL_SIZE
            self.grid[sr][sc] = START
            self.start_pos    = (sr, sc)

        # GOAL — center point snaps to cell
        gcx = goal_marker['rect'].centerx
        gcy = goal_marker['rect'].centery
        if 0 <= gcx < GRID_PIXEL and 0 <= gcy < WINDOW_HEIGHT:
            gr, gc         = gcy // CELL_SIZE, gcx // CELL_SIZE
            self.grid[gr][gc] = GOAL
            self.goal_pos     = (gr, gc)

    # ------------------------------------------------------------------ #
    #  BFS                                                                 #
    # ------------------------------------------------------------------ #
    def _bfs_check(self):
        queue   = deque([self.start_pos])
        visited = set([self.start_pos])
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal_pos:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    if (nr, nc) not in visited and self.grid[nr][nc] != OBSTACLE:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    # ------------------------------------------------------------------ #
    #  Training interface                                                  #
    # ------------------------------------------------------------------ #
    def reset(self):
        self.agent_pos     = self.start_pos
        self.episode_route = {0: self.start_pos}
        self.episode_step  = 1
        return self.start_pos

    def step(self, action):
        r, c = self.agent_pos
        nr, nc = r, c

        if action == UP:
            nr = r - 1
        elif action == DOWN:
            nr = r + 1
        elif action == RIGHT:
            nc = c + 1
        elif action == LEFT:
            nc = c - 1

        # Out of bounds — stay, penalise, not done
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
                    self.longest = len(self.episode_route)
            return next_pos, 10, True

        # Free cell
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def final(self):
        global a
        a = dict(self.best_route)
        print(f"Shortest route: {self.shortest} steps")
        print(f"Longest route:  {self.longest} steps")
        self._draw_final_route()

    # ------------------------------------------------------------------ #
    #  Drawing — Placement                                                 #
    # ------------------------------------------------------------------ #
    def _draw_placement(self):
        self.screen.fill(WHITE)

        # Blank white canvas
        pygame.draw.rect(self.screen, WHITE,
                         pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))

        # Placed obstacles
        for obs in self.placed_obstacles:
            self.screen.blit(obs['surface'], obs['rect'].topleft)

        # START / GOAL markers
        for marker in self.placed_markers:
            r = marker['rect']
            if marker['type'] == 'start':
                pygame.draw.rect(self.screen, GREEN, r, border_radius=6)
                lbl = self.font_sm.render("S", True, BLACK)
                self.screen.blit(lbl, (r.centerx - 5, r.centery - 7))
            else:
                if self.flag_surface:
                    scaled = pygame.transform.scale(
                        self.flag_surface, (r.width, r.height))
                    self.screen.blit(scaled, r.topleft)
                else:
                    pygame.draw.rect(self.screen, YELLOW, r, border_radius=6)
                    lbl = self.font_sm.render("G", True, BLACK)
                    self.screen.blit(lbl, (r.centerx - 5, r.centery - 7))

        # Canvas border
        pygame.draw.rect(self.screen, GREY,
                         pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT), 2)

        self._draw_sidebar_placement()
        self._draw_bottom_bar()
        pygame.display.flip()

    def _draw_sidebar_placement(self):
        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))

        for item in self.sidebar_items:
            is_sel = (
                (item['type'] == 'start'    and self.selected == 'start') or
                (item['type'] == 'goal'     and self.selected == 'goal')  or
                (item['type'] == 'obstacle' and self.selected == item.get('idx'))
            )
            bg = YELLOW if is_sel else DARK_GREY
            pygame.draw.rect(self.screen, bg,   item['rect'], border_radius=4)
            pygame.draw.rect(self.screen, GREY, item['rect'], 1, border_radius=4)

            if item['type'] == 'obstacle':
                self.screen.blit(self.sidebar_surfaces[item['idx']], item['rect'].topleft)
            else:
                col = BLACK if is_sel else WHITE
                lbl = self.font_sm.render(item['label'], True, col)
                self.screen.blit(lbl, (item['rect'].x + 5, item['rect'].y + 8))

        count_lbl = self.font_sm.render(
            f"Obs: {self.obstacle_count}/{self.max_obstacles}", True, GREY)
        self.screen.blit(count_lbl, (GRID_PIXEL + 10, self.obs_label_y))

        has_start = any(m['type'] == 'start' for m in self.placed_markers)
        has_goal  = any(m['type'] == 'goal'  for m in self.placed_markers)
        has_both  = has_start and has_goal
        done_col  = GREEN     if has_both else DARK_GREY
        text_col  = BLACK     if has_both else GREY
        pygame.draw.rect(self.screen, done_col, self.done_button_rect, border_radius=6)
        pygame.draw.rect(self.screen, WHITE,    self.done_button_rect, 2, border_radius=6)
        lbl = self.font_lg.render("DONE", True, text_col)
        self.screen.blit(lbl, (self.done_button_rect.x + 28, self.done_button_rect.y + 10))

    def _draw_bottom_bar(self):
        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        msg = self.font_sm.render(self.status_msg, True, self.status_color)
        self.screen.blit(msg, (10, WINDOW_HEIGHT + 12))

    # ------------------------------------------------------------------ #
    #  Drawing — Training                                                  #
    # ------------------------------------------------------------------ #
    def _draw_training(self):
        self.screen.fill(BLACK)

        # 1. Blank white canvas
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))

        # 2. Placed obstacles
        for obs in self.placed_obstacles:
            self.screen.blit(obs['surface'], obs['rect'].topleft)

        # 3. Grid overlay
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x    = col * CELL_SIZE
                y    = row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                cell = self.grid[row][col]

                if cell == OBSTACLE:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill((255, 50, 50, 100)) # Semi-transparent red overlay
                    self.screen.blit(s, (x, y))
                elif cell == START:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif cell == GOAL:
                    if self.flag_surface:
                        self.screen.blit(self.flag_surface, (x, y))
                    else:
                        pygame.draw.rect(self.screen, YELLOW, rect)

                pygame.draw.rect(self.screen, GREY, rect, 1)

        # Trail
        for pos in self.episode_route.values():
            if pos == self.agent_pos:
                continue
            tr, tc = pos
            pygame.draw.circle(self.screen, LIGHT_BLUE,
                               (tc * CELL_SIZE + CELL_SIZE // 2,
                                tr * CELL_SIZE + CELL_SIZE // 2), 4)

        # Agent
        if self.agent_pos:
            ar, ac = self.agent_pos
            ax, ay = ac * CELL_SIZE, ar * CELL_SIZE
            if self.agent_surface:
                self.screen.blit(self.agent_surface, (ax, ay))
            else:
                pygame.draw.circle(self.screen, RED,
                                   (ax + CELL_SIZE // 2, ay + CELL_SIZE // 2), 10)

        # Sidebar
        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        y = 15
        for line in ["-- TRAINING --",
                     "Episode:",
                     f"  {self.current_episode}/{self.total_episodes}",
                     "Epsilon:",
                     f"  {self.current_epsilon:.3f}",
                     "Shortest:",
                     f"  {self.shortest} steps"]:
            self.screen.blit(self.font_sm.render(line, True, WHITE), (GRID_PIXEL + 10, y))
            y += 22

        # Bottom bar
        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(
            self.font_sm.render(
                f"Training... Episode {self.current_episode}/{self.total_episodes}",
                True, WHITE),
            (10, WINDOW_HEIGHT + 12))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()

    # ------------------------------------------------------------------ #
    #  Drawing — Final route                                               #
    # ------------------------------------------------------------------ #
    def _draw_final_route(self):
        self.screen.fill(BLACK)

        # 1. Blank white canvas
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, 0, GRID_PIXEL, WINDOW_HEIGHT))

        # 2. Placed obstacles
        for obs in self.placed_obstacles:
            self.screen.blit(obs['surface'], obs['rect'].topleft)

        # 3. Grid overlay
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x    = col * CELL_SIZE
                y    = row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                cell = self.grid[row][col]

                if cell == OBSTACLE:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill((255, 50, 50, 100)) # Semi-transparent red overlay
                    self.screen.blit(s, (x, y))
                elif cell == START:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif cell == GOAL:
                    if self.flag_surface:
                        self.screen.blit(self.flag_surface, (x, y))
                    else:
                        pygame.draw.rect(self.screen, YELLOW, rect)

                pygame.draw.rect(self.screen, GREY, rect, 1)

        # Best route
        for pos in self.best_route.values():
            tr, tc = pos
            pygame.draw.circle(self.screen, BLUE,
                               (tc * CELL_SIZE + CELL_SIZE // 2,
                                tr * CELL_SIZE + CELL_SIZE // 2), 6)

        # Sidebar
        pygame.draw.rect(self.screen, SIDEBAR_BG,
                         pygame.Rect(GRID_PIXEL, 0, SIDEBAR_WIDTH, FULL_HEIGHT))
        y = 15
        for line in ["-- COMPLETE --",
                     "Shortest:",
                     f"  {self.shortest} steps",
                     "Longest:",
                     f"  {self.longest} steps"]:
            self.screen.blit(self.font_sm.render(line, True, WHITE), (GRID_PIXEL + 10, y))
            y += 22

        # Bottom bar
        pygame.draw.rect(self.screen, DARK_GREY,
                         pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR))
        self.screen.blit(
            self.font_sm.render("Training complete! Close window to exit.", True, GREEN),
            (10, WINDOW_HEIGHT + 12))

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.clock.tick(10)


def final_states():
    return a