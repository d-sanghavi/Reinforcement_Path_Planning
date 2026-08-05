\# Reinforcement Path Planning — v2 Engineering Specification \& Build Prompt

\### (CAD-Ingested, Symbol-Aware, DDQN-Driven Indoor Robot Path Planner)



> \*\*How to use this document:\*\* This is written to be pasted directly into Claude Code CLI (or Claude on Antigravity) as the governing project brief. Each numbered stage in Section 5 can also be pasted individually as a standalone task prompt when you want to build the pipeline incrementally. Nothing in this spec assumes any paid/external model API — every component is designed to run on local, open-source, self-trained models within your own compute/token budget.



\---



\## 0. Context: What Already Exists (v1)



Repo: `https://github.com/d-sanghavi/Reinforcement\_Path\_Planning`



\- \*\*Type:\*\* Q-Learning based path planner on a synthetic, hand-authored 2D grid ("urban world" — buildings, trees, shops, traffic signals, pedestrian crossings, road barriers).

\- \*\*Core files:\*\* `agent\_brain.py` (Q-learning agent), `env.py` (grid-world environment), `run\_agent.py` (training/execution loop).

\- \*\*Algorithm:\*\* Tabular Q-Learning, update rule `Q(s,a) ← Q(s,a) + α\[r + γ·max Q(s',a') − Q(s,a)]`.

\- \*\*State space:\*\* discrete `(x, y)` grid cell.

\- \*\*Action space:\*\* 4 discrete moves — UP / DOWN / LEFT / RIGHT.

\- \*\*Reward shaping:\*\* small negative step cost, large negative for obstacle collision, positive terminal reward at goal, penalty for invalid moves.

\- \*\*Output today:\*\* static grid visualization + training performance plots (reward curve, `Figure\_1/2/3.png`), no real-world file ingestion, no real-world coordinate output, no PDF export.

\- \*\*Gap this version closes:\*\* v1 has no real-world input source (grids are hand-built) and no deployment-facing output. v2 must ingest a real architectural drawing, understand it, and hand back an operator-usable artifact (annotated PDF + coordinate route + metrics).



\---



\## 1. V2 Goal (One Paragraph)



Given a user-supplied CAD/DXF floor plan of a house, the system must (a) understand the drawing semantically using a custom-trained ML/DRL symbol-recognition model (no external AI API calls), (b) rasterize it into a binary traversability grid (free-space vs. obstacle), (c) let the user visually confirm the grid and pick a start/goal cell, (d) run a Double DQN (DDQN) agent to compute the shortest collision-free path, and (e) deliver the result as real-world-scaled route coordinates, a black-and-white PDF of the floor plan with the path overlaid in a distinct color, and the same training/performance metrics dashboard used in v1 — extended with the new CAD-derived metrics.



\---



\## 2. Hard Constraints (Non-Negotiable)



1\. \*\*No external model API keys, ever\*\* — no OpenAI/Anthropic/Google Vision/AWS Rekognition calls for the vision or symbol-recognition step. Everything must be a locally trained/fine-tuned model (PyTorch/TensorFlow) or classical CV (OpenCV, `ezdxf`, `shapely`).

2\. \*\*Token/credit-budget conscious\*\* — prefer classical CV + small custom CNN/GNN over large foundation models; only use an LLM (e.g., local Claude session) for \*code generation/orchestration\*, never at inference time inside the running pipeline.

3\. \*\*Offline-capable\*\* — the trained symbol-recognition model and DDQN agent must run without any internet call once deployed.

4\. \*\*Deterministic, reproducible output\*\* — same input file + same start/end → same path and same PDF, every run (seed the DDQN inference / use greedy policy at inference time, not exploration).

5\. \*\*All CAD→DXF conversion must use a local converter\*\* (ODA File Converter, LibreCAD/`libdxfrw`, or FreeCAD's Python API) — not a SaaS conversion API.



\---



\## 3. High-Level Pipeline (8 Stages)



```

\[.cad or .dxf file]

&#x20;       │

&#x20;       ▼

Stage 1 — Format Normalization (CAD → DXF if needed)

&#x20;       │

&#x20;       ▼

Stage 2 — Symbol Knowledge Base (pre-trained once, reused every run)

&#x20;       │

&#x20;       ▼

Stage 3 — DXF Parsing + Symbol Recognition (ML/DRL model applies the KB to THIS file)

&#x20;       │

&#x20;       ▼

Stage 4 — Grid Construction (binary: traversable vs obstacle, 2-color)

&#x20;       │

&#x20;       ▼

Stage 5 — Grid Export \& Human-in-the-loop Review (user sees the grid)

&#x20;       │

&#x20;       ▼

Stage 6 — Start / Goal Point Selection (user picks cells on the grid)

&#x20;       │

&#x20;       ▼

Stage 7 — DDQN Path Planning (shortest path search over the confirmed grid)

&#x20;       │

&#x20;       ▼

Stage 8 — Output Generation

&#x20;  ├─ Real-world scaled coordinate list (per DXF drawing scale/units)

&#x20;  ├─ B/W PDF of the floor plan with path overlaid in a distinct color

&#x20;  └─ Metrics dashboard (v1 metrics + new CAD-specific metrics)

```



\---



\## 4. Detailed Stage Specification



\### Stage 1 — Input Ingestion \& Format Normalization



\- \*\*Accepted inputs:\*\* `.dxf` (native path) or `.dwg`/other AutoCAD `.cad` variants (conversion path).

\- \*\*Logic:\*\*

&#x20; 1. Detect file extension/magic bytes.

&#x20; 2. If `.dxf` → pass directly to Stage 3.

&#x20; 3. If `.dwg`/proprietary CAD → convert locally to `.dxf` using \*\*ODA File Converter CLI\*\* (free, local binary, no API key) or \*\*LibreCAD\*\*/`libredwg` as a fallback. Wrap the converter call in a subprocess with explicit version-locking (target DXF version: R2013/ASCII for `ezdxf` compatibility).

&#x20; 4. Validate the resulting DXF: confirm it opens cleanly with `ezdxf.readfile()`, log entity counts (LINE, ARC, LWPOLYLINE, INSERT/block-refs, TEXT, DIMENSION) as a sanity report before proceeding.

&#x20; 5. Extract and store the drawing's \*\*units and scale\*\* (`$INSUNITS`, `$MEASUREMENT` header vars, or a user-confirmed scale bar/annotation) — this value is required later for real-world coordinate conversion in Stage 8.

\- \*\*Failure handling:\*\* if conversion or parsing fails, surface the exact `ezdxf` / ODA error to the user rather than silently defaulting to a blank grid.



\### Stage 2 — Symbol Knowledge Base (built once, reused across all runs)



This is the "world knowledge" the custom ML/DRL model needs before it can read \*any\* floor plan.



\- \*\*Symbol classes to encode\*\* (non-exhaustive — extend as needed):

&#x20; - Wall (single-line, double-line/parallel-line wall representation)

&#x20; - Door — swing door (the 90° quarter-circle arc + leaf line), sliding door, double door, folding door

&#x20; - Window (parallel short perpendicular ticks on a wall segment)

&#x20; - Opening/archway (no door leaf, just a break in the wall line)

&#x20; - Stairs (parallel step lines + directional arrow)

&#x20; - Columns/pillars (filled or hatched squares/circles)

&#x20; - Fixed furniture commonly on floor plans: table, chair, sofa, bed, kitchen counter, sink, toilet, bathtub (relevant because these are \*obstacles\*, not walls)

&#x20; - Dimension lines / text annotations / title block (must be recognized and \*excluded\* from the obstacle/free-space grid — they are metadata, not geometry)

&#x20; - Scale bar / north arrow (used to calibrate Stage 8 coordinate conversion)

\- \*\*Representation of the KB:\*\* a labeled vector/raster template library (per symbol: DXF block geometry signature — line count, arc angles, bounding-box aspect ratio, relative-position rules such as "door arc always anchors to a wall line endpoint") plus, ideally, block \*names\* from common CAD libraries (AutoCAD Architecture, ARCAT, BIM object libraries) since floor plans frequently use named `INSERT` blocks for doors/windows/furniture — matching by block name is a free, deterministic shortcut before falling back to geometric shape classification.

\- \*\*Model architecture (custom, local, no external API):\*\*

&#x20; - A \*\*CNN or Graph Neural Network (GNN)\*\* classifier trained on:

&#x20;   - Rendered raster crops of each DXF entity/block (rasterize each entity/group of entities to a small fixed-size image, e.g. 64×64), labeled by symbol class → CNN classifier (ResNet-lite or a small custom CNN), OR

&#x20;   - A graph representation of DXF entities (nodes = entities, edges = spatial proximity/connectivity), labeled by symbol class → GNN classifier. GNN is preferred because CAD symbols are naturally graphs of connected primitives (arcs+lines for a door, parallel lines for a wall) rather than photographic images.

&#x20; - Output: per-entity/per-block \*\*symbol label + confidence score\*\*.

\- \*\*Training data sources (all offline/local once downloaded, not runtime APIs):\*\*

&#x20; - Public floor-plan/CAD symbol datasets: CubiCasa5k, ROBIN, RPLAN, FloorplanCAD (SFPD)/ SESYD — these already ship in vector/SVG/DXF-like or annotated raster form and are commonly used for exactly this symbol-recognition task.

&#x20; - Programmatically \*\*generate synthetic DXF training data\*\*: script random rooms with random combinations of the symbol library above (using `ezdxf` to write synthetic DXFs with ground-truth labels attached as XDATA) — this sidesteps any licensing/API concerns and lets you scale the dataset for free.

\- \*\*This model is trained once (offline, ahead of time) and only loaded for inference at pipeline run-time.\*\* No training happens per-user-upload.



\### Stage 3 — DXF Parsing + Symbol Recognition on the User's File



\- Parse the target DXF with `ezdxf`: enumerate `modelspace()` entities, resolve block `INSERT`s recursively, flatten polylines/arcs/circles to primitive geometry.

\- Group nearby/connected primitives into candidate symbol instances (simple proximity clustering, e.g. DBSCAN on entity midpoints/bounding boxes, or connected-component analysis on a rasterized version of the drawing).

\- Run each candidate group through the Stage 2 classifier → obtain per-region label: `wall | door | window | opening | stairs | column | furniture-obstacle | annotation/non-geometric | unknown`.

\- Reject/flag low-confidence classifications for manual user correction in the Stage 5 review UI rather than silently guessing.

\- Persist the labeled entity list (entity handle → class → geometry) as an intermediate artifact (e.g. `parsed\_symbols.json`) — this is what Stage 4 consumes, and it should be cached so re-running Stage 4+ doesn't require re-running the classifier.



\### Stage 4 — Grid Construction



\- Choose a grid resolution (cell size) derived from the DXF's real-world scale — e.g., 1 cell = 10 cm — configurable, with a sane default and a user override.

\- Rasterize the classified geometry onto this grid:

&#x20; - \*\*Obstacle cells (color/value B, e.g. black/1):\*\* wall, column, furniture-obstacle, closed/blocked regions.

&#x20; - \*\*Free/traversable cells (color/value A, e.g. white/0):\*\* open floor area, door openings, archways, hallway/passage space, window area is \*not\* traversable (it's on a wall) unless it's actually a floor-level opening.

&#x20; - Door swing arcs should mark the swept area as traversable (a door, unlike a wall, is a \*path\*) — this is the key symbolic reasoning step: recognizing the 90° arc means "this is where the wall is \*not\* a barrier."

\- Output a 2D array/matrix (e.g. NumPy `uint8`) representing the grid, plus a rendered preview image using exactly two colors for unambiguous visual review.

\- Store a mapping from grid-cell `(row, col)` back to real-world `(x, y)` DXF coordinates (needed for Stage 8 output scaling).



\### Stage 5 — Grid Export \& User Review



\- Render the binary grid as an image/overlay (matching "the grid the user is able to see, as in all the recent developments" — i.e., keep visual continuity with the v1 grid-visualization style).

\- Present it to the user (in whatever UI layer you build — CLI-served image, simple web viewer, or notebook display) so they can:

&#x20; - Confirm the free-space/obstacle classification looks correct.

&#x20; - Flag/correct any misclassified regions (feeds back into Stage 3's low-confidence queue for manual correction, and optionally into future re-training).

\- Do not proceed to Stage 6 without user confirmation of the grid.



\### Stage 6 — Start / End Point Selection



\- Let the user click/select a start cell and an end cell on the confirmed grid (must both land on traversable/free cells — validate and reject clicks on obstacle cells with a clear message).

\- Store both as grid coordinates `(row, col)` and their corresponding real-world DXF coordinates.



\### Stage 7 — DDQN Path Planning



\- \*\*Why DDQN over v1's tabular Q-Learning:\*\* the real-floor-plan grid is far larger and sparser than the v1 synthetic world, and DDQN's function approximation (vs. a full Q-table) generalizes better and avoids Q-value overestimation via the dual online/target network design — directly reusing the existing repo's RL foundation (`agent\_brain.py`, `env.py`) but upgrading the function approximator.

\- \*\*State representation:\*\* local/global grid observation around the agent (e.g., an n×n window centered on current position, or full grid + current position channel) — a CNN-based Q-network is a natural fit given the 2D grid nature of the environment (consistent with the existing `env.py` grid-world abstraction).

\- \*\*Action space:\*\* keep the existing 4-directional action space (UP/DOWN/LEFT/RIGHT); consider 8-directional if diagonal movement is physically valid for the target robot.

\- \*\*Network architecture:\*\* small CNN (2–3 conv layers) → FC layers → Q-values per action, duplicated as \*\*online network\*\* and \*\*target network\*\* (target network weights synced every N steps — this "double" aspect decouples action selection from evaluation to reduce overestimation bias vs. vanilla DQN).

\- \*\*Reward shaping (extend v1's scheme):\*\*

&#x20; - Small negative step cost (encourage shortest path).

&#x20; - Large negative for obstacle collision / attempted move into a black cell.

&#x20; - Positive terminal reward on reaching the goal cell.

&#x20; - Optional: distance-to-goal shaping term (potential-based reward shaping) to speed convergence on large grids.

\- \*\*Training:\*\* train (or fine-tune a pre-trained generalist DDQN) on a variety of synthetic and real floor-plan grids so the policy generalizes across houses rather than overfitting to one drawing; use experience replay buffer + epsilon-greedy during training, \*\*fully greedy (epsilon=0) at inference time\*\* for the deterministic-output constraint in Section 2.

\- \*\*Inference:\*\* given the confirmed grid + start/goal from Stages 4–6, roll out the trained policy (or run a lightweight A\*/Dijkstra sanity-check in parallel to validate DDQN's output isn't pathologically wrong — recommended for a robotics-safety-relevant deliverable) to produce the final cell-by-cell path.



\### Stage 8 — Output Generation



1\. \*\*Route coordinates:\*\* convert the DDQN's grid-cell path back to real-world coordinates using the DXF scale/units captured in Stage 1 and the grid↔DXF-coordinate mapping from Stage 4. Output as a simple ordered list, e.g. `\[(x0,y0), (x1,y1), ...]` in the drawing's real-world units (meters/mm/feet per the DXF header), plus the same list re-expressed in grid indices for debugging.

2\. \*\*Annotated PDF:\*\*

&#x20;  - Render the (converted, if needed) DXF as a clean \*\*black-and-white PDF\*\* of the floor plan (walls/fixed geometry in black on white, using `ezdxf`'s drawing add-on / `matplotlib` backend, or ODA/LibreCAD's own PDF export as the base render).

&#x20;  - Overlay the computed path as a polyline in a single clearly distinct color (e.g., red or blue) on top of the B/W base — using the real-world coordinates from step 1 so the overlay aligns pixel-accurately with the underlying drawing.

&#x20;  - Export as a single multi-layer-flattened PDF file.

3\. \*\*Metrics dashboard\*\* — carry over everything currently produced in the repo's `Results/` outputs, plus new CAD-run metrics:

&#x20;  - \*Carried over from v1:\* training reward curve, episode count to convergence, static path visualization (`staticQLearning.png`-style).

&#x20;  - \*New for v2:\*

&#x20;    - Path length in real-world units (meters/feet) and in grid cells.

&#x20;    - Planning/inference time (ms) for the DDQN rollout on this specific floor plan.

&#x20;    - Symbol-recognition confidence summary (how many entities were high-confidence vs. flagged for manual review in Stage 3/5).

&#x20;    - Path validity confirmation (cross-check against the classical A\*/Dijkstra baseline — flag if DDQN path length deviates from optimal by more than X%).

&#x20;    - Success/failure flag (did a valid path exist and was it found).



\---



\## 5. Proposed Repository Structure (v2)



```

Reinforcement\_Path\_Planning/

├── cad\_ingestion/

│   ├── convert\_cad\_to\_dxf.py       # wraps local ODA/LibreCAD converter

│   ├── dxf\_parser.py               # ezdxf-based entity extraction

│   └── scale\_utils.py              # units/scale detection + coord mapping

├── symbol\_model/

│   ├── symbol\_kb/                  # symbol class definitions, block-name lookup tables

│   ├── dataset\_gen.py              # synthetic DXF/labeled-data generator

│   ├── train\_symbol\_classifier.py  # CNN/GNN training script

│   └── infer\_symbols.py            # runs trained model on a new DXF

├── grid\_builder/

│   └── build\_grid.py               # symbols -> binary traversability grid

├── review\_ui/

│   └── grid\_review.py              # renders grid, captures corrections + start/end picks

├── rl\_agent/                       # existing, upgraded

│   ├── agent\_brain.py              # -> DDQN online + target network

│   ├── env.py                      # -> real-grid-driven environment

│   └── run\_agent.py                # training + inference entry point

├── output\_gen/

│   ├── export\_coordinates.py

│   ├── export\_annotated\_pdf.py

│   └── metrics\_report.py

├── Results/                        # existing, extended with new run artifacts

├── requirements.txt                # extend with ezdxf, opencv-python, torch/tensorflow, reportlab/matplotlib

└── README.md

```



\---



\## 6. Local/Open-Source Tech Stack (No API Keys Required)



| Need | Tool |

|---|---|

| CAD → DXF conversion | ODA File Converter (free, local CLI) / LibreCAD / FreeCAD Python API |

| DXF parsing | `ezdxf` |

| Raster/CV ops | OpenCV, NumPy |

| Symbol classifier | PyTorch or TensorFlow — small custom CNN or GNN (e.g., PyTorch Geometric) |

| RL framework | Custom DDQN on top of existing `agent\_brain.py`/`env.py`, or PyTorch + a lightweight custom Gym-style env (avoid heavy paid RL platforms) |

| PDF generation | `matplotlib` (vector export to PDF) or `reportlab`, using `ezdxf`'s drawing add-on to rasterize/vectorize DXF first |

| Grid/metrics visualization | `matplotlib` (already used in v1) |



\---



\## 7. Milestone Breakdown (Task List to Hand to Claude Code CLI)



1\. Set up `cad\_ingestion/` module: CAD→DXF conversion wrapper + DXF validity/entity-report logging + scale/unit extraction.

2\. Build the synthetic DXF dataset generator + assemble/download public floor-plan symbol datasets; define the full symbol class taxonomy in `symbol\_kb/`.

3\. Implement and train the custom symbol-recognition model (start with CNN-on-raster-crops baseline; iterate to GNN if accuracy is insufficient).

4\. Implement `dxf\_parser.py` + `infer\_symbols.py` to label entities in a \*new\* uploaded DXF using the trained model.

5\. Implement `build\_grid.py`: symbols → 2-color traversability grid, with the grid↔real-coordinate mapping persisted.

6\. Implement `grid\_review.py`: render grid, accept user corrections, accept start/end point selection, validate both points land on free cells.

7\. Upgrade `agent\_brain.py`/`env.py` to a DDQN (online + target network, replay buffer, CNN Q-function over the grid); keep v1's tabular Q-learning code path intact/available for comparison metrics.

8\. Implement a classical A\*/Dijkstra baseline solver to sanity-check DDQN output.

9\. Implement `export\_coordinates.py`, `export\_annotated\_pdf.py`, and `metrics\_report.py` for the three-part Stage 8 output.

10\. Wire all stages into a single `run\_pipeline.py` entry point mirroring today's `run\_agent.py` ergonomics (single command in → all artifacts out).

11\. Write/update `README.md` documenting the new end-to-end CAD-to-path workflow, replacing/extending the current "Installation/Running the Project" section.



\---



\## 8. Testing \& Validation Plan



\- \*\*Unit tests:\*\* DXF parsing on a handful of known CAD floor plans (verify entity counts and scale extraction).

\- \*\*Symbol classifier eval:\*\* held-out accuracy/F1 per symbol class; explicitly track door-recognition accuracy since it directly gates grid traversability correctness.

\- \*\*Grid correctness spot-check:\*\* manual visual QA comparing generated grid against source floor plan for a sample of test houses.

\- \*\*RL eval:\*\* compare DDQN path length/time against the A\*/Dijkstra baseline across multiple floor plans; track success rate (path found) and optimality gap (%).

\- \*\*Output eval:\*\* confirm PDF path overlay aligns spatially with the underlying drawing (no coordinate/scale drift) on at least one manually verified example.



\---



\## 9. Open Questions / Assumptions to Confirm Before Building



\- Target robot's physical footprint/turning radius isn't specified — affects whether the grid needs a "safety margin" dilation around obstacles (recommend adding one; confirm robot dimensions).

\- Whether multi-floor/multi-file plans are in scope for v2, or single-floor only (spec above assumes single floor plan per run, matching the "house floor plan" framing).

\- Whether the review/selection UI (Stage 5/6) should be a simple local web app, a CLI+image-viewer flow, or a Jupyter-based interactive tool — pick based on your deployment target.

\- Confirm acceptable inference latency budget for the DDQN rollout, since it affects how large a grid/how deep a network you can afford under the stated token/compute constraints.

