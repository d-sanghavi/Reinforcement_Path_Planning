"""
dataset_gen.py
──────────────
Synthetic DXF floor plan dataset generator for offline CNN training.

Generates thousands of labeled training examples by programmatically
constructing DXF files with known symbol types, then rasterizing them
into 64×64 PNG crops. No external datasets required.

Usage:
    python symbol_model/dataset_gen.py --output-dir data/synthetic --n-plans 200

Outputs:
    data/synthetic/
      wall/          *.png (crops)
      door/          *.png
      window/        *.png
      stairs/        *.png
      column/        *.png
      furniture_obstacle/  *.png
      annotation/    *.png
"""

import argparse
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Symbol classes to generate (subset — excludes 'unknown', 'opening', 'equipment')
GENERATE_CLASSES = [
    "wall", "door", "window", "stairs",
    "column", "furniture_obstacle", "annotation",
]

# Samples per class per floor plan
SAMPLES_PER_CLASS_PER_PLAN = {
    "wall": 4,
    "door": 3,
    "window": 2,
    "stairs": 1,
    "column": 3,
    "furniture_obstacle": 4,
    "annotation": 2,
}

IMG_SIZE = 64


def generate_dataset(output_dir: str, n_plans: int = 200, seed: int = 42):
    """
    Generate synthetic training dataset.

    Parameters
    ----------
    output_dir : str
        Root directory to write class-named subdirectories.
    n_plans : int
        Number of synthetic floor plans to generate per class.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)

    # Clean and recreate output dirs
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for cls in GENERATE_CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    counters = {cls: 0 for cls in GENERATE_CLASSES}

    for plan_idx in range(n_plans):
        for cls in GENERATE_CLASSES:
            n = SAMPLES_PER_CLASS_PER_PLAN.get(cls, 2)
            for _ in range(n):
                img = _generate_sample(cls)
                fname = output_dir / cls / f"{cls}_{plan_idx:04d}_{counters[cls]:04d}.png"
                cv2.imwrite(str(fname), img)
                counters[cls] += 1

        if plan_idx % 50 == 0:
            logger.info(f"[DatasetGen] Generated plan {plan_idx}/{n_plans}")

    total = sum(counters.values())
    logger.info(f"[DatasetGen] Done. Total samples: {total}")
    for cls, cnt in counters.items():
        logger.info(f"  {cls}: {cnt} samples")

    return counters


def _generate_sample(class_name: str) -> np.ndarray:
    """Generate a single 64×64 grayscale raster for the given class."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Add slight noise for augmentation
    noise = np.random.randint(0, 15, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    img = cv2.add(img, noise)

    generators = {
        "wall": _gen_wall,
        "door": _gen_door,
        "window": _gen_window,
        "stairs": _gen_stairs,
        "column": _gen_column,
        "furniture_obstacle": _gen_furniture,
        "annotation": _gen_annotation,
    }

    gen_fn = generators.get(class_name, _gen_wall)
    img = gen_fn(img)

    # Apply random augmentation (rotation, flip, brightness)
    img = _augment(img)

    return img


# ── Symbol generators ──────────────────────────────────────────────────────────

def _gen_wall(img: np.ndarray) -> np.ndarray:
    """Long straight line segment (horizontal, vertical, or diagonal)."""
    margin = 8
    angle = random.choice([0, 45, 90, 135, random.uniform(0, 180)])
    angle_r = math.radians(angle)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    length = random.randint(30, 55)
    thickness = random.choice([1, 2, 3])

    dx = int(math.cos(angle_r) * length / 2)
    dy = int(math.sin(angle_r) * length / 2)

    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy

    # Double-line wall (parallel lines)
    if random.random() > 0.5:
        offset = random.randint(3, 6)
        perp_angle = angle_r + math.pi / 2
        ox = int(math.cos(perp_angle) * offset)
        oy = int(math.sin(perp_angle) * offset)
        cv2.line(img, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), 240, thickness)
        cv2.line(img, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), 240, thickness)
    else:
        cv2.line(img, (x1, y1), (x2, y2), 240, thickness)

    return img


def _gen_door(img: np.ndarray) -> np.ndarray:
    """Standard swing door: line (door leaf) + quarter-circle arc."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    radius = random.randint(12, 22)
    wall_angle = random.uniform(0, 360)
    wall_r = math.radians(wall_angle)
    sweep = random.uniform(85, 95)  # ~90 degrees

    # Door leaf line
    lx = int(cx + radius * math.cos(wall_r))
    ly = int(cy + radius * math.sin(wall_r))
    cv2.line(img, (cx, cy), (lx, ly), 240, 2)

    # Arc (quarter circle)
    start_angle = int(-wall_angle)
    end_angle = int(-wall_angle - sweep)
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, end_angle, start_angle, 200, 1)

    # Short wall segment at hinge point
    hinge_len = random.randint(5, 10)
    perp_r = wall_r + math.pi / 2
    wx = int(cx + hinge_len * math.cos(perp_r))
    wy = int(cy + hinge_len * math.sin(perp_r))
    cv2.line(img, (cx, cy), (wx, wy), 240, 2)

    return img


def _gen_window(img: np.ndarray) -> np.ndarray:
    """Window: wall segment with parallel internal tick marks."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    width = random.randint(20, 35)
    thickness = random.randint(4, 8)

    angle = random.choice([0, 90, 45])
    angle_r = math.radians(angle)

    # Outer wall lines (double line)
    dx = int(math.cos(angle_r) * width / 2)
    dy = int(math.sin(angle_r) * width / 2)
    perp_r = angle_r + math.pi / 2
    ox = int(math.cos(perp_r) * thickness / 2)
    oy = int(math.sin(perp_r) * thickness / 2)

    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    cv2.line(img, (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy), 240, 1)
    cv2.line(img, (p1[0] - ox, p1[1] - oy), (p2[0] - ox, p2[1] - oy), 240, 1)
    # Glass line in middle
    cv2.line(img, p1, p2, 180, 1)

    # End caps
    cv2.line(img, (p1[0] + ox, p1[1] + oy), (p1[0] - ox, p1[1] - oy), 240, 1)
    cv2.line(img, (p2[0] + ox, p2[1] + oy), (p2[0] - ox, p2[1] - oy), 240, 1)

    return img


def _gen_stairs(img: np.ndarray) -> np.ndarray:
    """Staircase: parallel horizontal lines of equal length."""
    n_steps = random.randint(5, 10)
    step_height = IMG_SIZE // (n_steps + 2)
    width = random.randint(25, 50)
    cx = IMG_SIZE // 2

    for i in range(n_steps):
        y = 8 + i * step_height
        x1 = cx - width // 2
        x2 = cx + width // 2
        cv2.line(img, (x1, y), (x2, y), 240, 1)

    # Direction arrow
    arrow_y = 8 + n_steps * step_height
    cv2.arrowedLine(img, (cx, arrow_y), (cx, arrow_y + 8), 180, 1, tipLength=0.5)

    return img


def _gen_column(img: np.ndarray) -> np.ndarray:
    """Column: small filled square or circle."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    size = random.randint(8, 16)

    if random.random() > 0.5:
        # Square column
        pts = np.array([
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size],
        ])
        cv2.fillPoly(img, [pts], 200)
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), 240, 2)
    else:
        # Round column
        cv2.circle(img, (cx, cy), size, 200, -1)
        cv2.circle(img, (cx, cy), size, 240, 2)

    return img


def _gen_furniture(img: np.ndarray) -> np.ndarray:
    """Furniture: various closed rectangles representing tables, chairs, etc."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    choice = random.randint(0, 4)

    if choice == 0:
        # Table (rectangle)
        w, h = random.randint(20, 35), random.randint(12, 22)
        cv2.rectangle(img, (cx - w, cy - h), (cx + w, cy + h), 220, 1)

    elif choice == 1:
        # Chair (small square with circle)
        s = random.randint(10, 16)
        cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), 220, 1)
        cv2.circle(img, (cx, cy - s - 4), 4, 200, 1)

    elif choice == 2:
        # Bed (large rectangle with pillow indicators)
        w, h = random.randint(18, 28), random.randint(25, 38)
        cv2.rectangle(img, (cx - w, cy - h), (cx + w, cy + h), 220, 1)
        cv2.rectangle(img, (cx - w + 3, cy - h + 3), (cx + w - 3, cy - h + 10), 200, 1)

    elif choice == 3:
        # Desk/counter (L-shape)
        w, h = random.randint(15, 25), random.randint(15, 25)
        pts = np.array([
            [cx - w, cy - h], [cx + w, cy - h],
            [cx + w, cy], [cx, cy],
            [cx, cy + h], [cx - w, cy + h],
        ])
        cv2.polylines(img, [pts], True, 220, 1)

    else:
        # Sofa (rounded rectangle)
        w, h = random.randint(20, 30), random.randint(10, 18)
        cv2.rectangle(img, (cx - w, cy - h), (cx + w, cy + h), 220, 2)
        # Cushion dividers
        cv2.line(img, (cx, cy - h), (cx, cy + h), 200, 1)

    return img


def _gen_annotation(img: np.ndarray) -> np.ndarray:
    """Annotation: text-like patterns, dimension lines."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    choice = random.randint(0, 2)

    if choice == 0:
        # Dimension line with arrowheads
        y = cy
        cv2.line(img, (10, y), (54, y), 200, 1)
        cv2.line(img, (10, y - 5), (10, y + 5), 200, 1)  # tick
        cv2.line(img, (54, y - 5), (54, y + 5), 200, 1)  # tick
        # Hatched text placeholder
        for i in range(3):
            cv2.line(img, (20 + i * 8, y - 10), (20 + i * 8, y - 4), 180, 1)

    elif choice == 1:
        # Leader arrow + text
        cv2.arrowedLine(img, (10, cy), (30, cy - 10), 200, 1)
        cv2.line(img, (30, cy - 10), (54, cy - 10), 200, 1)

    else:
        # Hatch pattern (diagonal lines)
        for i in range(-20, 70, 5):
            x1, y1 = max(0, i), max(0, 0)
            x2, y2 = min(IMG_SIZE, i + IMG_SIZE), min(IMG_SIZE, IMG_SIZE)
            cv2.line(img, (x1, y1), (x2, y2), 120, 1)

    return img


def _augment(img: np.ndarray) -> np.ndarray:
    """Apply random augmentation to a generated sample."""
    # Random rotation (0, 90, 180, 270 + small jitter)
    k = random.randint(0, 3)
    img = np.rot90(img, k)

    # Random horizontal flip
    if random.random() > 0.5:
        img = np.fliplr(img)

    # Random brightness adjustment
    bright_offset = random.randint(-20, 20)
    img = np.clip(img.astype(int) + bright_offset, 0, 255).astype(np.uint8)

    # Slight Gaussian blur (simulates rasterization artifacts)
    if random.random() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic DXF symbol training data")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--n-plans", type=int, default=200, help="Number of floor plans to synthesize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    counts = generate_dataset(args.output_dir, args.n_plans, args.seed)
    total = sum(counts.values())
    print(f"\n✓ Dataset generated: {total} samples in '{args.output_dir}'")
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls:<25} {cnt:>4} samples")
