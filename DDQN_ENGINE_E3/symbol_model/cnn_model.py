"""
cnn_model.py
────────────
Tier-2 CNN-based symbol classifier for DXF entity group crops.

Architecture: Lightweight custom CNN (~500K parameters)
  - Input: 64×64 grayscale raster crop of an entity group
  - 3 Conv blocks → AdaptiveAvgPool → 2 FC layers → N class logits
  - Suitable for CPU inference (no GPU required)
  - Training: ~20 epochs on synthetic data, ~5 min on modern CPU

Classes (9):
  0: wall
  1: door
  2: window
  3: opening
  4: stairs
  5: column
  6: furniture_obstacle
  7: annotation
  8: unknown
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Class mapping ─────────────────────────────────────────────────────────────
CLASSES = [
    "wall", "door", "window", "opening",
    "stairs", "column", "furniture_obstacle", "annotation", "unknown",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ── Input spec ───────────────────────────────────────────────────────────────
INPUT_SIZE = 64   # pixels (square)


# ═══════════════════════════════════════════════════════════════════════════════
# CNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SymbolCNN(nn.Module):
    """
    Lightweight CNN for floor plan symbol classification.

    Input:  (B, 1, 64, 64) grayscale
    Output: (B, NUM_CLASSES) logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32, pool=True),   # → (B, 32, 32, 32)
            ConvBlock(32,  64, pool=True),   # → (B, 64, 16, 16)
            ConvBlock(64, 128, pool=True),   # → (B, 128, 8, 8)
            ConvBlock(128, 256, pool=False), # → (B, 256, 8, 8)
            nn.AdaptiveAvgPool2d((4, 4)),    # → (B, 256, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET (for training)
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolDataset(torch.utils.data.Dataset):
    """
    Dataset of rasterized DXF entity crops with class labels.

    Expects a directory structure:
        data_dir/
          wall/
            *.png
          door/
            *.png
          ...

    Or a flat list of (image_path, class_idx) tuples.
    """

    def __init__(self, samples: list, transform=None):
        """
        Parameters
        ----------
        samples : list of (image_path: str, class_idx: int)
        transform : torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torchvision.transforms as T
        from PIL import Image

        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        img = img.resize((INPUT_SIZE, INPUT_SIZE))

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, label

    @classmethod
    def from_directory(cls, data_dir: str, transform=None):
        """Build dataset from class-named subdirectories."""
        data_dir = Path(data_dir)
        samples = []
        for class_name in CLASSES:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.png"):
                samples.append((str(img_path), CLASS_TO_IDX[class_name]))
        logger.info(f"[SymbolDataset] Loaded {len(samples)} samples from {data_dir}")
        return cls(samples, transform)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(weights_path: str, device: str = "cpu") -> Optional[SymbolCNN]:
    """
    Load a trained SymbolCNN from saved weights.

    Returns None if the file doesn't exist (graceful degradation to Tier-1 only).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.warning(
            f"[CNN] No weights found at {weights_path}. "
            "Running in Tier-1 heuristic-only mode. "
            "Run: python symbol_model/train_symbol_classifier.py to train."
        )
        return None

    model = SymbolCNN()
    try:
        state = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        logger.info(f"[CNN] Loaded weights from {weights_path} ({model.param_count:,} params)")
        return model
    except Exception as e:
        logger.error(f"[CNN] Failed to load weights: {e}")
        return None


def rasterize_entity_group(group, target_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Rasterize an EntityGroup into a fixed-size grayscale numpy array.

    Renders all line/arc/polyline primitives as white strokes on black background.
    Returns shape (target_size, target_size) uint8 in [0, 255].
    """
    import cv2

    img = np.zeros((target_size, target_size), dtype=np.uint8)

    bbox = group.bbox
    w = bbox.width or 1.0
    h = bbox.height or 1.0

    def to_px(x, y):
        px = int((x - bbox.min_x) / w * (target_size - 1))
        py = int((bbox.max_y - y) / h * (target_size - 1))  # flip Y
        return (
            max(0, min(target_size - 1, px)),
            max(0, min(target_size - 1, py)),
        )

    for entity in group.entities:
        pts = entity.points
        if not pts:
            continue

        if entity.entity_type in ("LINE", "SPLINE"):
            for i in range(len(pts) - 1):
                p1 = to_px(*pts[i])
                p2 = to_px(*pts[i + 1])
                cv2.line(img, p1, p2, 255, 1)

        elif entity.entity_type == "LWPOLYLINE":
            for i in range(len(pts) - 1):
                p1 = to_px(*pts[i])
                p2 = to_px(*pts[i + 1])
                cv2.line(img, p1, p2, 255, 1)
            if entity.extra.get("is_closed", False) and len(pts) > 2:
                cv2.line(img, to_px(*pts[-1]), to_px(*pts[0]), 255, 1)

        elif entity.entity_type == "ARC":
            center_dxf = entity.extra.get("center", (0, 0))
            cx_px, cy_px = to_px(*center_dxf)
            r_px = max(1, int(entity.extra.get("radius", 0) / w * (target_size - 1)))
            start_a = int(entity.extra.get("start_angle", 0))
            end_a = int(entity.extra.get("end_angle", 360)) % 360
            cv2.ellipse(img, (cx_px, cy_px), (r_px, r_px), 0, -end_a, -start_a, 255, 1)

        elif entity.entity_type == "CIRCLE":
            center_dxf = entity.extra.get("center", (0, 0))
            cx_px, cy_px = to_px(*center_dxf)
            r_px = max(1, int(entity.extra.get("radius", 0) / w * (target_size - 1)))
            cv2.circle(img, (cx_px, cy_px), r_px, 255, 1)

    return img


def predict_group(
    model: SymbolCNN,
    group,
    device: str = "cpu",
) -> tuple:
    """
    Run CNN inference on a single entity group.

    Returns (class_name: str, confidence: float)
    """
    import torchvision.transforms as T

    img_np = rasterize_entity_group(group)
    tensor = T.ToTensor()(img_np).unsqueeze(0).to(device)  # (1,1,64,64)

    probs = model.predict_proba(tensor)  # (1, NUM_CLASSES)
    probs_np = probs.cpu().numpy()[0]

    best_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[best_idx])
    class_name = IDX_TO_CLASS[best_idx]

    return class_name, confidence
