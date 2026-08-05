"""
train_symbol_classifier.py
───────────────────────────
Offline training script for the Tier-2 CNN symbol classifier.

Run ONCE before using the pipeline on new DXF files:
    python symbol_model/train_symbol_classifier.py

Steps:
  1. Generate synthetic dataset (if not already generated)
  2. Train SymbolCNN for 20 epochs with Adam + CrossEntropy
  3. Save best weights to models/symbol_cnn.pth
  4. Print val accuracy + per-class F1 report

Training takes ~5 min on CPU, ~30s on GPU.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from symbol_model.cnn_model import (
    SymbolCNN, SymbolDataset, NUM_CLASSES, CLASSES, CLASS_TO_IDX
)
from symbol_model.dataset_gen import generate_dataset

logger = logging.getLogger(__name__)


# ── Default Hyperparameters ───────────────────────────────────────────────────
DEFAULTS = {
    "data_dir": "data/synthetic",
    "output_weights": "models/symbol_cnn.pth",
    "n_plans": 300,
    "epochs": 25,
    "batch_size": 64,
    "lr": 1e-3,
    "lr_decay_epochs": [15, 20],
    "val_split": 0.2,
    "seed": 42,
}


def train(config: dict):
    """Main training loop."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Train] Device: {device}")

    # ── 1. Generate / verify dataset ──────────────────────────────────────────
    data_dir = Path(config["data_dir"])
    if not data_dir.exists() or sum(1 for _ in data_dir.rglob("*.png")) < 100:
        logger.info("[Train] Generating synthetic dataset...")
        generate_dataset(str(data_dir), n_plans=config["n_plans"], seed=config["seed"])
    else:
        total = sum(1 for _ in data_dir.rglob("*.png"))
        logger.info(f"[Train] Using existing dataset: {total} samples in {data_dir}")

    # ── 2. Build augmented dataset ────────────────────────────────────────────
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = SymbolDataset.from_directory(str(data_dir))
    n_total = len(full_dataset)
    n_val = int(n_total * config["val_split"])
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    # Apply different transforms
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    logger.info(f"[Train] Dataset split: {n_train} train / {n_val} val")

    # ── 3. Model + optimizer ──────────────────────────────────────────────────
    model = SymbolCNN(num_classes=NUM_CLASSES).to(device)
    logger.info(f"[Train] SymbolCNN params: {model.param_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_decay_epochs"], gamma=0.3
    )

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    output_weights = Path(config["output_weights"])
    output_weights.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= n_train
        scheduler.step()

        # Validation
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, n_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:>2}/{config['epochs']} | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"ValAcc={val_acc:.2%} | {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_weights))
            logger.info(f"  ✓ Best model saved (acc={val_acc:.2%})")

    logger.info(f"\n[Train] Complete. Best val accuracy: {best_val_acc:.2%}")
    logger.info(f"[Train] Weights saved to: {output_weights}")

    # ── 5. Per-class evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(str(output_weights), map_location=device))
    _print_classification_report(model, val_loader, device)

    return history, best_val_acc


def _evaluate(model, loader, criterion, device, n_samples) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / n_samples, correct / n_samples


def _print_classification_report(model, loader, device):
    from collections import defaultdict

    model.eval()
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1

    print("\n─── Per-class Accuracy ───────────────────────────────")
    for idx, cls_name in enumerate(CLASSES):
        total = per_class_total[idx]
        correct = per_class_correct[idx]
        acc = correct / total if total > 0 else 0.0
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cls_name:<22} [{bar}] {acc:.1%}  ({correct}/{total})")
    print("─────────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train Tier-2 CNN symbol classifier")
    parser.add_argument("--data-dir",  default=DEFAULTS["data_dir"])
    parser.add_argument("--output",    default=DEFAULTS["output_weights"])
    parser.add_argument("--n-plans",   type=int, default=DEFAULTS["n_plans"])
    parser.add_argument("--epochs",    type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch",     type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULTS["lr"])
    parser.add_argument("--seed",      type=int, default=DEFAULTS["seed"])
    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "output_weights": args.output,
        "n_plans": args.n_plans,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "lr_decay_epochs": DEFAULTS["lr_decay_epochs"],
        "val_split": DEFAULTS["val_split"],
        "seed": args.seed,
    }

    history, best_acc = train(config)
    print(f"\n✓ Training complete! Best validation accuracy: {best_acc:.2%}")
    print(f"✓ Weights saved to: {config['output_weights']}")
    print("\nNext step: Run the pipeline with your DXF file:")
    print("  python run_pipeline.py --input <your_floor_plan.dxf>")
