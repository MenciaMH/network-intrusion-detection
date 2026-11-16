"""
Fine-tunes the trained CNN (epoch-28 checkpoint from 8_train_cnn.py) with one
targeted change: BENIGN's class weight is raised, informed directly by the
Grad-CAM findings (10_gradcam.py, Notion Work log Section 19).

This is NOT a retrain from scratch. It loads the existing epoch-28 weights
and continues training for a handful of epochs at a lower learning rate, so
the 28 epochs of convergence already achieved aren't undone by a large step.

Why raise BENIGN's weight specifically (see Work log Sections 15 and 19):
    - Section 15 found BENIGN's balanced class weight (0.179) was a clear
      outlier -- 2.5x lower than the next-lowest class (DoS Hulk, 0.454) --
      which makes misclassifying BENIGN artificially "cheap" for a weighted
      loss, biasing the model against predicting BENIGN on ambiguous flows.
    - Section 19's Grad-CAM results confirmed this isn't purely a weight
      artifact -- there's real timing-feature overlap between some BENIGN
      flows and Bot/Web Attack/PortScan -- so this fine-tune is expected to
      REDUCE the leakage, not eliminate it. Whatever precision gap remains
      afterward gets documented as a feature-overlap limitation, not chased
      further.

config.BENIGN_WEIGHT_MULTIPLIER (2.5) brings BENIGN's weight back in line
with DoS Hulk's instead of leaving it as the obvious outlier.

Pipeline position:
    models/cnn_resnet18_best.pt (epoch-28 checkpoint, REQUIRED -- see
    10_gradcam.py's same requirement and its note on why this likely only
    exists on Google Drive, not locally)
    -> [THIS SCRIPT] -> models/cnn_resnet18_finetuned_benign_weight.pt,
       reports/cnn_finetune_*

Usage:
    python 11_finetune_cnn_benign_weight.py
    python 11_finetune_cnn_benign_weight.py --epochs 5   # override config.py
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

import config
from flow_image_dataset import FlowImageDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TRAIN_GRIDS_PATH = config.TRAIN_GRIDS_PATH
TRAIN_LABELS_PATH = config.TRAIN_LABELS_PATH
TEST_GRIDS_PATH = config.TEST_GRIDS_PATH
TEST_LABELS_PATH = config.TEST_LABELS_PATH
ENCODING_METADATA_PATH = config.ENCODING_METADATA_PATH

MODELS_DIR = config.MODELS_DIR
BASE_CHECKPOINT_PATH = config.CNN_MODEL_PATH  # the epoch-28 checkpoint to start from
FINETUNE_MODEL_PATH = config.CNN_FINETUNE_MODEL_PATH

REPORTS_DIR = config.REPORTS_DIR
TRAINING_LOG_CSV = config.CNN_FINETUNE_TRAINING_LOG_CSV
CLASSIFICATION_REPORT_CSV = config.CNN_FINETUNE_CLASSIFICATION_REPORT_CSV
CONFUSION_MATRIX_PNG = config.CNN_FINETUNE_CONFUSION_MATRIX_PNG
TRAINING_CURVES_PNG = config.CNN_FINETUNE_TRAINING_CURVES_PNG

CNN_INPUT_SIZE = config.CNN_INPUT_SIZE
VAL_SIZE = config.CNN_VAL_SIZE
BATCH_SIZE = config.CNN_BATCH_SIZE
NUM_WORKERS = config.CNN_NUM_WORKERS
LEARNING_RATE = config.CNN_FINETUNE_LEARNING_RATE
WEIGHT_DECAY = config.CNN_WEIGHT_DECAY
MAX_EPOCHS = config.CNN_FINETUNE_MAX_EPOCHS
PATIENCE = config.CNN_FINETUNE_PATIENCE
RANDOM_STATE = config.RANDOM_STATE
BENIGN_CLASS_NAME = config.BENIGN_CLASS_NAME
BENIGN_WEIGHT_MULTIPLIER = config.BENIGN_WEIGHT_MULTIPLIER
BASELINE_PRECISION_REFERENCE = config.BASELINE_PRECISION_REFERENCE


# --------------------------------------------------------------------------- #
# Model / helpers (identical to 8_train_cnn.py)
# --------------------------------------------------------------------------- #


def build_model(n_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_samples = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            preds = outputs.argmax(dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    mean_loss = total_loss / n_samples
    return mean_loss, np.array(all_true), np.array(all_pred)


def compute_full_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    present_classes = np.unique(y)
    weights_present = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y
    )
    weights = np.ones(n_classes, dtype=np.float32)
    for cls, w in zip(present_classes, weights_present):
        weights[cls] = w
    return weights


def apply_benign_weight_multiplier(
    class_weights: np.ndarray, class_names: list[str]
) -> np.ndarray:
    """Multiplies BENIGN's balanced weight by BENIGN_WEIGHT_MULTIPLIER,
    leaving every other class's weight untouched. See the module docstring
    and config.py's comment for why."""
    weights = class_weights.copy()
    benign_idx = class_names.index(BENIGN_CLASS_NAME)
    before = weights[benign_idx]
    weights[benign_idx] = before * BENIGN_WEIGHT_MULTIPLIER
    print(
        f"  BENIGN weight: {before:.3f} -> {weights[benign_idx]:.3f} "
        f"(x{BENIGN_WEIGHT_MULTIPLIER}, per Work log Sections 15/19)"
    )
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the CNN from its epoch-28 checkpoint with a raised BENIGN class weight.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Override CNN_FINETUNE_MAX_EPOCHS from config.py (default: {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--benign-weight-multiplier", type=float, default=None,
        help=f"Override config.BENIGN_WEIGHT_MULTIPLIER (default: {BENIGN_WEIGHT_MULTIPLIER}).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    max_epochs = args.epochs if args.epochs is not None else MAX_EPOCHS
    benign_multiplier = (
        args.benign_weight_multiplier
        if args.benign_weight_multiplier is not None
        else BENIGN_WEIGHT_MULTIPLIER
    )

    for path in (
        TRAIN_GRIDS_PATH, TRAIN_LABELS_PATH, TEST_GRIDS_PATH, TEST_LABELS_PATH,
        ENCODING_METADATA_PATH,
    ):
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            print("Run 7_flow_to_image_encoding.py first.")
            return

    if not os.path.exists(BASE_CHECKPOINT_PATH):
        print(f"Required file not found: {BASE_CHECKPOINT_PATH}")
        print(
            "This needs the REAL trained checkpoint (epoch 28), which lives on Google "
            "Drive (training ran on Colab -- see Work log Section 14). Either copy it "
            "into models/cnn_resnet18_best.pt, or run BDAPP_finetune_cnn_colab.ipynb "
            "instead, which loads it from Drive directly."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading encoding metadata from: {ENCODING_METADATA_PATH}")
    metadata = joblib.load(ENCODING_METADATA_PATH)
    label_encoder = metadata["label_encoder"]
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)
    print(f"  {n_classes} classes: {class_names}")

    print(f"\nLoading train grids/labels from: {TRAIN_GRIDS_PATH}")
    train_grids_full = np.load(TRAIN_GRIDS_PATH)
    train_labels_full = np.load(TRAIN_LABELS_PATH)
    print(f"  {train_grids_full.shape[0]:,} rows")

    print(f"Loading test grids/labels from: {TEST_GRIDS_PATH}")
    test_grids = np.load(TEST_GRIDS_PATH)
    test_labels = np.load(TEST_LABELS_PATH)
    print(f"  {test_grids.shape[0]:,} rows")

    class_counts = np.bincount(train_labels_full, minlength=n_classes)
    can_stratify = class_counts[class_counts > 0].min() >= 2
    train_grids, val_grids, train_labels, val_labels = train_test_split(
        train_grids_full, train_labels_full,
        test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=train_labels_full if can_stratify else None,
    )
    print(f"  Train: {len(train_grids):,}  Val: {len(val_grids):,}  Test: {len(test_grids):,}")

    train_ds = FlowImageDataset(train_grids, train_labels, CNN_INPUT_SIZE)
    val_ds = FlowImageDataset(val_grids, val_labels, CNN_INPUT_SIZE)
    test_ds = FlowImageDataset(test_grids, test_labels, CNN_INPUT_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # --- Load the epoch-28 checkpoint as the starting point ---
    print(f"\nLoading base checkpoint: {BASE_CHECKPOINT_PATH}")
    base_checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=device)
    model = build_model(n_classes).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])
    print(
        f"  Loaded epoch {base_checkpoint['epoch']} "
        f"(val macro-F1={base_checkpoint['val_macro_f1']:.4f}) as fine-tune starting point."
    )

    # --- Class weights: same balanced computation, then BENIGN boosted ---
    class_weights = compute_full_class_weights(train_labels, n_classes)
    print(f"\n  Balanced class weights: {dict(zip(class_names, class_weights.round(3)))}")
    class_weights = apply_benign_weight_multiplier(class_weights, class_names)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    log_rows = []

    print(
        f"\nFine-tuning for up to {max_epochs} epochs "
        f"(lr={LEARNING_RATE:.1e}, patience={PATIENCE}, metric=val macro-F1)...\n"
    )
    for epoch in range(1, max_epochs + 1):
        start = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion, device, optimizer=None)

        val_accuracy = float((val_true == val_pred).mean())
        val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | val_macro_f1={val_macro_f1:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
        })

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "class_names": class_names,
                    "base_checkpoint_epoch": base_checkpoint["epoch"],
                    "benign_weight_multiplier": benign_multiplier,
                },
                FINETUNE_MODEL_PATH,
            )
            print(f"  -> New best val macro-F1 ({val_macro_f1:.4f}). Saved to {FINETUNE_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"\nEarly stopping: no val macro-F1 improvement for "
                    f"{PATIENCE} epochs (best={best_val_f1:.4f} at epoch "
                    f"{epoch - epochs_without_improvement})."
                )
                break

    with open(TRAINING_LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nFine-tune training log saved to: {TRAINING_LOG_CSV}")

    epochs_ran = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs_ran, [r["train_loss"] for r in log_rows], label="train")
    axes[0].plot(epochs_ran, [r["val_loss"] for r in log_rows], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Fine-tune loss per epoch")
    axes[0].legend()

    axes[1].plot(epochs_ran, [r["val_accuracy"] for r in log_rows], label="val accuracy")
    axes[1].plot(epochs_ran, [r["val_macro_f1"] for r in log_rows], label="val macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Fine-tune validation metrics per epoch")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(TRAINING_CURVES_PNG, dpi=150)
    plt.close(fig)
    print(f"Fine-tune training curves saved to: {TRAINING_CURVES_PNG}")

    # --- Final evaluation on the untouched test set, using the BEST fine-tuned checkpoint ---
    checkpoint = torch.load(FINETUNE_MODEL_PATH, map_location=device)
    print(f"\nLoading best fine-tuned checkpoint (epoch {checkpoint['epoch']}) for final test evaluation...")
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_true, test_pred = run_epoch(model, test_loader, criterion, device, optimizer=None)

    all_labels = np.arange(n_classes)
    report_dict = classification_report(
        test_true, test_pred, labels=all_labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(CLASSIFICATION_REPORT_CSV)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (test set, best fine-tuned checkpoint)")
    print("=" * 70)
    print(classification_report(
        test_true, test_pred, labels=all_labels, target_names=class_names, zero_division=0
    ))
    print(f"Saved to: {CLASSIFICATION_REPORT_CSV}")

    cm = confusion_matrix(test_true, test_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("CNN (ResNet18) - Confusion Matrix, fine-tuned (test set)")
    plt.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PNG}")

    # --- Direct before/after on exactly the 3 classes Grad-CAM analyzed ---
    print("\n" + "=" * 70)
    print("PRECISION: BASELINE (Section 15) vs FINE-TUNED (this run)")
    print("=" * 70)
    for cls_name, baseline_precision in BASELINE_PRECISION_REFERENCE.items():
        new_precision = report_dict.get(cls_name, {}).get("precision")
        if new_precision is None:
            continue
        delta = new_precision - baseline_precision
        arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "=")
        print(f"  {cls_name:<12s}  {baseline_precision:.2f} -> {new_precision:.2f}  ({arrow} {delta:+.2f})")

    print(
        f"\nBest fine-tuned checkpoint: {FINETUNE_MODEL_PATH} "
        f"(val macro-F1={checkpoint['val_macro_f1']:.4f}, epoch {checkpoint['epoch']}, "
        f"started from base epoch {checkpoint['base_checkpoint_epoch']})"
    )


if __name__ == "__main__":
    main()
