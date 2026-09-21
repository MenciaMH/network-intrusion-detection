"""
Trains a ResNet18 CNN (ImageNet-pretrained, transfer learning) on the
flow-to-image grids produced by 7_flow_to_image_encoding.py.

Design choices:

    - Transfer learning from ImageNet-pretrained ResNet18: the final fully
      connected layer is replaced to output n_classes instead of 1000, the
      rest of the network is fine-tuned (not frozen) since the input
      domain (flow-image grids) is far enough from natural photos that
      frozen early layers wouldn't obviously help, and the dataset is large
      enough (690K+ train rows) to fine-tune the full network without
      immediately overfitting.

    - Upscaling from the stored grid_size x grid_size grids to 224x224 is
      lazy, per-sample, via FlowImageDataset (nearest-neighbor only -- see
      flow_image_dataset.py's docstring).

    - Class weights in the loss (class_weight='balanced'-equivalent,
      computed with sklearn's compute_class_weight), matching the RF
      baseline's approach and the class-imbalance strategy documented in
      the README: BENIGN is still ~50.68% of train_undersampled even after
      undersampling, so the loss needs to keep compensating for that.

    - A held-out validation split (10% of train_undersampled, stratified)
      is used for early stopping and picking the best epoch, so the actual
      test set stays untouched until the final evaluation -- consistent
      with how test.parquet was used in train_baseline_rf.py.

    - The metric tracked for "best epoch" and early stopping is
      **macro-F1** on validation, not accuracy or loss: with this class
      imbalance, accuracy or micro-averaged loss can look good while a
      small class like Bot is still being missed. Macro-F1 weighs every
      class equally, mirroring why train_baseline_rf.py reports per-class
      metrics instead of accuracy alone.

    - Per-epoch metrics (train loss, val loss, val accuracy, val macro-F1,
      learning rate) are logged to a CSV for the report's training curves,
      and a PNG plot is generated directly from that CSV.

Pipeline position:
    flow_images/{train,test}_grids.npy + labels + encoding_metadata.joblib
    -> [THIS SCRIPT] -> models/cnn_resnet18_best.pt, reports/cnn_*

Usage:
    python 8_train_cnn.py                    # full run, per config.py
    python 8_train_cnn.py --smoke-test        # 1 epoch, small subsample --
                                               # sanity-check memory/time
                                               # before committing to a full
                                               # run (see --smoke-test below)
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
CNN_MODEL_PATH = config.CNN_MODEL_PATH

REPORTS_DIR = config.REPORTS_DIR
TRAINING_LOG_CSV = config.CNN_TRAINING_LOG_CSV
CLASSIFICATION_REPORT_CSV = config.CNN_CLASSIFICATION_REPORT_CSV
CONFUSION_MATRIX_PNG = config.CNN_CONFUSION_MATRIX_PNG
TRAINING_CURVES_PNG = config.CNN_TRAINING_CURVES_PNG

CNN_INPUT_SIZE = config.CNN_INPUT_SIZE
VAL_SIZE = config.CNN_VAL_SIZE
BATCH_SIZE = config.CNN_BATCH_SIZE
NUM_WORKERS = config.CNN_NUM_WORKERS
LEARNING_RATE = config.CNN_LEARNING_RATE
WEIGHT_DECAY = config.CNN_WEIGHT_DECAY
MAX_EPOCHS = config.CNN_MAX_EPOCHS
PATIENCE = config.CNN_EARLY_STOPPING_PATIENCE
RANDOM_STATE = config.RANDOM_STATE


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(n_classes: int) -> nn.Module:
    """ImageNet-pretrained ResNet18 with the final FC layer replaced to
    output n_classes. The rest of the network is fine-tuned, not frozen."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Runs one epoch. If optimizer is given, trains; otherwise evaluates
    (no gradient updates). Returns (mean_loss, y_true, y_pred)."""
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
    """Like sklearn's compute_class_weight(class_weight='balanced', ...),
    but tolerant of classes missing from y (weight defaults to 1.0 for
    those) -- sklearn's version raises if any class in `classes` doesn't
    appear in y at all, which a small --smoke-test subsample can easily
    trigger for a rare class like Bot (~0.08% of the data)."""
    present_classes = np.unique(y)
    weights_present = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y
    )
    weights = np.ones(n_classes, dtype=np.float32)
    for cls, w in zip(present_classes, weights_present):
        weights[cls] = w
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the flow-to-image CNN (ResNet18).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Quick sanity run before committing to the full training job: "
            "1 epoch on a small subsample (train<=5000, val<=1000, "
            "test<=1000 rows by default), to confirm the pipeline runs "
            "without a memory/CUDA/shape error and to see how long one "
            "epoch takes (extrapolate to the full ~690K-row epoch from "
            "that). Writes to separate *_smoketest output files, so it "
            "never overwrites a real run's model/report/plots."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Override CNN_MAX_EPOCHS from config.py (default: {MAX_EPOCHS}; "
             f"--smoke-test alone defaults this to 1).",
    )
    parser.add_argument(
        "--limit-train", type=int, default=None,
        help="Use only this many rows from train_undersampled (before the "
             "train/val split). --smoke-test alone defaults this to 5000.",
    )
    parser.add_argument(
        "--limit-val", type=int, default=None,
        help="Cap the validation set to this many rows. --smoke-test alone "
             "defaults this to 1000.",
    )
    parser.add_argument(
        "--limit-test", type=int, default=None,
        help="Cap the test set to this many rows. --smoke-test alone "
             "defaults this to 1000.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()

    max_epochs = args.epochs if args.epochs is not None else MAX_EPOCHS
    limit_train = args.limit_train
    limit_val = args.limit_val
    limit_test = args.limit_test
    model_path = CNN_MODEL_PATH
    training_log_csv = TRAINING_LOG_CSV
    classification_report_csv = CLASSIFICATION_REPORT_CSV
    confusion_matrix_png = CONFUSION_MATRIX_PNG
    training_curves_png = TRAINING_CURVES_PNG

    if args.smoke_test:
        max_epochs = args.epochs if args.epochs is not None else 1
        limit_train = args.limit_train if args.limit_train is not None else 5000
        limit_val = args.limit_val if args.limit_val is not None else 1000
        limit_test = args.limit_test if args.limit_test is not None else 1000
        model_path = CNN_MODEL_PATH.replace(".pt", "_smoketest.pt")
        training_log_csv = TRAINING_LOG_CSV.replace(".csv", "_smoketest.csv")
        classification_report_csv = CLASSIFICATION_REPORT_CSV.replace(".csv", "_smoketest.csv")
        confusion_matrix_png = CONFUSION_MATRIX_PNG.replace(".png", "_smoketest.png")
        training_curves_png = TRAINING_CURVES_PNG.replace(".png", "_smoketest.png")
        print(
            "\n*** SMOKE TEST MODE ***\n"
            f"  epochs={max_epochs}  train<={limit_train}  val<={limit_val}  test<={limit_test}\n"
            "  This is ONLY to check the pipeline runs and to time one epoch -- "
            "metrics from this run are meaningless (tiny, non-stratified "
            "subsample). Outputs go to *_smoketest files, never overwriting a "
            "real run.\n"
        )

    for path in (TRAIN_GRIDS_PATH, TRAIN_LABELS_PATH, TEST_GRIDS_PATH, TEST_LABELS_PATH, ENCODING_METADATA_PATH):
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            print("Run 7_flow_to_image_encoding.py first.")
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
    print(f"  {train_grids_full.shape[0]:,} rows, grid shape {train_grids_full.shape[1:]}")

    print(f"Loading test grids/labels from: {TEST_GRIDS_PATH}")
    test_grids = np.load(TEST_GRIDS_PATH)
    test_labels = np.load(TEST_LABELS_PATH)
    print(f"  {test_grids.shape[0]:,} rows")

    # --smoke-test / --limit-*: take a plain slice (train_undersampled and
    # test were both already shuffled upstream -- step 5's undersampling
    # re-shuffles, and train_test_split shuffles by default -- so this is a
    # reasonable-enough sample for a speed/memory check, not a stratified
    # sample suitable for real metrics).
    if limit_train is not None and limit_train < len(train_grids_full):
        train_grids_full = train_grids_full[:limit_train]
        train_labels_full = train_labels_full[:limit_train]
        print(f"  [limit] train truncated to {len(train_grids_full):,} rows")
    if limit_test is not None and limit_test < len(test_grids):
        test_grids = test_grids[:limit_test]
        test_labels = test_labels[:limit_test]
        print(f"  [limit] test truncated to {len(test_grids):,} rows")

    # --- Stratified train/val split (test stays untouched until the end) ---
    # A tiny --limit-train subsample can leave a rare class (e.g. Bot,
    # ~0.08% of the data) with fewer than 2 rows, which train_test_split's
    # stratify would reject -- fall back to an unstratified split in that
    # case, since a smoke test's split doesn't need to be representative.
    class_counts = np.bincount(train_labels_full, minlength=n_classes)
    can_stratify = class_counts[class_counts > 0].min() >= 2
    print(f"\nSplitting train into train/val ({1 - VAL_SIZE:.0%}/{VAL_SIZE:.0%})...")
    if not can_stratify:
        print("  (unstratified -- a truncated class has < 2 rows; fine for --smoke-test)")
    train_grids, val_grids, train_labels, val_labels = train_test_split(
        train_grids_full,
        train_labels_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_labels_full if can_stratify else None,
    )
    if limit_val is not None and limit_val < len(val_grids):
        val_grids = val_grids[:limit_val]
        val_labels = val_labels[:limit_val]
    print(f"  Train: {len(train_grids):,}  Val: {len(val_grids):,}")

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

    # --- Model, loss (class-weighted, matching the RF baseline's approach) ---
    print("\nBuilding ResNet18 (ImageNet-pretrained, fine-tuned)...")
    model = build_model(n_classes).to(device)

    class_weights = compute_full_class_weights(train_labels, n_classes)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"  Class weights (balanced): {dict(zip(class_names, class_weights.round(3)))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # --- Training loop with early stopping on val macro-F1 ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    log_rows = []

    print(
        f"\nTraining for up to {max_epochs} epochs "
        f"(early stopping patience={PATIENCE}, metric=val macro-F1)...\n"
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
                },
                model_path,
            )
            print(f"  -> New best val macro-F1 ({val_macro_f1:.4f}). Saved to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"\nEarly stopping: no val macro-F1 improvement for "
                    f"{PATIENCE} epochs (best={best_val_f1:.4f} at epoch "
                    f"{epoch - epochs_without_improvement})."
                )
                break

    # --- Save training log CSV + curves plot ---
    with open(training_log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to: {training_log_csv}")

    epochs_ran = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs_ran, [r["train_loss"] for r in log_rows], label="train")
    axes[0].plot(epochs_ran, [r["val_loss"] for r in log_rows], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per epoch")
    axes[0].legend()

    axes[1].plot(epochs_ran, [r["val_accuracy"] for r in log_rows], label="val accuracy")
    axes[1].plot(epochs_ran, [r["val_macro_f1"] for r in log_rows], label="val macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics per epoch")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(training_curves_png, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {training_curves_png}")

    # --- Final evaluation on the untouched test set, using the BEST checkpoint ---
    checkpoint = torch.load(model_path, map_location=device)
    print(f"\nLoading best checkpoint (epoch {checkpoint['epoch']}) for final test evaluation...")
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_true, test_pred = run_epoch(model, test_loader, criterion, device, optimizer=None)

    # labels=np.arange(n_classes) keeps the report/matrix shaped for all
    # classes even if a tiny --smoke-test subsample happens to miss one.
    all_labels = np.arange(n_classes)
    report_dict = classification_report(
        test_true, test_pred, labels=all_labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(classification_report_csv)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (test set, best checkpoint)")
    print("=" * 70)
    print(classification_report(
        test_true, test_pred, labels=all_labels, target_names=class_names, zero_division=0
    ))
    print(f"Saved to: {classification_report_csv}")

    cm = confusion_matrix(test_true, test_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("CNN (ResNet18) - Confusion Matrix (test set)")
    plt.tight_layout()
    fig.savefig(confusion_matrix_png, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {confusion_matrix_png}")

    print(f"\nBest model checkpoint: {model_path} (val macro-F1={checkpoint['val_macro_f1']:.4f}, epoch {checkpoint['epoch']})")


if __name__ == "__main__":
    main()
