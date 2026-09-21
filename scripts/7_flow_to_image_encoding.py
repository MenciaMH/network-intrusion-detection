"""
Flow-to-image encoding: turns each tabular flow (a row of numeric features)
into a small square "image" that a CNN (ResNet18) can consume.

Design decisions (see project discussion / report for the full rationale):

    1. Grid size & padding
       The number of features after cleaning is not a perfect square, so
       the grid is padded up to the next perfect square with constant-zero
       "dummy" cells (grid_size = ceil(sqrt(n_features))). Padding cells
       carry no information and are flagged as such in the saved metadata,
       so a later Grad-CAM pass can ignore them.

    2. Feature ordering = Random Forest importance (descending)
       An arbitrary (e.g. original CSV column) order gives a grid with no
       real spatial relationship between neighboring cells, which limits
       what a CNN's local filters can learn and makes Grad-CAM output
       meaningless (a highlighted region wouldn't correspond to anything
       coherent). Ordering by the RF baseline's feature_importances_ is a
       simple, defensible, "documented" choice that reuses work already
       done, rather than DeepInsight-style similarity layout (left as a
       possible improvement if time allows).

    3. Normalization: per-feature min-max to [0, 255], fit on TRAIN ONLY
       Features live on wildly different scales (durations in microseconds
       vs. packet counts vs. byte counts). Fitting the scaler on train only
       and reusing it on test avoids leaking test statistics into the
       encoding.

    4. Storage: SMALL grids, not pre-rendered 224x224 images
       Each row is stored as its native grid_size x grid_size uint8 array
       (a few dozen bytes/row), not upscaled to ResNet18's 224x224 input.
       Upscaling ahead of time would balloon disk usage by ~2000x for no
       benefit. The nearest-neighbor upscale (never bilinear/bicubic --
       smooth interpolation would invent values between features that have
       no real relationship, and would blur Grad-CAM's cell boundaries) is
       meant to happen lazily, per-sample, in the CNN training script's
       Dataset.__getitem__ (see the FlowImageDataset sketch at the bottom
       of this file's docstring / the next pipeline script).

Pipeline position:
    train_undersampled.parquet, test.parquet, rf_baseline.joblib
    -> [THIS SCRIPT] -> flow_images/{train,test}_grids.npy + labels + metadata
    -> CNN (ResNet18) training script

Usage:
    python encode_flow_images.py
"""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TRAIN_PATH = config.TRAIN_UNDERSAMPLED_PATH
TEST_PATH = config.TEST_PATH
RF_MODEL_PATH = config.RF_MODEL_PATH

FLOW_IMAGE_DIR = config.FLOW_IMAGE_DIR
TRAIN_GRIDS_PATH = config.TRAIN_GRIDS_PATH
TRAIN_LABELS_PATH = config.TRAIN_LABELS_PATH
TEST_GRIDS_PATH = config.TEST_GRIDS_PATH
TEST_LABELS_PATH = config.TEST_LABELS_PATH
ENCODING_METADATA_PATH = config.ENCODING_METADATA_PATH

LABEL_COLUMN = config.LABEL_COLUMN

# Value written into padding cells after scaling (mid-range gray, not 0)
# would also be defensible, but 0 ("nothing here") is the more standard and
# more interpretable choice for a cell that carries no real feature.
PADDING_VALUE = 0


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def compute_grid_layout(n_features: int) -> tuple[int, int]:
    """Smallest square grid that fits all features, plus how many padding
    cells are needed to fill it."""
    grid_size = math.ceil(math.sqrt(n_features))
    n_padding = grid_size**2 - n_features
    return grid_size, n_padding


def get_feature_order_by_rf_importance(
    rf_model, feature_cols: list[str]
) -> list[str]:
    """Order features by descending Random Forest importance. Falls back to
    the original column order (with a warning) if the RF model can't be
    loaded or its features don't match."""
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    ordered = importances.sort_values(ascending=False).index.tolist()
    return ordered


def encode_split(
    df: pd.DataFrame,
    feature_order: list[str],
    scaler: MinMaxScaler,
    label_encoder: LabelEncoder,
    grid_size: int,
    n_padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale, order, pad and reshape one split (train or test) into
    (N, grid_size, grid_size) uint8 grids + (N,) integer label array."""
    X = df[feature_order].to_numpy(dtype=np.float32)

    # Scale with the TRAIN-fit scaler (test just transforms, never fits),
    # then map [0, 1] -> [0, 255] and clip: test values can fall slightly
    # outside the train min/max range, and clipping is simpler and more
    # honest than re-fitting on test (which would leak test statistics).
    X_scaled = scaler.transform(X)
    X_scaled = np.clip(X_scaled, 0.0, 1.0)
    X_uint8 = (X_scaled * 255).round().astype(np.uint8)

    n_rows = X_uint8.shape[0]
    if n_padding > 0:
        padding = np.full((n_rows, n_padding), PADDING_VALUE, dtype=np.uint8)
        X_uint8 = np.concatenate([X_uint8, padding], axis=1)

    grids = X_uint8.reshape(n_rows, grid_size, grid_size)

    labels = label_encoder.transform(df[LABEL_COLUMN].to_numpy())
    labels = labels.astype(np.int64)

    return grids, labels


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            print("Run split_data.py and undersample_train.py first.")
            return

    if not os.path.exists(RF_MODEL_PATH):
        print(f"RF baseline model not found: {RF_MODEL_PATH}")
        print("Run train_baseline_rf.py first (its feature_importances_ are "
              "used to order features in the grid).")
        return

    print(f"Loading train set from: {TRAIN_PATH}")
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"  {len(train_df):,} rows x {train_df.shape[1]} columns")

    print(f"Loading test set from: {TEST_PATH}")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  {len(test_df):,} rows x {test_df.shape[1]} columns")

    feature_cols = [c for c in train_df.columns if c != LABEL_COLUMN]
    n_features = len(feature_cols)

    grid_size, n_padding = compute_grid_layout(n_features)
    print(
        f"\n{n_features} features -> {grid_size}x{grid_size} grid "
        f"({grid_size**2} cells, {n_padding} padding cell(s))"
    )

    print(f"\nLoading RF baseline model from: {RF_MODEL_PATH}")
    rf_model = joblib.load(RF_MODEL_PATH)

    rf_feature_names = list(getattr(rf_model, "feature_names_in_", feature_cols))
    if set(rf_feature_names) != set(feature_cols):
        print(
            "  WARNING: RF model's features don't match train_undersampled's "
            "columns exactly -- falling back to original column order instead "
            "of RF importance order. Re-run train_baseline_rf.py if the "
            "feature set changed since it was trained."
        )
        feature_order = feature_cols
    else:
        feature_order = get_feature_order_by_rf_importance(rf_model, feature_cols)
        print("Feature order (descending RF importance), top 10:")
        for f in feature_order[:10]:
            print(f"  {f}")

    # --- Fit scaler and label encoder on TRAIN only ---
    print("\nFitting MinMaxScaler on train features only...")
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_order].to_numpy(dtype=np.float32))

    print("Fitting LabelEncoder on train labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[LABEL_COLUMN].to_numpy())
    print(f"  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    # Sanity check: every test label must be one the encoder has seen.
    unseen_in_test = set(test_df[LABEL_COLUMN].unique()) - set(label_encoder.classes_)
    if unseen_in_test:
        print(
            f"  WARNING: test set contains labels not seen in train: "
            f"{unseen_in_test}. These rows cannot be encoded and will cause "
            f"an error -- check the train/test split."
        )

    # --- Encode both splits ---
    print("\nEncoding train set...")
    train_grids, train_labels = encode_split(
        train_df, feature_order, scaler, label_encoder, grid_size, n_padding
    )
    print(f"  train_grids: {train_grids.shape} ({train_grids.nbytes / 1024**2:.1f} MB)")

    print("Encoding test set...")
    test_grids, test_labels = encode_split(
        test_df, feature_order, scaler, label_encoder, grid_size, n_padding
    )
    print(f"  test_grids: {test_grids.shape} ({test_grids.nbytes / 1024**2:.1f} MB)")

    # --- Save ---
    os.makedirs(FLOW_IMAGE_DIR, exist_ok=True)
    np.save(TRAIN_GRIDS_PATH, train_grids)
    np.save(TRAIN_LABELS_PATH, train_labels)
    np.save(TEST_GRIDS_PATH, test_grids)
    np.save(TEST_LABELS_PATH, test_labels)

    metadata = {
        "grid_size": grid_size,
        "n_features": n_features,
        "n_padding": n_padding,
        "feature_order": feature_order,  # index i -> cell (i // grid_size, i % grid_size)
        "padding_value": PADDING_VALUE,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "cnn_input_size": config.CNN_INPUT_SIZE,
        "upscale_interpolation": "nearest",
        "notes": (
            "Grids are stored at native grid_size x grid_size resolution, "
            "uint8 in [0, 255]. Upscale to cnn_input_size with NEAREST-"
            "neighbor interpolation only (never bilinear/bicubic) at "
            "training/inference time, e.g. cv2.resize(grid, (cnn_input_size, "
            "cnn_input_size), interpolation=cv2.INTER_NEAREST) or "
            "PIL Image.resize(..., Image.NEAREST), then replicate to 3 "
            "channels for ResNet18. Cells with index >= n_features are "
            "padding and carry no feature information -- exclude them when "
            "interpreting Grad-CAM output."
        ),
    }
    joblib.dump(metadata, ENCODING_METADATA_PATH)

    print("\n" + "=" * 60)
    print("FLOW-TO-IMAGE ENCODING COMPLETE")
    print("=" * 60)
    print(f"Train grids:  {TRAIN_GRIDS_PATH}")
    print(f"Train labels: {TRAIN_LABELS_PATH}")
    print(f"Test grids:   {TEST_GRIDS_PATH}")
    print(f"Test labels:  {TEST_LABELS_PATH}")
    print(f"Metadata:     {ENCODING_METADATA_PATH}")
    print(
        f"\nOn-disk footprint: "
        f"{(train_grids.nbytes + test_grids.nbytes) / 1024**2:.1f} MB total "
        f"-- versus an estimated "
        f"{(len(train_df) + len(test_df)) * config.CNN_INPUT_SIZE**2 * 3 / 1024**3:.1f} GB "
        f"if pre-rendered at {config.CNN_INPUT_SIZE}x{config.CNN_INPUT_SIZE}x3. "
        "The CNN training script upscales lazily per-batch instead."
    )


if __name__ == "__main__":
    main()
