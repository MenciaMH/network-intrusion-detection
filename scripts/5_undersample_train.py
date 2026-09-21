"""
Undersample the majority class (BENIGN) in the training set.

This script operates ONLY on train.parquet -- the test set must never be
resampled, since it needs to keep the real, imbalanced class distribution
for evaluation to be meaningful.

BENIGN currently makes up ~83% of the training data (1,717,519 rows). This
script randomly downsamples it to a fixed target size, while leaving every
other class untouched. This is a first, cheap step towards balancing the
classes -- it does not create any synthetic data, it just removes redundant
BENIGN samples (2M examples of normal traffic is far more than a model needs
to learn what "normal" looks like).

Pipeline position:
    cleaned data -> [train/test split] -> [THIS SCRIPT: undersample BENIGN
    in train] -> train with class weights -> evaluate -> SMOTE on remaining
    weak classes if needed -> flow-to-image encoding

Usage:
    python undersample_train.py
"""

from __future__ import annotations

import os

import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROCESSED_DIR = config.PROCESSED_DIR
INPUT_PATH = config.TRAIN_PATH
OUTPUT_PATH = config.TRAIN_UNDERSAMPLED_PATH

LABEL_COLUMN = config.LABEL_COLUMN
MAJORITY_CLASS = "BENIGN"
TARGET_MAJORITY_SIZE = 350_000
RANDOM_STATE = config.RANDOM_STATE  # same seed as 4_train_test_split.py, for consistency


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def print_class_distribution(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} ({len(df):,} rows):")
    counts = df[LABEL_COLUMN].value_counts()
    total = len(df)
    for label, count in counts.items():
        print(f"  {label:20s} {count:>10,}  ({100 * count / total:5.2f}%)")


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"Train set not found at: {INPUT_PATH}")
        print("Run split_data.py first.")
        return

    print(f"Loading training set from: {INPUT_PATH}")
    train_df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(train_df):,} rows x {train_df.shape[1]} columns")

    print_class_distribution(train_df, "TRAIN (before undersampling)")

    majority_mask = train_df[LABEL_COLUMN] == MAJORITY_CLASS
    majority_rows = train_df[majority_mask]
    other_rows = train_df[~majority_mask]

    current_majority_size = len(majority_rows)
    if current_majority_size <= TARGET_MAJORITY_SIZE:
        print(
            f"\n{MAJORITY_CLASS} already has {current_majority_size:,} rows, "
            f"at or below the target of {TARGET_MAJORITY_SIZE:,}. Nothing to do."
        )
        majority_sampled = majority_rows
    else:
        majority_sampled = majority_rows.sample(
            n=TARGET_MAJORITY_SIZE, random_state=RANDOM_STATE
        )

    # Concatenate and shuffle so BENIGN rows aren't all grouped together
    # (matters for any training loop that reads the file sequentially/in
    # batches without its own shuffling).
    result_df = pd.concat([majority_sampled, other_rows], ignore_index=True)
    result_df = result_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    result_df.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("UNDERSAMPLING COMPLETE")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_PATH}")
    print_class_distribution(result_df, "TRAIN (after undersampling)")

    print(
        "\nNote: only BENIGN was reduced. Every other class keeps its "
        "original count from train.parquet -- this step alone won't fully "
        "balance the dataset, it just brings the majority class down to a "
        "more reasonable size. Minority classes with poor recall after a "
        "first training run are the ones to address with SMOTE next."
    )


if __name__ == "__main__":
    main()
