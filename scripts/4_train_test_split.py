"""
Stratified train/test split for the cleaned CICIDS2017 dataset.

This must run BEFORE any class-imbalance handling (undersampling, SMOTE) and
BEFORE the flow-to-image encoding step. Everything downstream (undersampling,
SMOTE, image encoding for training) operates only on the resulting train set;
the test set is left untouched with its real, imbalanced class distribution,
so evaluation reflects how the model would actually perform in the real world.

The split is stratified by `Label`, so every class keeps the same proportion
in both train and test as it has in the full cleaned dataset -- this matters
especially for the smaller classes (e.g. Bot, Web Attack), where a plain
random split could by chance leave very few (or very many) of their samples
in one of the two sets.

Usage:
    python split_data.py
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROCESSED_DIR = config.PROCESSED_DIR
INPUT_PATH = config.CLEANED_DATA_PATH
TRAIN_PATH = config.TRAIN_PATH
TEST_PATH = config.TEST_PATH

LABEL_COLUMN = config.LABEL_COLUMN
TEST_SIZE = 0.20                    # 80/20 split
RANDOM_STATE = config.RANDOM_STATE  # fixed seed for reproducibility


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
        print(f"Cleaned dataset not found at: {INPUT_PATH}")
        print("Run clean_data.py first.")
        return

    print(f"Loading cleaned dataset from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN],
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"Train saved to: {TRAIN_PATH}")
    print(f"Test saved to:  {TEST_PATH}")

    print_class_distribution(train_df, "TRAIN")
    print_class_distribution(test_df, "TEST")

    print(
        "\nNote: class proportions above should match closely between TRAIN "
        "and TEST (that's what stratification guarantees) -- this is just a "
        "sanity check, not something you need to fix."
    )


if __name__ == "__main__":
    main()
