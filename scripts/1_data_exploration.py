"""
Initial exploration of the CICIDS2017 dataset (MachineLearningCSV).

Designed for machines with limited RAM (16GB): processes one CSV at a time,
reduces memory usage via dtype downcasting, and only keeps a summary in
memory (not the full DataFrame for all 8 days at once).

Usage:
    python 1_data_exploration.py
"""

import glob
import os

import numpy as np
import pandas as pd

import config

DATA_DIR = config.RAW_DATA_DIR
LABEL_COLUMN = config.LABEL_COLUMN


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage: float64 -> float32, int64 -> smallest int type
    that fits."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CICIDS2017 has column names with leading/trailing whitespace."""
    df.columns = df.columns.str.strip()
    return df


def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)

    # CICIDS2017 CSVs have Infinity and NaN values in ratio columns
    # (e.g. Flow Bytes/s, Flow Packets/s). Replace them with NaN so they
    # can be handled consistently later on.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = downcast_dtypes(df)
    return df


def summarize(df: pd.DataFrame, name: str) -> dict:
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    n_nulls = df.isna().sum().sum()
    n_dupes = df.duplicated().sum()

    label_col = LABEL_COLUMN if LABEL_COLUMN in df.columns else None
    class_counts = df[label_col].value_counts() if label_col else None

    print(f"\n=== {name} ===")
    print(f"Rows: {n_rows:,}  Columns: {n_cols}  Memory: {mem_mb:.1f} MB")
    print(f"Total null values: {n_nulls:,}  Duplicate rows: {n_dupes:,}")
    if class_counts is not None:
        print("Class distribution (Label):")
        print(class_counts.to_string())

    return {
        "file": name,
        "rows": n_rows,
        "cols": n_cols,
        "memory_mb": mem_mb,
        "nulls": n_nulls,
        "duplicates": n_dupes,
        "classes": class_counts.to_dict() if class_counts is not None else {},
    }


def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in: {DATA_DIR}")
        print("Check that DATA_DIR points to the correct folder.")
        return

    print(f"Found {len(csv_files)} CSV files.\n")

    all_summaries = []
    all_dtypes_seen = set()

    for path in csv_files:
        name = os.path.basename(path)
        df = load_one_csv(path)

        summary = summarize(df, name)
        all_summaries.append(summary)
        all_dtypes_seen.update(df.dtypes.astype(str).unique())

        # Explicitly free memory before moving on to the next file.
        del df

    # --- Global summary ---
    total_rows = sum(s["rows"] for s in all_summaries)
    total_dupes = sum(s["duplicates"] for s in all_summaries)

    # Merge class counts across all 8 days
    global_classes: dict[str, int] = {}
    for s in all_summaries:
        for label, count in s["classes"].items():
            global_classes[label] = global_classes.get(label, 0) + count

    print("\n" + "=" * 50)
    print("GLOBAL SUMMARY")
    print("=" * 50)
    print(f"Total rows (all 8 days): {total_rows:,}")
    print(f"Total duplicate rows: {total_dupes:,}")
    print(f"Dtypes seen: {sorted(all_dtypes_seen)}")
    print("\nGlobal class distribution (Label):")
    for label, count in sorted(global_classes.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_rows
        print(f"  {label:35s} {count:>10,}  ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
