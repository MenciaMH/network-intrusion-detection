"""
Cleaning pipeline for the CICIDS2017 dataset (MachineLearningCSV / MachineLearningCVE).

This script turns the 8 raw daily CSV files into a single cleaned dataset
ready for the flow-to-image encoding step. It performs, per file:

    1. Column name cleanup      -> strips leading/trailing whitespace from headers.
    2. Infinity handling        -> Flow Bytes/s and Flow Packets/s contain `Infinity`
                                    when Flow Duration == 0; these are converted to NaN.
    3. Row de-duplication       -> CICIDS2017 has a well-documented duplicate-row
                                    issue (~9% of rows). Exact duplicates are dropped
                                    to avoid train/test leakage later on.
    4. Missing value removal    -> rows with any remaining NaN are dropped (a small
                                    fraction of the dataset).
    5. Label normalization      -> some labels ship with a corrupted separator
                                    character (encoding issue in the original CIC
                                    files, e.g. "Web Attack <?> Brute Force"). Labels
                                    are matched by keyword (robust to the corrupted
                                    character) and rewritten to clean, consistent names.
    6. Rare-class handling      -> Heartbleed (11 samples) and Infiltration (36 samples)
                                    are EXCLUDED: with so few samples there is no way to
                                    build a meaningful train/test split or to evaluate a
                                    model on them, and they share no traffic pattern with
                                    any other class, so grouping them elsewhere would
                                    create an incoherent "misc" class. This exclusion is
                                    a documented limitation, not silent data loss.
                                    The three Web Attack subtypes (Brute Force, XSS,
                                    SQL Injection) ARE grouped into one "Web Attack"
                                    class: they are the same attack family at the
                                    network-flow level and combined they have enough
                                    samples (~2,180) for a reasonable split.
    7. Downcasting               -> float64 -> float32, int64 -> smallest int type that
                                    fits, to roughly halve memory usage.

After all 8 files are cleaned and concatenated into a single DataFrame, three
more checks run on the FULL dataset (these need the complete data, not a
single day, since a column might look constant/unique within one day but not
across the whole week):

    8. Constant columns          -> columns with only one distinct value carry
                                    zero information for the model and are dropped
                                    (CICIDS2017 is known to ship a few all-zero
                                    columns, e.g. some flag counters).
    9. Duplicate columns         -> columns that are exact copies of another
                                    column (same values in every row) are dropped,
                                    keeping only the first occurrence.
   10. Dtype confirmation        -> asserts that every column except Label is
                                    numeric, so nothing unexpected slipped through.
   11. Final checkpoint          -> asserts zero remaining NaNs and zero remaining
                                    infinite values before writing the output file.

The cleaned data is written out as one Parquet file (much smaller and faster
to reload than CSV).

Usage:
    python clean_data.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RAW_DATA_DIR = config.RAW_DATA_DIR
OUTPUT_DIR = config.PROCESSED_DIR
OUTPUT_PATH = config.CLEANED_DATA_PATH
LABEL_COLUMN = config.LABEL_COLUMN

# Labels excluded entirely: too few samples for a meaningful train/test split,
# and not related in traffic pattern to any other class (see module docstring).
EXCLUDED_LABELS = {"Heartbleed", "Infiltration"}

# Any label containing one of these keywords is a Web Attack subtype and gets
# grouped into a single "Web Attack" class. Matching by keyword (rather than
# the full label string) sidesteps the corrupted-separator-character issue,
# since the keyword itself is unaffected by the encoding problem.
WEB_ATTACK_KEYWORDS = ("Brute Force", "XSS", "Sql Injection", "SQL Injection")
WEB_ATTACK_LABEL = "Web Attack"


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names (CICIDS2017 headers
    have inconsistent spacing, e.g. ' Label' or 'Flow Duration ')."""
    df.columns = df.columns.str.strip()
    return df


def replace_infinities_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Flow Bytes/s and Flow Packets/s are computed as bytes/duration. When a
    flow has zero duration, this produces `Infinity`, which pandas reads as a
    literal float. Convert both +/-Infinity to NaN so they can be handled
    uniformly with other missing values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Group the Web Attack subtypes into a single class, matching by keyword
    to avoid depending on the corrupted separator character in the original
    labels (e.g. 'Web Attack \\ufffd Brute Force' -> 'Web Attack')."""
    is_web_attack = df[LABEL_COLUMN].str.contains(
        "|".join(WEB_ATTACK_KEYWORDS), case=False, na=False, regex=True
    )
    df.loc[is_web_attack, LABEL_COLUMN] = WEB_ATTACK_LABEL

    # Also strip whitespace from the remaining labels for consistency.
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip()
    return df


def drop_excluded_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows belonging to classes with too few samples to be usable
    (see EXCLUDED_LABELS above)."""
    return df[~df[LABEL_COLUMN].isin(EXCLUDED_LABELS)]


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint: float64 -> float32, int64 -> smallest int
    type that fits the data. The Label column (object/string) is untouched."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

    int_cols = df.select_dtypes(include=["int64"]).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with only one distinct value across the whole dataset.
    A constant column carries zero information for a classifier and only
    adds dead weight to the flow-to-image encoding. Must run on the full,
    concatenated dataset -- a column can look constant within a single day
    but vary across the week (or vice versa)."""
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    if constant_cols:
        print(f"  Dropping {len(constant_cols)} constant column(s): {constant_cols}")
        df = df.drop(columns=constant_cols)
    else:
        print("  No constant columns found.")
    return df


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact duplicates of another column (identical
    values in every row), keeping only the first occurrence. Columns are
    first grouped by a cheap hash to avoid an O(n_cols^2) full comparison;
    only columns that collide on the hash are compared with .equals() to
    guard against hash collisions."""
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]

    hash_to_cols: dict[int, list[str]] = {}
    for col in feature_cols:
        col_hash = int(pd.util.hash_pandas_object(df[col], index=False).sum())
        hash_to_cols.setdefault(col_hash, []).append(col)

    to_drop: list[str] = []
    for cols_with_same_hash in hash_to_cols.values():
        if len(cols_with_same_hash) < 2:
            continue
        # Confirm with an exact comparison (hash collisions are possible,
        # though rare) before dropping anything.
        kept = cols_with_same_hash[0]
        for other in cols_with_same_hash[1:]:
            if df[kept].equals(df[other]):
                to_drop.append(other)

    if to_drop:
        print(f"  Dropping {len(to_drop)} duplicate column(s): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print("  No duplicate columns found.")
    return df


def verify_dtypes(df: pd.DataFrame) -> None:
    """Confirm every column except Label is numeric. Raises if not, instead
    of silently letting a stray non-numeric column reach the model/encoding
    step."""
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            f"Expected all columns except '{LABEL_COLUMN}' to be numeric, "
            f"but found non-numeric columns: {non_numeric}"
        )
    print(f"  Dtype check passed: all {len(feature_cols)} feature columns are numeric.")


def final_checkpoint(df: pd.DataFrame) -> None:
    """Hard checkpoint before writing the output: zero NaNs, zero infinite
    values anywhere in the numeric columns. Raises instead of silently
    writing a file with lingering data quality issues."""
    n_nans = int(df.isna().sum().sum())
    numeric_df = df.select_dtypes(include=[np.number])
    n_infs = int(np.isinf(numeric_df).sum().sum())

    assert n_nans == 0, f"Checkpoint failed: {n_nans} NaN values remain."
    assert n_infs == 0, f"Checkpoint failed: {n_infs} infinite values remain."
    print("  Checkpoint passed: 0 NaNs, 0 infinite values.")


def clean_one_file(path: str) -> pd.DataFrame:
    """Run the full cleaning pipeline on a single daily CSV file."""
    df = pd.read_csv(path, low_memory=False)

    df = clean_column_names(df)
    df = replace_infinities_with_nan(df)

    rows_before = len(df)
    df = df.dropna()
    rows_after_na = len(df)

    df = df.drop_duplicates()
    rows_after_dedup = len(df)

    df = normalize_labels(df)
    df = drop_excluded_labels(df)
    rows_after_labels = len(df)

    df = downcast_dtypes(df)

    print(
        f"  {os.path.basename(path)}: "
        f"{rows_before:,} -> {rows_after_na:,} (dropna) "
        f"-> {rows_after_dedup:,} (dedup) "
        f"-> {rows_after_labels:,} (excluded labels)"
    )

    return df


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def main() -> None:
    csv_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in: {RAW_DATA_DIR}")
        print("Check that RAW_DATA_DIR points to the correct unzipped folder.")
        return

    print(f"Found {len(csv_files)} CSV files. Cleaning each one:\n")

    cleaned_frames = [clean_one_file(path) for path in csv_files]

    print("\nConcatenating all cleaned files...")
    full_df = pd.concat(cleaned_frames, ignore_index=True)
    del cleaned_frames  # free memory before writing to disk

    print("\nRunning whole-dataset checks (need the full data, not per-file):")
    full_df = drop_constant_columns(full_df)
    full_df = drop_duplicate_columns(full_df)
    verify_dtypes(full_df)
    final_checkpoint(full_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_df.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Final shape: {full_df.shape[0]:,} rows x {full_df.shape[1]} columns")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nFinal class distribution:")
    counts = full_df[LABEL_COLUMN].value_counts()
    total = len(full_df)
    for label, count in counts.items():
        print(f"  {label:20s} {count:>10,}  ({100 * count / total:5.2f}%)")


if __name__ == "__main__":
    main()
