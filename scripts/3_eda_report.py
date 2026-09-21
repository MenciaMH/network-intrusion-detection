"""
Exploratory data analysis (EDA) on the cleaned CICIDS2017 dataset.

Run this AFTER clean_data.py (needs the updated cicids2017_clean.parquet,
with constant/duplicate columns already removed) and BEFORE training any
model. The goal is to build trust in the data before spending compute on a
baseline, and to produce material for the written report.

Produces, in data/processed/eda/:
    - class_distribution.png   bar chart of class counts (log scale y-axis,
                                given the imbalance)
    - describe.csv             df.describe() for every numeric feature
    - correlation_heatmap.png  feature-to-feature correlation matrix

Also prints to the console:
    - flags for suspicious features (e.g. a feature that is ~always 0,
      even though drop_constant_columns() should already have caught a
      literal constant -- this catches near-constant features too)
    - the most highly correlated feature pairs (candidates for removal in
      a later feature-selection pass)
    - a draft summary paragraph to start from for the written report
      (rewrite it in your own words once you've looked at the plots --
      this is scaffolding, not a finished summary)

Usage:
    python eda_report.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROCESSED_DIR = config.PROCESSED_DIR
INPUT_PATH = config.CLEANED_DATA_PATH

EDA_DIR = config.EDA_DIR
CLASS_DIST_PLOT = os.path.join(EDA_DIR, "class_distribution.png")
DESCRIBE_CSV = os.path.join(EDA_DIR, "describe.csv")
CORR_HEATMAP_PLOT = os.path.join(EDA_DIR, "correlation_heatmap.png")

LABEL_COLUMN = config.LABEL_COLUMN

# A feature where one value covers this much of the data isn't a hard
# constant (already handled in clean_data.py) but is worth flagging as
# "almost always the same value" -- e.g. a flag that's 0 in 99.9% of flows.
NEAR_CONSTANT_THRESHOLD = 0.99

# Feature pairs with |correlation| above this are flagged as redundancy
# candidates for later feature selection.
HIGH_CORRELATION_THRESHOLD = 0.95


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMN].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Number of flows (log scale)")
    ax.set_xlabel("Class")
    ax.set_title("CICIDS2017 (cleaned) - Class distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(CLASS_DIST_PLOT, dpi=150)
    plt.close(fig)

    print(f"  Saved: {CLASS_DIST_PLOT}")

    majority = counts.iloc[0]
    minority = counts.iloc[-1]
    print(
        f"  Imbalance ratio (largest / smallest class): "
        f"{majority:,} / {minority:,} = {majority / minority:,.1f}x"
    )


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    desc = df[feature_cols].describe().T
    desc.to_csv(DESCRIBE_CSV)
    print(f"  Saved: {DESCRIBE_CSV}")

    # Flag near-constant features: one value covers >= threshold of all rows.
    print(f"\n  Near-constant features (one value in >={NEAR_CONSTANT_THRESHOLD:.0%} of rows):")
    n_flagged = 0
    for col in feature_cols:
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        if top_freq >= NEAR_CONSTANT_THRESHOLD:
            most_common_value = df[col].value_counts().index[0]
            print(f"    {col}: {top_freq:.2%} of rows are {most_common_value}")
            n_flagged += 1
    if n_flagged == 0:
        print("    None found.")

    return desc


def plot_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_cols, fontsize=6)
    ax.set_title("Feature correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(CORR_HEATMAP_PLOT, dpi=150)
    plt.close(fig)

    print(f"\n  Saved: {CORR_HEATMAP_PLOT}")
    return corr


def report_high_correlation_pairs(corr: pd.DataFrame) -> None:
    print(f"\n  Feature pairs with |correlation| >= {HIGH_CORRELATION_THRESHOLD}:")
    seen = set()
    n_pairs = 0
    for col_a in corr.columns:
        for col_b in corr.columns:
            if col_a == col_b or (col_b, col_a) in seen:
                continue
            seen.add((col_a, col_b))
            value = corr.loc[col_a, col_b]
            if abs(value) >= HIGH_CORRELATION_THRESHOLD:
                print(f"    {col_a}  <->  {col_b}: {value:.3f}")
                n_pairs += 1
    if n_pairs == 0:
        print("    None found.")
    else:
        print(
            f"\n  ({n_pairs} highly correlated pair(s) found -- candidates for "
            "removal in a later feature-selection pass, e.g. Phase 4. Not "
            "acted on here, just flagged.)"
        )


def print_draft_summary(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMN].value_counts()
    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    ratio = counts.iloc[0] / counts.iloc[-1]
    n_features = len([c for c in df.columns if c != LABEL_COLUMN])

    n_high_corr = 0
    seen = set()
    for col_a in corr.columns:
        for col_b in corr.columns:
            if col_a == col_b or (col_b, col_a) in seen:
                continue
            seen.add((col_a, col_b))
            if abs(corr.loc[col_a, col_b]) >= HIGH_CORRELATION_THRESHOLD:
                n_high_corr += 1

    print("\n" + "=" * 60)
    print("DRAFT SUMMARY (rewrite in your own words for the report)")
    print("=" * 60)
    print(
        f"The cleaned dataset has {len(df):,} flows across {df[LABEL_COLUMN].nunique()} "
        f"classes and {n_features} numeric features. Class imbalance remains severe: "
        f"{majority_class} accounts for {counts.iloc[0] / len(df):.1%} of the data, while "
        f"{minority_class}, the smallest class, has only {counts.iloc[-1]:,} samples -- "
        f"a {ratio:,.0f}x ratio between the largest and smallest class. "
        f"The correlation analysis found {n_high_corr} feature pair(s) with |correlation| "
        f"above {HIGH_CORRELATION_THRESHOLD}, suggesting some redundancy among the flow "
        f"features that could be addressed with feature selection later in the project. "
        f"[Add here anything specific you noticed in describe.csv or the heatmap plot -- "
        f"e.g. a feature with a surprisingly large max value, or a cluster of related "
        f"byte/packet-count features that move together.]"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"Cleaned dataset not found at: {INPUT_PATH}")
        print("Run clean_data.py first.")
        return

    os.makedirs(EDA_DIR, exist_ok=True)

    print(f"Loading cleaned dataset from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns\n")

    print("1. Class distribution")
    plot_class_distribution(df)

    print("\n2. Feature statistics (describe)")
    describe_features(df)

    print("\n3. Correlation heatmap")
    corr = plot_correlation_heatmap(df)
    report_high_correlation_pairs(corr)

    print_draft_summary(df, corr)


if __name__ == "__main__":
    main()
