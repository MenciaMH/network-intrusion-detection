"""
Baseline Random Forest classifier for CICIDS2017 (tabular features, no
flow-to-image encoding yet).

This is the "RF first" feasibility baseline: fast to train, no GPU needed,
and gives an early read on how separable the classes are before investing
time in the flow-to-image encoding + CNN. It also tells us, per class,
which ones still struggle after undersampling + class weights alone --
those are the candidates for SMOTE in a follow-up script.

Pipeline position:
    cleaned data -> split -> undersample BENIGN in train -> [THIS SCRIPT]
    -> (if needed) SMOTE on weak classes -> flow-to-image encoding -> CNN

Design choices:
    - Trained on train_undersampled.parquet (BENIGN capped at 350K, every
      other class at its original train count).
    - class_weight='balanced' additionally reweights the loss inside the
      forest based on the CURRENT (undersampled) class frequencies, so the
      still-small classes (Bot, Web Attack, SSH-Patator...) get more say
      per sample than BENIGN or DoS Hulk.
    - Evaluated ONLY on test.parquet, which was never touched by
      undersampling -- this is what tells you how the model would actually
      perform on real, imbalanced traffic.
    - Metrics are reported PER CLASS (precision/recall/F1), not just
      overall accuracy: with this imbalance, a model that always predicts
      BENIGN would still score ~83% accuracy while being useless.

Usage:
    python train_baseline_rf.py
"""

from __future__ import annotations

import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROCESSED_DIR = config.PROCESSED_DIR
TRAIN_PATH = config.TRAIN_UNDERSAMPLED_PATH
TEST_PATH = config.TEST_PATH

MODELS_DIR = config.MODELS_DIR
MODEL_PATH = config.RF_MODEL_PATH

REPORTS_DIR = config.REPORTS_DIR
CLASSIFICATION_REPORT_CSV = os.path.join(REPORTS_DIR, "rf_baseline_classification_report.csv")
CONFUSION_MATRIX_PNG = os.path.join(REPORTS_DIR, "rf_baseline_confusion_matrix.png")

LABEL_COLUMN = config.LABEL_COLUMN
RANDOM_STATE = config.RANDOM_STATE

# Modest defaults: enough trees to be stable, capped depth to keep training
# time reasonable on a laptop CPU. n_jobs=-1 uses all CPU cores.
N_ESTIMATORS = 200
MAX_DEPTH = 20
N_JOBS = -1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            print("Run split_data.py and undersample_train.py first.")
            return

    print(f"Loading train set from: {TRAIN_PATH}")
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"  {len(train_df):,} rows x {train_df.shape[1]} columns")

    print(f"Loading test set from: {TEST_PATH}")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  {len(test_df):,} rows x {test_df.shape[1]} columns")

    feature_cols = [c for c in train_df.columns if c != LABEL_COLUMN]

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COLUMN]
    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COLUMN]

    print(f"\nTraining RandomForestClassifier "
          f"(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, class_weight='balanced')...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
    )

    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed / 60:.1f} minutes.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    print("\nEvaluating on test set (never touched by undersampling)...")
    y_pred = clf.predict(X_test)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --- Per-class classification report ---
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(CLASSIFICATION_REPORT_CSV)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (test set)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Saved to: {CLASSIFICATION_REPORT_CSV}")

    # --- Confusion matrix ---
    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("RF Baseline - Confusion Matrix (test set)")
    plt.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PNG}")

    # --- Flag weak classes (recall below 0.80) as SMOTE candidates ---
    print("\n" + "=" * 70)
    print("CLASSES WITH RECALL < 0.80 (candidates for SMOTE next)")
    print("=" * 70)
    weak_found = False
    for label in labels_sorted:
        recall = report_dict[label]["recall"]
        support = int(report_dict[label]["support"])
        if recall < 0.80:
            print(f"  {label:20s} recall={recall:.3f}  (support in test: {support:,})")
            weak_found = True
    if not weak_found:
        print("  None -- every class has recall >= 0.80 on the test set.")

    # --- Feature importance (useful context for the later feature-selection pass) ---
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print("\nTop 10 most important features:")
    print(importances.head(10).to_string())


if __name__ == "__main__":
    main()
