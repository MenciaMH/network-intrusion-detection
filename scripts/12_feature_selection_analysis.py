"""
Feature-selection analysis informed by the correlation findings in the Notion
Work log's Section 8 (24 feature pairs with |correlation| >= 0.95, confirmed
visually by the correlation heatmap in clear structural blocks -- families of
features computed from the same underlying quantity).

This is a READ-ONLY analysis script. It does not touch any pipeline
artifact, does not re-run the flow-to-image encoding, and does not retrain
anything -- per the project decision (documented in the Work log) to treat
this as a justified analysis for the report rather than a full re-pipeline
given the time remaining in the course.

Method:
    1. Compute the full correlation matrix on train_undersampled's 63
       numeric features (the same data RF/CNN train on -- consistent with
       Sections 11/13 of the Work log).
    2. Build a graph: an edge between two features if |corr| >= 0.95.
    3. Find connected components (clusters). This correctly handles chains
       (A-B correlated, B-C correlated, A-C not directly checked but in the
       same family) instead of only looking at pairs in isolation, which
       could under- or over-drop features from a family of 3+ redundant
       columns.
    4. Within each cluster, keep the ONE feature with the highest Random
       Forest feature_importances_ (Section 11's trained model) -- this
       keeps the decision consistent with the project's existing pattern of
       tying feature-level decisions to the RF baseline (already used for
       the flow-to-image feature ordering in Section 13). Drop the rest.
    5. Features not in any cluster (correlation < 0.95 with every other
       feature) are kept automatically.

Pipeline position:
    data/processed/train_undersampled.parquet, models/rf_baseline.joblib
    -> [THIS SCRIPT, read-only] -> reports/feature_selection_drop_report.csv,
       reports/feature_selection_keep_list.txt,
       reports/feature_selection_drop_list.txt

Usage:
    python 12_feature_selection_analysis.py
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import config

TRAIN_PATH = config.TRAIN_UNDERSAMPLED_PATH
RF_MODEL_PATH = config.RF_MODEL_PATH
REPORTS_DIR = config.REPORTS_DIR
LABEL_COLUMN = config.LABEL_COLUMN

CORR_THRESHOLD = 0.95  # matches Work log Section 8's reported threshold

DROP_REPORT_CSV = os.path.join(REPORTS_DIR, "feature_selection_drop_report.csv")
KEEP_LIST_TXT = os.path.join(REPORTS_DIR, "feature_selection_keep_list.txt")
DROP_LIST_TXT = os.path.join(REPORTS_DIR, "feature_selection_drop_list.txt")


def main() -> None:
    for path in (TRAIN_PATH, RF_MODEL_PATH):
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            print("Run 5_undersample_train.py and 6_train_baseline_rf.py first.")
            return

    print(f"Loading {TRAIN_PATH}...")
    df = pd.read_parquet(TRAIN_PATH)
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    print(f"  {df.shape[0]:,} rows, {len(feature_cols)} numeric features")

    print(f"\nLoading RF baseline model from {RF_MODEL_PATH}...")
    rf = joblib.load(RF_MODEL_PATH)
    rf_feature_names = (
        list(rf.feature_names_in_) if hasattr(rf, "feature_names_in_") else feature_cols
    )
    importances = dict(zip(rf_feature_names, rf.feature_importances_))
    missing = [f for f in feature_cols if f not in importances]
    if missing:
        print(f"  WARNING: {len(missing)} features not in RF importances, treated as importance 0: {missing}")
        for f in missing:
            importances[f] = 0.0

    print(f"\nComputing correlation matrix on {len(feature_cols)} features...")
    corr = df[feature_cols].corr().abs()

    # --- Build graph (edge if |corr| >= threshold), find connected components ---
    n = len(feature_cols)
    idx = {f: i for i, f in enumerate(feature_cols)}
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pairs_found = []
    corr_vals = corr.values
    for i in range(n):
        for j in range(i + 1, n):
            if corr_vals[i, j] >= CORR_THRESHOLD:
                pairs_found.append((feature_cols[i], feature_cols[j], corr_vals[i, j]))
                union(i, j)

    print(
        f"\n{len(pairs_found)} feature pairs with |corr| >= {CORR_THRESHOLD} "
        f"(Work log Section 8 reports 24 -- sanity-check this matches)"
    )

    clusters: dict[int, list[str]] = {}
    for f in feature_cols:
        root = find(idx[f])
        clusters.setdefault(root, []).append(f)

    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"{len(multi_clusters)} correlated clusters (connected components of size > 1)")

    keep_features: list[str] = []
    drop_features: list[str] = []
    drop_report_rows = []

    for root, members in clusters.items():
        if len(members) == 1:
            keep_features.append(members[0])
            continue
        members_sorted = sorted(members, key=lambda f: importances[f], reverse=True)
        kept = members_sorted[0]
        keep_features.append(kept)
        for dropped in members_sorted[1:]:
            drop_features.append(dropped)
            drop_report_rows.append(
                {
                    "cluster_kept_feature": kept,
                    "cluster_kept_importance": importances[kept],
                    "dropped_feature": dropped,
                    "dropped_importance": importances[dropped],
                    "cluster_size": len(members),
                }
            )

    print("\n=== RESULT ===")
    print(f"Original features: {len(feature_cols)}")
    print(f"Features to DROP:  {len(drop_features)}")
    print(f"Features to KEEP:  {len(keep_features)}")

    new_grid_size = int(np.ceil(np.sqrt(len(keep_features))))
    print(
        f"\nNew grid_size if re-encoded: ceil(sqrt({len(keep_features)})) = {new_grid_size} "
        f"({new_grid_size ** 2} cells, {new_grid_size ** 2 - len(keep_features)} padding)"
    )
    print("(Current pipeline: grid_size=8, 63 features, 1 padding -- Section 13)")

    print("\n=== Clusters and drop decisions ===")
    for root, members in sorted(multi_clusters.items(), key=lambda kv: -len(kv[1])):
        members_sorted = sorted(members, key=lambda f: importances[f], reverse=True)
        print(
            f"\nCluster ({len(members)} features), keeping '{members_sorted[0]}' "
            f"(importance={importances[members_sorted[0]]:.4f}):"
        )
        for m in members_sorted:
            tag = "KEEP" if m == members_sorted[0] else "DROP"
            print(f"  [{tag}] {m}  (importance={importances[m]:.4f})")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    drop_df = pd.DataFrame(drop_report_rows).sort_values(
        ["cluster_size", "cluster_kept_feature"], ascending=[False, True]
    )
    drop_df.to_csv(DROP_REPORT_CSV, index=False)
    with open(KEEP_LIST_TXT, "w") as f:
        f.write("\n".join(sorted(keep_features)))
    with open(DROP_LIST_TXT, "w") as f:
        f.write("\n".join(sorted(drop_features)))

    print(f"\nSaved:\n  {DROP_REPORT_CSV}\n  {KEEP_LIST_TXT}\n  {DROP_LIST_TXT}")


if __name__ == "__main__":
    main()
