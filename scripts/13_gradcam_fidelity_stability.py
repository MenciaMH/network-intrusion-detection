"""
Fidelity and stability checks for the Grad-CAM feature rankings computed by
10_gradcam.py / Section 19 of the Work log.

Why this script exists:

    Section 19 (and Table VI of the report) rank which features drove each
    BENIGN->X confusion -- but a ranking by itself only shows CORRELATION
    between a feature and the wrong prediction, not that the feature is
    CAUSALLY responsible for it, and it says nothing about whether a
    different sample of misclassified flows would have produced a
    different top-10. This was left as "future work" (a deletion/insertion
    test) in Section IX of the report. Udurume, Shakhov & Koo (Sci. Rep.,
    2026) validate their SHAP rankings with exactly two checks -- feature
    masking (fidelity) and bootstrap Jaccard overlap (stability) -- and
    this script applies the same two checks to the already-computed
    Grad-CAM rankings, instead of leaving them unvalidated.

Method:

    1. FIDELITY (feature masking / "deletion test"). For each confusion
       pair, take every real test flow misclassified as true_name->pred_name
       (not just the 25 averaged in script 10 -- using the full set gives a
       more reliable flip-rate). For k in GRADCAM_FIDELITY_TOP_K_LIST, zero
       out the grid cells of the top-k features already ranked for that
       pair (read from script 10's summary CSV -- this script does not
       re-rank, it tests the existing ranking), then re-run the model on
       the masked flows and measure:
           - flip_rate: fraction whose prediction changes away from the
             wrong class pred_name
           - reclaim_rate: fraction whose prediction changes specifically
             BACK to the true class (BENIGN)
           - mean_confidence_drop: how much the softmax probability of the
             wrong class pred_name falls, on average
       A global, Udurume-et-al.-style check is also run: masking the SAME
       top-k features across the WHOLE test set and reporting the overall
       accuracy/macro-F1 drop (a smaller, broader-but-shallower signal than
       the pair-specific flip-rate above).

    2. STABILITY (bootstrap Jaccard overlap). For each confusion pair, draw
       GRADCAM_STABILITY_N_BOOTSTRAPS bootstrap resamples (with replacement,
       same size as script 10's GRADCAM_SAMPLES_PER_CONFUSION) from the real
       misclassified flows, run Grad-CAM fresh on each resample, rank the
       top GRADCAM_STABILITY_TOP_N features per resample, and report the
       mean +/- std pairwise Jaccard similarity across all resample pairs.
       A low mean Jaccard would mean Section 19's top-10 depended heavily
       on which 25 flows happened to be sampled; a high one (Udurume et al.
       report 0.84-0.89) means the ranking is a property of the confusion
       itself, not of the sample.

    IMPORTANT: this runs on the SAME checkpoint Section 19/Table VI used --
    the ORIGINAL epoch-28 checkpoint, before the benign-weight fine-tune --
    since the point is to validate THAT ranking, not re-derive a new one.

Pipeline position:
    models/cnn_resnet18_best.pt + flow_images/test_grids.npy +
    encoding_metadata.joblib + reports/gradcam_feature_importance_summary.csv
    (script 10's output)
    -> [THIS SCRIPT] -> reports/gradcam_fidelity_stability.csv

Note on where this runs: same requirement as script 10 -- needs the REAL
epoch-28 checkpoint. If you're running this locally and already copied that
checkpoint into models/cnn_resnet18_best.pt (as you did for the fine-tune),
this runs locally on CPU in a few minutes; it is inference-only, no
training. If you'd rather run it on Colab next to the checkpoint on Drive,
the logic is identical -- just swap the path constants for Drive paths the
way BDAPP_gradcam_colab.ipynb already does.

Usage:
    python 13_gradcam_fidelity_stability.py
    python 13_gradcam_fidelity_stability.py --n-bootstraps 10
"""

from __future__ import annotations

import argparse
import itertools
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import ResNet18_Weights, resnet18

import config
from flow_image_dataset import FlowImageDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TEST_GRIDS_PATH = config.TEST_GRIDS_PATH
TEST_LABELS_PATH = config.TEST_LABELS_PATH
ENCODING_METADATA_PATH = config.ENCODING_METADATA_PATH
CNN_MODEL_PATH = config.CNN_MODEL_PATH
CNN_INPUT_SIZE = config.CNN_INPUT_SIZE

REPORTS_DIR = config.REPORTS_DIR
GRADCAM_SUMMARY_CSV = config.GRADCAM_SUMMARY_CSV
GRADCAM_CONFUSION_PAIRS = config.GRADCAM_CONFUSION_PAIRS
GRADCAM_SAMPLES_PER_CONFUSION = config.GRADCAM_SAMPLES_PER_CONFUSION
RANDOM_STATE = config.RANDOM_STATE

FIDELITY_TOP_K_LIST = config.GRADCAM_FIDELITY_TOP_K_LIST
FIDELITY_STABILITY_CSV = config.GRADCAM_FIDELITY_STABILITY_CSV
N_BOOTSTRAPS = config.GRADCAM_STABILITY_N_BOOTSTRAPS
STABILITY_TOP_N = config.GRADCAM_STABILITY_TOP_N


# --------------------------------------------------------------------------- #
# Model (identical to 8_train_cnn.py / 10_gradcam.py -- module names can't
# start with a digit, so this is duplicated rather than imported)
# --------------------------------------------------------------------------- #


def build_model(n_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def block_average_to_grid(cam_224: np.ndarray, grid_size: int) -> np.ndarray:
    block = CNN_INPUT_SIZE // grid_size
    return cam_224.reshape(grid_size, block, grid_size, block).mean(axis=(1, 3))


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def predict_with_probs(
    model: nn.Module, grids: np.ndarray, labels: np.ndarray, device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_labels, softmax_probs) for a batch of grids."""
    ds = FlowImageDataset(grids, labels, CNN_INPUT_SIZE)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_probs.append(probs)
    return np.array(all_preds), np.concatenate(all_probs, axis=0)


def mask_top_k_features(
    grids: np.ndarray, feature_order: list[str], grid_size: int, top_k_feature_names: list[str],
) -> np.ndarray:
    """Returns a COPY of `grids` with the cells of `top_k_feature_names`
    zeroed out. Zeroing means the cell's scaled value is set to 0 -- the
    feature's observed minimum in the training set, not a neutral/missing
    value -- which is the same convention Udurume et al. use ("zeroing the
    top-ranked features") and is the simplest, most literal reading of a
    "deletion" test; it is flagged here as a known approximation rather than
    silently assumed to mean "no information"."""
    masked = grids.copy()
    name_to_cell = {name: idx for idx, name in enumerate(feature_order)}
    for name in top_k_feature_names:
        if name not in name_to_cell:
            continue  # e.g. "<padding cell>", never masked
        cell_idx = name_to_cell[name]
        row, col = divmod(cell_idx, grid_size)
        masked[:, row, col] = 0
    return masked


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# --------------------------------------------------------------------------- #
# Fidelity
# --------------------------------------------------------------------------- #


def run_fidelity_for_pair(
    model: nn.Module, device: torch.device, true_name: str, pred_name: str,
    match_indices: np.ndarray, test_grids: np.ndarray, test_labels: np.ndarray,
    label_encoder, feature_order: list[str], grid_size: int,
    top_k_rows: pd.DataFrame,
) -> list[dict]:
    true_idx = label_encoder.transform([true_name])[0]
    pred_idx = label_encoder.transform([pred_name])[0]

    pair_grids = test_grids[match_indices]
    pair_labels = test_labels[match_indices]

    # Baseline: confirm these flows really are predicted as pred_name right now.
    base_preds, base_probs = predict_with_probs(model, pair_grids, pair_labels, device)
    assert (base_preds == pred_idx).all(), (
        "Some 'misclassified' flows did not predict pred_name on re-check -- "
        "check that this is the same checkpoint script 10 used."
    )
    base_conf_wrong_class = base_probs[:, pred_idx].mean()

    pair_key = f"{true_name}->{pred_name}"
    ranked = (
        top_k_rows[top_k_rows["confusion_pair"] == pair_key]
        .sort_values("rank")["feature"].tolist()
    )

    rows = []
    for k in FIDELITY_TOP_K_LIST:
        top_k_names = ranked[:k]
        masked_grids = mask_top_k_features(pair_grids, feature_order, grid_size, top_k_names)
        masked_preds, masked_probs = predict_with_probs(model, masked_grids, pair_labels, device)

        flipped_away = masked_preds != pred_idx
        reclaimed = masked_preds == true_idx
        conf_wrong_after = masked_probs[:, pred_idx].mean()

        rows.append({
            "check": "fidelity_pair_specific",
            "confusion_pair": pair_key,
            "top_k": k,
            "masked_features": "; ".join(top_k_names),
            "n_flows": len(match_indices),
            "flip_rate": flipped_away.mean(),
            "reclaim_rate": reclaimed.mean(),
            "mean_confidence_wrong_class_before": base_conf_wrong_class,
            "mean_confidence_wrong_class_after": conf_wrong_after,
            "confidence_drop": base_conf_wrong_class - conf_wrong_after,
        })
    return rows


def run_global_fidelity(
    model: nn.Module, device: torch.device, test_grids: np.ndarray, test_labels: np.ndarray,
    feature_order: list[str], grid_size: int, top_k_rows: pd.DataFrame,
) -> list[dict]:
    """Udurume-et-al.-style global check: mask the same top-k features
    ACROSS THE WHOLE TEST SET (not just the misclassified pair) and report
    the overall accuracy/macro-F1 drop. Broader but shallower than the
    pair-specific flip-rate above -- included for direct comparability with
    their methodology."""
    base_preds, _ = predict_with_probs(model, test_grids, test_labels, device)
    base_acc = accuracy_score(test_labels, base_preds)
    base_macro_f1 = f1_score(test_labels, base_preds, average="macro")

    rows = []
    for true_name, pred_name in GRADCAM_CONFUSION_PAIRS:
        pair_key = f"{true_name}->{pred_name}"
        ranked = (
            top_k_rows[top_k_rows["confusion_pair"] == pair_key]
            .sort_values("rank")["feature"].tolist()
        )
        for k in FIDELITY_TOP_K_LIST:
            top_k_names = ranked[:k]
            masked_grids = mask_top_k_features(test_grids, feature_order, grid_size, top_k_names)
            masked_preds, _ = predict_with_probs(model, masked_grids, test_labels, device)
            masked_acc = accuracy_score(test_labels, masked_preds)
            masked_macro_f1 = f1_score(test_labels, masked_preds, average="macro")

            rows.append({
                "check": "fidelity_global",
                "confusion_pair": pair_key,
                "top_k": k,
                "masked_features": "; ".join(top_k_names),
                "n_flows": len(test_labels),
                "accuracy_before": base_acc,
                "accuracy_after": masked_acc,
                "macro_f1_before": base_macro_f1,
                "macro_f1_after": masked_macro_f1,
                "macro_f1_drop": base_macro_f1 - masked_macro_f1,
            })
    return rows


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #


def run_stability_for_pair(
    cam: GradCAM, model: nn.Module, device: torch.device, true_name: str, pred_name: str,
    match_indices: np.ndarray, test_grids: np.ndarray, test_labels: np.ndarray,
    label_encoder, feature_order: list[str], grid_size: int,
) -> list[dict]:
    pred_idx = label_encoder.transform([pred_name])[0]
    n_sample = min(GRADCAM_SAMPLES_PER_CONFUSION, len(match_indices))

    rng = np.random.RandomState(RANDOM_STATE)
    top_sets = []
    for b in range(N_BOOTSTRAPS):
        boot_indices = rng.choice(match_indices, size=n_sample, replace=True)
        ds = FlowImageDataset(
            test_grids[boot_indices], test_labels[boot_indices], CNN_INPUT_SIZE,
        )
        batch = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
        targets = [ClassifierOutputTarget(int(pred_idx))] * len(boot_indices)
        grayscale_cam = cam(input_tensor=batch, targets=targets)

        cell_importances = np.stack(
            [block_average_to_grid(c, grid_size) for c in grayscale_cam]
        )
        mean_importance = cell_importances.mean(axis=0).flatten()
        ranked_cells = np.argsort(-mean_importance)[:STABILITY_TOP_N]
        top_names = {
            feature_order[c] for c in ranked_cells if c < len(feature_order)
        }
        top_sets.append(top_names)

    pairwise_jaccards = [
        jaccard(a, b) for a, b in itertools.combinations(top_sets, 2)
    ]
    return [{
        "check": "stability",
        "confusion_pair": f"{true_name}->{pred_name}",
        "n_bootstraps": N_BOOTSTRAPS,
        "bootstrap_sample_size": n_sample,
        "top_n_compared": STABILITY_TOP_N,
        "jaccard_mean": float(np.mean(pairwise_jaccards)),
        "jaccard_std": float(np.std(pairwise_jaccards)),
        "jaccard_min": float(np.min(pairwise_jaccards)),
        "jaccard_max": float(np.max(pairwise_jaccards)),
    }]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fidelity + stability checks for the Grad-CAM feature rankings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global N_BOOTSTRAPS
    N_BOOTSTRAPS = args.n_bootstraps

    required = (TEST_GRIDS_PATH, TEST_LABELS_PATH, ENCODING_METADATA_PATH,
                CNN_MODEL_PATH, GRADCAM_SUMMARY_CSV)
    for path in required:
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            if path == CNN_MODEL_PATH:
                print(
                    "Needs the REAL epoch-28 checkpoint (same one script 10 used), "
                    "not the smoke-test checkpoint."
                )
            if path == GRADCAM_SUMMARY_CSV:
                print("Run 10_gradcam.py first -- this script reuses its feature ranking.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata = joblib.load(ENCODING_METADATA_PATH)
    label_encoder = metadata["label_encoder"]
    class_names = list(label_encoder.classes_)
    feature_order = metadata["feature_order"]
    grid_size = metadata["grid_size"]

    test_grids = np.load(TEST_GRIDS_PATH)
    test_labels = np.load(TEST_LABELS_PATH)
    top_k_rows = pd.read_csv(GRADCAM_SUMMARY_CSV)

    print(f"Loading checkpoint from: {CNN_MODEL_PATH}")
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Checkpoint epoch {checkpoint['epoch']}, val macro-F1={checkpoint['val_macro_f1']:.4f}")
    print(
        "  NOTE: this must be the SAME checkpoint script 10 used to build Table VI -- "
        "this script validates that ranking, it does not recompute it from scratch."
    )

    print("\nRunning inference over the full test set (to find real misclassified flows)...")
    test_pred, _ = predict_with_probs(model, test_grids, test_labels, device)
    print(f"  Test accuracy: {(test_pred == test_labels).mean():.4f}")

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    all_rows: list[dict] = []

    for true_name, pred_name in GRADCAM_CONFUSION_PAIRS:
        true_idx = label_encoder.transform([true_name])[0]
        pred_idx = label_encoder.transform([pred_name])[0]
        match_indices = np.where((test_labels == true_idx) & (test_pred == pred_idx))[0]

        print(f"\n{'=' * 70}\n{true_name} -> {pred_name}  ({len(match_indices)} misclassified flows)\n{'=' * 70}")
        if len(match_indices) == 0:
            print("  No misclassified flows found -- skipping.")
            continue

        print("  Fidelity (feature masking)...")
        fidelity_rows = run_fidelity_for_pair(
            model, device, true_name, pred_name, match_indices, test_grids, test_labels,
            label_encoder, feature_order, grid_size, top_k_rows,
        )
        for r in fidelity_rows:
            print(
                f"    top_k={r['top_k']:>2d}  flip_rate={r['flip_rate']:.3f}  "
                f"reclaim_rate={r['reclaim_rate']:.3f}  "
                f"confidence_drop={r['confidence_drop']:.3f}  "
                f"(masked: {r['masked_features']})"
            )
        all_rows.extend(fidelity_rows)

        print("  Stability (bootstrap Jaccard overlap)...")
        stability_rows = run_stability_for_pair(
            cam, model, device, true_name, pred_name, match_indices, test_grids, test_labels,
            label_encoder, feature_order, grid_size,
        )
        for r in stability_rows:
            print(
                f"    Jaccard top-{r['top_n_compared']} over {r['n_bootstraps']} bootstraps: "
                f"mean={r['jaccard_mean']:.3f} +/- {r['jaccard_std']:.3f} "
                f"(min={r['jaccard_min']:.3f}, max={r['jaccard_max']:.3f})"
            )
        all_rows.extend(stability_rows)

    print("\nGlobal fidelity (whole test set, Udurume-et-al.-style)...")
    global_rows = run_global_fidelity(
        model, device, test_grids, test_labels, feature_order, grid_size, top_k_rows,
    )
    for r in global_rows:
        print(
            f"  {r['confusion_pair']}  top_k={r['top_k']:>2d}  "
            f"macro_f1 {r['macro_f1_before']:.4f} -> {r['macro_f1_after']:.4f} "
            f"(drop={r['macro_f1_drop']:.4f})"
        )
    all_rows.extend(global_rows)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(FIDELITY_STABILITY_CSV, index=False)
    print(f"\nFull results saved to: {FIDELITY_STABILITY_CSV}")


if __name__ == "__main__":
    main()
