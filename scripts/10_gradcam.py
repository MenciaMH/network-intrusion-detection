"""
Grad-CAM explainability pass over the trained CNN (ResNet18) checkpoint.

This is a DIAGNOSTIC step, not a fix. Sections 15-16 of the Work log found
that BENIGN leaks into every attack class, and corrected the root cause to
BENIGN's own (very low) class weight -- but that math alone can't say WHICH
features actually drive the model to mistake a given BENIGN flow for Bot,
Web Attack, or PortScan. Grad-CAM answers that: for each misclassified
sample, it shows which grid cells (-> which original features, via
encoding_metadata.joblib's feature_order) the CNN's last conv layer weighted
most heavily when producing that specific wrong prediction. The decision to
run this BEFORE touching the CNN's class weights is documented in the Work
log, Section 16 -- this evidence is what turns the eventual fix (if any)
into a targeted one instead of blind tuning.

Method:

    1. Load the trained checkpoint and run it over the (untouched) test set
       to get real predictions -- the actual misclassified BENIGN flows are
       needed, not just the aggregate confusion-matrix counts already in
       the log.

    2. For each of the three documented confusion pairs (BENIGN->Bot,
       BENIGN->Web Attack, BENIGN->PortScan), sample up to
       GRADCAM_SAMPLES_PER_CONFUSION actual misclassified test flows.

    3. Run pytorch-grad-cam's GradCAM, hooked into model.layer4[-1] (ResNet18's
       last conv block), with the TARGET set to the (wrong) PREDICTED class --
       the question is "what made the model say Bot", not "what BENIGN
       normally looks like".

    4. Grad-CAM's output heatmap is natively 224x224 (upsampled from the
       7x7 conv feature map by the library). Each original 8x8 grid cell
       occupies an exact, non-overlapping 28x28 block of that 224x224 image
       (224 / 8 = 28 exactly, because flow_image_dataset.py upscales with
       plain nearest-neighbor replication -- see its docstring), so the
       heatmap is downsampled back to one importance score per ORIGINAL
       cell by block-averaging each 28x28 block -- never by resizing/
       interpolating, which would blur importance across cell boundaries
       that don't correspond to anything in a smoothly-interpolated image.

    5. Importance scores are averaged across the sampled misclassified
       flows per confusion pair, then mapped back to feature names and
       ranked -- plus one example heatmap overlay saved per pair for the
       report.

Pipeline position:
    models/cnn_resnet18_best.pt (real checkpoint -- see note below) +
    flow_images/test_grids.npy + encoding_metadata.joblib
    -> [THIS SCRIPT] -> reports/gradcam_feature_importance_summary.csv,
       reports/gradcam_heatmaps/*.png

Note on where this runs: the REAL trained checkpoint (epoch 28, the one
Section 15's results are based on) lives on Google Drive, since training
ran on Colab (see Work log Section 14) -- `models/cnn_resnet18_best.pt`
locally is only the tiny --smoke-test checkpoint. Use
BDAPP_gradcam_colab.ipynb (same logic as this script, loading from Drive)
unless you've already copied the real checkpoint into `models/` locally.

Usage:
    python 10_gradcam.py
    python 10_gradcam.py --samples-per-confusion 50
"""

from __future__ import annotations

import argparse
import os

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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
GRADCAM_HEATMAP_DIR = config.GRADCAM_HEATMAP_DIR
GRADCAM_SUMMARY_CSV = config.GRADCAM_SUMMARY_CSV
GRADCAM_CONFUSION_PAIRS = config.GRADCAM_CONFUSION_PAIRS
GRADCAM_TOP_N_FEATURES = config.GRADCAM_TOP_N_FEATURES
RANDOM_STATE = config.RANDOM_STATE


# --------------------------------------------------------------------------- #
# Model (identical to 8_train_cnn.py's build_model -- can't import it
# directly since Python module names can't start with a digit)
# --------------------------------------------------------------------------- #


def build_model(n_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def predict_all(model: nn.Module, grids: np.ndarray, labels: np.ndarray,
                 device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Runs the model over the full test set and returns predicted labels
    (not loss -- there's no training/early-stopping happening here)."""
    ds = FlowImageDataset(grids, labels, CNN_INPUT_SIZE)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            preds.extend(model(images).argmax(dim=1).cpu().numpy().tolist())
    return np.array(preds)


def block_average_to_grid(cam_224: np.ndarray, grid_size: int) -> np.ndarray:
    """Downsamples a (224, 224) Grad-CAM heatmap back to (grid_size,
    grid_size) by averaging each exact block -- see module docstring for
    why this must be a block-average, not a resize."""
    block = CNN_INPUT_SIZE // grid_size
    return cam_224.reshape(grid_size, block, grid_size, block).mean(axis=(1, 3))


def run_gradcam_for_pair(
    cam: GradCAM,
    model: nn.Module,
    true_name: str,
    pred_name: str,
    grids: np.ndarray,
    test_pred: np.ndarray,
    test_true: np.ndarray,
    label_encoder,
    grid_size: int,
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Finds actual test flows misclassified as true_name->pred_name,
    samples up to n_samples of them, runs Grad-CAM targeted at the WRONG
    predicted class, and returns (mean_cell_importance, n_found,
    one_example_overlay_or_None)."""
    true_idx = label_encoder.transform([true_name])[0]
    pred_idx = label_encoder.transform([pred_name])[0]

    match_indices = np.where((test_true == true_idx) & (test_pred == pred_idx))[0]
    n_found = len(match_indices)
    if n_found == 0:
        return np.zeros((grid_size, grid_size)), 0, None

    rng = np.random.RandomState(RANDOM_STATE)
    sample_indices = rng.choice(
        match_indices, size=min(n_samples, n_found), replace=False
    )

    ds = FlowImageDataset(
        grids[sample_indices],
        test_true[sample_indices],  # labels unused by Grad-CAM itself here
        CNN_INPUT_SIZE,
    )
    batch = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)

    targets = [ClassifierOutputTarget(int(pred_idx))] * len(sample_indices)
    grayscale_cam = cam(input_tensor=batch, targets=targets)  # (N, 224, 224)

    cell_importances = np.stack(
        [block_average_to_grid(c, grid_size) for c in grayscale_cam]
    )
    mean_importance = cell_importances.mean(axis=0)

    # One example overlay (first sampled flow) for the report.
    example_grid = grids[sample_indices[0]]
    example_upscaled = cv2.resize(
        example_grid, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), interpolation=cv2.INTER_NEAREST
    )
    example_rgb = np.repeat(example_upscaled[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    overlay = show_cam_on_image(example_rgb, grayscale_cam[0], use_rgb=True)

    return mean_importance, n_found, overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM diagnostic pass over the trained CNN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-per-confusion", type=int, default=config.GRADCAM_SAMPLES_PER_CONFUSION,
        help="Max misclassified test flows averaged per confusion pair.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    n_samples = args.samples_per_confusion

    for path in (TEST_GRIDS_PATH, TEST_LABELS_PATH, ENCODING_METADATA_PATH, CNN_MODEL_PATH):
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            if path == CNN_MODEL_PATH:
                print(
                    "This needs the REAL trained checkpoint (epoch 28), which lives on "
                    "Google Drive (training ran on Colab -- see Work log Section 14). "
                    "Either copy it into models/cnn_resnet18_best.pt, or run "
                    "BDAPP_gradcam_colab.ipynb instead, which loads it from Drive directly."
                )
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading encoding metadata from: {ENCODING_METADATA_PATH}")
    metadata = joblib.load(ENCODING_METADATA_PATH)
    label_encoder = metadata["label_encoder"]
    class_names = list(label_encoder.classes_)
    feature_order = metadata["feature_order"]
    grid_size = metadata["grid_size"]
    n_features = metadata["n_features"]
    print(f"  {len(class_names)} classes, grid_size={grid_size}, n_features={n_features}")

    print(f"\nLoading test grids/labels from: {TEST_GRIDS_PATH}")
    test_grids = np.load(TEST_GRIDS_PATH)
    test_labels = np.load(TEST_LABELS_PATH)
    print(f"  {test_grids.shape[0]:,} rows")

    print(f"\nLoading checkpoint from: {CNN_MODEL_PATH}")
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Checkpoint epoch {checkpoint['epoch']}, val macro-F1={checkpoint['val_macro_f1']:.4f}")

    print("\nRunning inference over the full test set (to find real misclassified flows)...")
    test_pred = predict_all(model, test_grids, test_labels, device)
    print(f"  Test accuracy: {(test_pred == test_labels).mean():.4f}")

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    os.makedirs(GRADCAM_HEATMAP_DIR, exist_ok=True)
    summary_rows = []

    for true_name, pred_name in GRADCAM_CONFUSION_PAIRS:
        print(f"\n{'=' * 70}\n{true_name} -> {pred_name}\n{'=' * 70}")
        mean_importance, n_found, overlay = run_gradcam_for_pair(
            cam, model, true_name, pred_name, test_grids, test_pred, test_labels,
            label_encoder, grid_size, n_samples, device,
        )
        if n_found == 0:
            print("  No misclassified flows found for this pair -- skipping.")
            continue
        n_used = min(n_samples, n_found)
        print(f"  {n_found} misclassified flows found in test set, averaged over {n_used}.")

        flat_importance = mean_importance.flatten()
        ranked_cells = np.argsort(-flat_importance)
        print(f"  Top {GRADCAM_TOP_N_FEATURES} driving features:")
        for rank, cell_idx in enumerate(ranked_cells[:GRADCAM_TOP_N_FEATURES], start=1):
            is_padding = cell_idx >= n_features
            feature_name = "<padding cell>" if is_padding else feature_order[cell_idx]
            importance = flat_importance[cell_idx]
            print(f"    {rank:2d}. {feature_name:<30s} importance={importance:.4f}")
            summary_rows.append({
                "confusion_pair": f"{true_name}->{pred_name}",
                "rank": rank,
                "feature": feature_name,
                "importance": importance,
                "n_misclassified_found": n_found,
                "n_samples_used": n_used,
            })

        if overlay is not None:
            overlay_path = os.path.join(
                GRADCAM_HEATMAP_DIR, f"gradcam_{true_name}_to_{pred_name}.png".replace(" ", "_")
            )
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.set_title(f"{true_name} misclassified as {pred_name}\n(example, Grad-CAM overlay)")
            ax.axis("off")
            fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Example heatmap saved to: {overlay_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(GRADCAM_SUMMARY_CSV, index=False)
    print(f"\nFull summary saved to: {GRADCAM_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
