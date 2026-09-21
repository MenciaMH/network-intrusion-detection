"""
Shared configuration for the BDAPP NIDS pipeline scripts.

Import from here instead of redefining paths/constants in every script, so
a change -- e.g. moving data/processed/ elsewhere, or changing the random
seed -- only has to happen in one place instead of six.

This file lives in scripts/, alongside the numbered pipeline scripts, and
is imported as a plain local module (`import config`) since Python adds a
script's own directory to sys.path when it's run directly.
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "MachineLearningCVE")
# Adjust this if your unzipped folder is named differently (the CIC zip is
# called "MachineLearningCSV.zip" but the folder inside it is sometimes
# named "MachineLearningCVE").

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
EDA_DIR = os.path.join(PROCESSED_DIR, "eda")
REPORTS_DIR = os.path.join(PROCESSED_DIR, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Pipeline artifact filenames. These are each written by one script and
# read by one or more later scripts, so they live here to guarantee every
# script agrees on the same filename.
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, "cicids2017_clean.parquet")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.parquet")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.parquet")
TRAIN_UNDERSAMPLED_PATH = os.path.join(PROCESSED_DIR, "train_undersampled.parquet")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_baseline.joblib")

# Flow-to-image encoding outputs. Grids are stored SMALL (grid_size x
# grid_size, uint8) rather than pre-rendered at the CNN's 224x224 input
# size -- the nearest-neighbor upscale is cheap and lossless in the sense
# that matters (no new values invented between features), so it belongs in
# the training Dataset's __getitem__, not baked into a much larger file on
# disk (grid_size^2 bytes/row vs. 224*224*3 bytes/row -- roughly 2000x more).
FLOW_IMAGE_DIR = os.path.join(PROCESSED_DIR, "flow_images")
TRAIN_GRIDS_PATH = os.path.join(FLOW_IMAGE_DIR, "train_grids.npy")
TRAIN_LABELS_PATH = os.path.join(FLOW_IMAGE_DIR, "train_labels.npy")
TEST_GRIDS_PATH = os.path.join(FLOW_IMAGE_DIR, "test_grids.npy")
TEST_LABELS_PATH = os.path.join(FLOW_IMAGE_DIR, "test_labels.npy")
ENCODING_METADATA_PATH = os.path.join(FLOW_IMAGE_DIR, "encoding_metadata.joblib")

CNN_INPUT_SIZE = 224  # ResNet18's expected input resolution

# CNN (ResNet18) training outputs.
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_resnet18_best.pt")
CNN_TRAINING_LOG_CSV = os.path.join(REPORTS_DIR, "cnn_training_log.csv")
CNN_CLASSIFICATION_REPORT_CSV = os.path.join(REPORTS_DIR, "cnn_classification_report.csv")
CNN_CONFUSION_MATRIX_PNG = os.path.join(REPORTS_DIR, "cnn_confusion_matrix.png")
CNN_TRAINING_CURVES_PNG = os.path.join(REPORTS_DIR, "cnn_training_curves.png")

# CNN hyperparameters. Kept here (not buried in the training script) so
# they're easy to find and cite in the report.
CNN_VAL_SIZE = 0.10          # fraction of train_grids held out for validation
CNN_BATCH_SIZE = 128
CNN_NUM_WORKERS = 4
CNN_LEARNING_RATE = 1e-4
CNN_WEIGHT_DECAY = 1e-4
CNN_MAX_EPOCHS = 30
CNN_EARLY_STOPPING_PATIENCE = 5  # epochs without val macro-F1 improvement

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

LABEL_COLUMN = "Label"
RANDOM_STATE = 42  # used everywhere a split/sample/model needs a fixed seed
