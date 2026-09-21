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

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

LABEL_COLUMN = "Label"
RANDOM_STATE = 42  # used everywhere a split/sample/model needs a fixed seed
