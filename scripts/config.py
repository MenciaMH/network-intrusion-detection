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

# RAG explainability layer (script 9): grounds each CNN prediction in the
# real MITRE ATT&CK knowledge base instead of letting an LLM free-associate
# about "what a PortScan is". Retrieval-only step (7-8) needs no GPU and no
# API key; generation (Qwen2.5-3B-Instruct, local, free) benefits from GPU
# but runs on CPU too, just slower.
ATTACK_DIR = os.path.join(PROCESSED_DIR, "attack_context")
ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
ATTACK_STIX_PATH = os.path.join(ATTACK_DIR, "enterprise-attack.json")
ATTACK_TECHNIQUES_PATH = os.path.join(ATTACK_DIR, "attack_techniques.json")
ATTACK_EMBEDDINGS_PATH = os.path.join(ATTACK_DIR, "attack_technique_embeddings.npy")

RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers, CPU-friendly
RAG_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Apache-2.0, ungated, free, local
RAG_TOP_K = 3  # ATT&CK techniques retrieved per prediction
RAG_EXPLANATIONS_PATH = os.path.join(REPORTS_DIR, "rag_attack_explanations.json")

# Maps each of the CNN's 11 output classes to a natural-language query used
# to retrieve semantically related ATT&CK techniques -- the class label
# alone ("PortScan") retrieves worse than a short behavioral description,
# since ATT&CK technique text is written in analyst language, not CICIDS2017
# label vocabulary. BENIGN is intentionally absent: there's no attack to
# explain, so it's skipped upstream rather than given an empty explanation.
RAG_CLASS_QUERIES = {
    # Deliberately NOT "botnet ... beaconing" -- that phrasing retrieved
    # ATT&CK's "Botnet" technique (T1583.005/T1584.005), which is filed
    # under resource-development (an adversary acquiring/renting botnet
    # infrastructure BEFORE an attack), not an already-compromised host
    # talking to its controller, which is what this CICIDS2017 class
    # actually captures. Rephrased toward command-and-control-tactic
    # language instead.
    "Bot": "command and control beaconing malicious network communication compromised host",
    "DDoS": "distributed denial of service network flooding endpoint impact availability",
    "DoS GoldenEye": "denial of service http flood application layer attack availability",
    "DoS Hulk": "denial of service http flood application layer attack availability",
    "DoS Slowhttptest": "slow denial of service low bandwidth application layer attack",
    "DoS slowloris": "slow denial of service low bandwidth application layer attack",
    "FTP-Patator": "brute force credential guessing ftp login attack",
    "PortScan": "network service scanning discovery reconnaissance",
    "SSH-Patator": "brute force credential guessing ssh login attack",
    "Web Attack": "web application attack exploit public facing application injection",
}

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

LABEL_COLUMN = "Label"
RANDOM_STATE = 42  # used everywhere a split/sample/model needs a fixed seed
