# BDAPP — Network Intrusion Detection via Flow-to-Image Encoding

Course project for **BDAPP (Big Data Analysis and Prediction Project)**, Sejong University — AI Convergence College, Dept. of AI Data Science.

Supervised classification of known network attack categories from the CIC-IDS2017 dataset, using a flow-to-image encoding + CNN, with Grad-CAM explainability. A Random Forest baseline is used as a feasibility check before the deep learning stage.

> **Scope note:** this is supervised classification of *known* attack categories. It is explicitly **not** zero-day / anomaly detection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity, UNB) — `MachineLearningCSV` version (pre-extracted flow features via CICFlowMeter, one CSV per day/segment).

Download `MachineLearningCSV.zip` from the [CIC download page](https://cicresearch.ca/CICDataset/CIC-IDS-2017/) and unzip it into:

```
data/MachineLearningCVE/
```

The 8 daily CSV files are not committed to this repository (see `.gitignore`) — several hundred MB, and the license terms are to cite the dataset, not to redistribute it.

## Pipeline

Scripts in `scripts/` are numbered in the order they must run — each one reads the output of the previous step from `data/processed/`.

| # | Script | What it does |
|---|---|---|
| 1 | `1_data_exploration.py` | First pass over the 8 raw CSVs: shape, dtypes, memory usage, null/duplicate counts, per-file class distribution. Used to decide what the cleaning step needs to fix. |
| 2 | `2_data_cleaning.py` | Cleans and merges the 8 CSVs into one dataset: strips column-name whitespace, converts `Infinity` to `NaN`, drops rows with NaN, drops duplicate rows (~9% of the raw data), fixes the corrupted Web Attack label encoding and groups the 3 subtypes into one `Web Attack` class, excludes `Heartbleed`/`Infiltration` (too few samples for a meaningful split), drops constant and duplicate columns, verifies all feature columns are numeric, and asserts zero remaining NaN/Infinity before saving. Output: `data/processed/cicids2017_clean.parquet`. |
| 3 | `3_eda_report.py` | Exploratory analysis on the cleaned data: class distribution plot (log scale), per-feature `describe()`, correlation heatmap, flags near-constant features and highly correlated feature pairs, and prints a draft summary paragraph for the written report. Output: `data/processed/eda/`. |
| 4 | `4_train_test_split.py` | Stratified 80/20 train/test split by `Label`, fixed random seed. The test set is never touched again — it keeps the real, imbalanced distribution for honest evaluation. Output: `data/processed/train.parquet`, `data/processed/test.parquet`. |
| 5 | `5_undersample_train.py` | Downsamples `BENIGN` in the training set to 350,000 rows (from ~1.7M); every other class is left untouched. Only applied to train, never to test. Output: `data/processed/train_undersampled.parquet`. |
| 6 | `6_train_baseline_rf.py` | Trains a Random Forest baseline (`class_weight='balanced'`) on the undersampled train set, evaluates per-class precision/recall/F1 and a confusion matrix on the untouched test set, and flags classes with recall < 0.80 as SMOTE candidates for the next step. Output: `models/rf_baseline.joblib`, `data/processed/reports/`. |
| 7 | `7_flow_to_image_encoding.py` | Encodes each flow into a small square grid for the CNN: pads the feature count up to the next perfect square, orders features by descending RF importance (from step 6) so neighboring grid cells are meaningfully related, fits a per-feature min-max scaler on train only, and stores compact `grid_size x grid_size` uint8 arrays (not pre-rendered 224x224 images -- the nearest-neighbor upscale to ResNet18's input size happens lazily at training time, never bilinear/bicubic, so no values are invented between features and Grad-CAM cells stay crisp). Output: `data/processed/flow_images/`. |
| 8 | `8_train_cnn.py` | Fine-tunes an ImageNet-pretrained ResNet18 on the flow-image grids (upscaled lazily per-sample via `flow_image_dataset.py`'s `FlowImageDataset`, nearest-neighbor only), with class-weighted loss matching the RF baseline's approach, a stratified 90/10 train/val split for early stopping on validation macro-F1 (test stays untouched until the final evaluation), and per-epoch CSV logging. Output: `models/cnn_resnet18_best.pt`, `data/processed/reports/cnn_*`. |

Run them in order:

```bash
python scripts/1_data_exploration.py
python scripts/2_data_cleaning.py
python scripts/3_eda_report.py
python scripts/4_train_test_split.py
python scripts/5_undersample_train.py
python scripts/6_train_baseline_rf.py
python scripts/7_flow_to_image_encoding.py
python scripts/8_train_cnn.py
```

`scripts/flow_image_dataset.py` and `scripts/config.py` are shared modules imported by the numbered scripts, not run directly.

## Class balancing strategy

Given the severe imbalance (BENIGN is ~83% of the cleaned data; the smallest class, Bot, is ~0.08%), three techniques are combined, applied **only to the training set, after the train/test split** (never to test, to keep evaluation honest; never before the split, to avoid train/test leakage):

1. **Class weights** — reweights the loss so errors on minority classes count more (`class_weight='balanced'` in both RF and, later, the CNN's loss function). No synthetic data, no data removed.
2. **Undersampling of BENIGN** — random downsampling of the majority class only, since 1.7M+ benign flows carry a lot of redundant information.
3. **SMOTE** — applied selectively to classes that still show poor recall after (1) and (2), on the tabular features (before flow-to-image encoding, since interpolating pixels of an already-encoded image has no physical meaning).

## Known dataset limitations

- `Heartbleed` (11 samples) and `Infiltration` (36 samples) are excluded from the cleaned dataset: too few samples for a meaningful train/test split, and not related in traffic pattern to any other class.
- Even after undersampling, class imbalance remains significant — model evaluation always reports per-class metrics, never accuracy alone.

## Project structure

```
BDAPP_project/
├── data/
│   ├── MachineLearningCVE/     # raw CSVs (not committed, see .gitignore)
│   └── processed/              # cleaned/split/undersampled parquet files, EDA plots, reports
├── models/                     # trained model artifacts
├── scripts/                    # numbered pipeline scripts (see table above)
├── requirements.txt
└── README.md
```
