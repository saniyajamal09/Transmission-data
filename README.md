# EM S-Parameter Classification — ML Pipeline

A machine learning pipeline for classifying electromagnetic S-parameter measurements from a 7-port VNA (Vector Network Analyzer). Two models are compared: **LightGBM** on engineered numerical features, and **EfficientNetV2** on spectral images — both solving a pairwise port-group similarity task.

---

## Problem Overview

The dataset contains 42 Touchstone `.s2p` files, each recording a two-port S-parameter measurement between a pair of ports on a 7-port RF network (e.g. S12, S31, S47 …). Because every file is a unique port-pair, the task is framed as **binary similarity classification**:

> Given two S-parameter measurements, do they originate from the same transmitting port?

Port groups are defined by the transmitting port number:

| Group | Port-pairs |
|-------|-----------|
| Port 1 | S12, S13, S14, S15, S16, S17 |
| Port 2 | S21, S23, S24, S25, S26, S27 |
| Port 3 | S31, S32, S34, S35, S36, S37 |
| Port 4 | S41, S42, S43, S45, S46, S47 |
| Port 5 | S51, S52, S53, S54, S56, S57 |
| Port 6 | S61, S62, S63, S64, S65, S67 |
| Port 7 | S71, S72, S73, S74, S75, S76 |

All 42 images are combined into **861 pairwise samples** (C(42,2)), each labelled 1 (same port group) or 0 (different). Models are evaluated using 5-fold stratified cross-validation.

---

## Project Structure

```
transmissiondata/
├── *.s2p                          # Raw Touchstone VNA measurement files (input data)
└── ml_project/
    ├── config.py                  # Paths, hyperparameters, global settings
    ├── data_loader.py             # Parses .s2p files, extracts features & generates images
    ├── lgbm_model.py              # LightGBM pairwise similarity classifier
    ├── efficientnet_model.py      # EfficientNetV2 frozen backbone + MLP classifier
    ├── evaluate.py                # Shared metrics: accuracy, precision, recall, F1, ROC/AUC
    ├── generate_graphs.py         # Standalone comparison chart generator
    ├── main.py                    # Pipeline entry point — runs both models end-to-end
    └── outputs/
        ├── images/                # Generated 4-panel spectral PNGs (one per .s2p file)
        ├── models/                # Saved model artefacts (.pkl, .pth, .npy)
        └── results/               # Confusion matrices, ROC curves, evaluation_report.json
```

---

## Models

### LightGBM (Numerical Features)

Each `.s2p` file is parsed and 163 statistical features are extracted per sample:
- Per-channel statistics: mean, std, min, max, percentiles, median, slope, peak location
- FFT magnitude statistics for dB channels
- Band-averaged values (4 frequency bands)
- Cross-parameter features: reciprocity error, S21 dynamic range, frequency span

For each pair of samples, the feature vector is the **absolute difference** between the two feature vectors. LightGBM is trained on these difference vectors with balanced class weights and early stopping.

### EfficientNetV2 (Spectral Images)

Each `.s2p` file is visualised as a **4-panel PNG** showing S11, S21, S12, and S22 magnitude (dB) vs frequency. A pretrained EfficientNetV2-S backbone (ImageNet weights, frozen) extracts a 1280-dimensional feature vector per image. For each pair, the feature is the absolute difference between the two vectors. A lightweight 3-layer MLP is trained on these pair features.

---

## Results

Evaluated with 5-fold stratified cross-validation on 861 pairwise samples.

| Metric | LightGBM | EfficientNetV2 |
|--------|----------|----------------|
| Accuracy | **83.97%** | **87.80%** |
| Precision | 84.36% | 77.10% |
| Recall | 83.97% | 87.80% |
| F1-Score | 84.16% | 82.10% |
| Mean AUC | 83.66% | 53.54% |
| Training time | 2.1s | 6.2s |

**Key observations:**
- EfficientNetV2 achieves the highest accuracy (87.8%) and recall, making it better at catching true same-group pairs.
- LightGBM shows more balanced precision/recall and substantially better AUC (83.7% vs 53.5%), indicating more reliable probability calibration.
- Both models run in seconds on CPU due to the small dataset size and frozen backbone strategy.

---

## Requirements

```
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
torch
torchvision
Pillow
joblib
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm torch torchvision Pillow joblib
```

---

## Usage

Place all `.s2p` files in the root `transmissiondata/` directory, then run:

```bash
cd ml_project
python main.py
```

This will:
1. Parse all `.s2p` files and extract numerical features
2. Generate 4-panel spectral images for each file
3. Train and evaluate LightGBM (5-fold CV)
4. Train and evaluate EfficientNetV2 (frozen backbone + MLP, 5-fold CV)
5. Save all outputs to `outputs/`

To regenerate comparison charts only (without retraining):

```bash
python generate_graphs.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/results/evaluation_report.json` | Full metrics for both models |
| `outputs/results/effnet_confusion_matrix.png` | EfficientNetV2 confusion matrix |
| `outputs/results/effnet_roc_curves.png` | EfficientNetV2 ROC curves |
| `outputs/results/lgbm_confusion_matrix.png` | LightGBM confusion matrix |
| `outputs/results/lgbm_roc_curves.png` | LightGBM ROC curves |
| `outputs/results/comparison_accuracy.png` | Accuracy bar chart |
| `outputs/results/comparison_precision.png` | Precision bar chart |
| `outputs/results/comparison_all_metrics.png` | All metrics grouped bar chart |
| `outputs/models/lgbm_model.pkl` | Trained LightGBM model |
| `outputs/models/efficientnet_mlp_head.pth` | Trained MLP classifier weights |
| `outputs/models/efficientnet_features.npy` | Extracted backbone features (42 × 1280) |
