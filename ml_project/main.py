"""
EM S-Parameter Classification — Full ML Pipeline
=================================================
Models:
  1. LightGBM       — on numerical/statistical features extracted from .s2p files
  2. EfficientNetV2 — on 4-panel spectral images generated from .s2p files

Evaluation (both models):
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix  (saved as PNG)
  - ROC Curve + AUC   (saved as PNG)

Dataset: 42 Touchstone .s2p files (VNA S-parameter measurements)
Strategy: Leave-One-Out Cross-Validation (LOO-CV) — correct for 42 samples
"""
import os
import sys
import json
import time

# Ensure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config      import RESULTS_DIR, OUTPUT_DIR
from data_loader import load_dataset
from lgbm_model  import run_lgbm
from efficientnet_model import run_efficientnet


def banner(text):
    w = 62
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,  exist_ok=True)

    total_start = time.time()

    # ── STEP 1: Load & parse all .s2p files ──────────────────────────────────
    banner("STEP 1 — Loading S-parameter data & generating images")
    df_num, img_df, label_map = load_dataset(generate_images=True)

    n_samples = len(df_num)
    n_classes = len(label_map)
    print(f"\n  Samples  : {n_samples}")
    print(f"  Classes  : {n_classes}")
    print(f"  Features : {len(df_num.columns) - 2}")   # minus label + label_str

    # ── STEP 2: LightGBM ─────────────────────────────────────────────────────
    banner("STEP 2 — LightGBM on Numerical Features (LOO-CV)")
    t0 = time.time()
    lgbm_metrics = run_lgbm(df_num, label_map)
    lgbm_time    = time.time() - t0
    print(f"\n  LightGBM completed in {lgbm_time:.1f}s")

    # ── STEP 3: EfficientNetV2 ────────────────────────────────────────────────
    banner("STEP 3 — EfficientNetV2 on Spectral Images (LOO-CV)")
    t0 = time.time()
    effnet_metrics = run_efficientnet(img_df, label_map)
    effnet_time    = time.time() - t0
    print(f"\n  EfficientNetV2 completed in {effnet_time:.1f}s")

    # ── STEP 4: Save & print final report ────────────────────────────────────
    banner("FINAL EVALUATION REPORT")

    summary = {
        "dataset": {
            "n_samples"  : n_samples,
            "n_classes"  : n_classes,
            "n_features" : len(df_num.columns) - 2,
            "eval_method": "Leave-One-Out Cross-Validation",
        },
        "LightGBM": {
            **lgbm_metrics,
            "training_time_sec": round(lgbm_time, 2),
        },
        "EfficientNetV2": {
            **effnet_metrics,
            "training_time_sec": round(effnet_time, 2),
        },
    }

    report_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print comparison table
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "mean_auc"]
    col_w = 18
    print(f"\n  {'Metric':<{col_w}} {'LightGBM':>14} {'EfficientNetV2':>16}")
    print(f"  {'-'*col_w} {'-'*14} {'-'*16}")
    for k in metric_keys:
        lv = lgbm_metrics.get(k, 0.0)
        ev = effnet_metrics.get(k, 0.0)
        lv_s = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
        ev_s = f"{ev:.4f}" if isinstance(ev, float) else str(ev)
        print(f"  {k:<{col_w}} {lv_s:>14} {ev_s:>16}")

    total_time = time.time() - total_start
    print(f"\n  Total runtime : {total_time:.1f}s")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"    models/   — trained model files (.pkl, .pth)")
    print(f"    results/  — confusion matrices, ROC curves, report.json")
    print(f"    images/   — spectral PNG images used by EfficientNetV2")
    print("=" * 62)


if __name__ == "__main__":
    main()
