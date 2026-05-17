"""
LightGBM classifier on numerical S-parameter features.

Dataset reality: 42 samples, 42 unique classes (1 per class).
This means standard train/test split is impossible for multi-class classification.

SOLUTION: We reframe the problem as BINARY SIMILARITY classification.
  - For each pair of samples, compute feature difference vector
  - Label = 1 if same port group (e.g. both from port 1: S12,S13..S17), else 0
  - Train LightGBM on these pairs → learns which S-param patterns are similar
  - Evaluate with standard metrics on held-out pairs

Port groups (from the 7-port network):
  Port 1 → S12, S13, S14, S15, S16, S17
  Port 2 → S21, S23, S24, S25, S26, S27
  Port 3 → S31, S32, S34, S35, S36, S37
  Port 4 → S41, S42, S43, S45, S46, S47
  Port 5 → S51, S52, S53, S54, S56, S57
  Port 6 → S61, S62, S63, S64, S65, S67
  Port 7 → S71, S72, S73, S74, S75, S76

This gives a meaningful binary classification task with enough samples.
"""
import os
import joblib
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from config import RANDOM_STATE, LGBM_PARAMS, MODEL_DIR, RESULTS_DIR
from evaluate import (compute_metrics, print_metrics,
                      plot_confusion_matrix, plot_roc_curves,
                      print_classification_report)


# Port group mapping
PORT_GROUPS = {
    "S12": 1, "S13": 1, "S14": 1, "S15": 1, "S16": 1, "S17": 1,
    "S21": 2, "S23": 2, "S24": 2, "S25": 2, "S26": 2, "S27": 2,
    "S31": 3, "S32": 3, "S34": 3, "S35": 3, "S36": 3, "S37": 3,
    "S41": 4, "S42": 4, "S43": 4, "S45": 4, "S46": 4, "S47": 4,
    "S51": 5, "S52": 5, "S53": 5, "S54": 5, "S56": 5, "S57": 5,
    "S61": 6, "S62": 6, "S63": 6, "S64": 6, "S65": 6, "S67": 6,
    "S71": 7, "S72": 7, "S73": 7, "S74": 7, "S75": 7, "S76": 7,
}


def _build_pair_dataset(df_num):
    """
    Build pairwise feature dataset.
    For each pair (i, j): feature = |xi - xj| (absolute difference)
    Label = 1 if same port group, 0 otherwise.
    """
    feature_cols = [c for c in df_num.columns if c not in ("label", "label_str")]
    X_raw = df_num[feature_cols].values
    labels_str = df_num["label_str"].tolist()

    pair_feats, pair_labels, pair_idx = [], [], []

    for i, j in combinations(range(len(df_num)), 2):
        diff = np.abs(X_raw[i] - X_raw[j])
        pair_feats.append(diff)

        gi = PORT_GROUPS.get(labels_str[i], -1)
        gj = PORT_GROUPS.get(labels_str[j], -1)
        same = 1 if (gi == gj and gi != -1) else 0
        pair_labels.append(same)
        pair_idx.append((i, j))

    X_pairs = np.array(pair_feats, dtype=np.float64)
    y_pairs = np.array(pair_labels, dtype=int)
    return X_pairs, y_pairs, pair_idx, feature_cols


def run_lgbm(df_num, label_map):
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[LightGBM] Building pairwise similarity dataset …")
    X_pairs, y_pairs, pair_idx, feature_cols = _build_pair_dataset(df_num)

    n_pos = y_pairs.sum()
    n_neg = len(y_pairs) - n_pos
    print(f"  Total pairs : {len(y_pairs)}  (same-group={n_pos}, diff-group={n_neg})")

    # ── 5-Fold Stratified CV ──────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_preds = np.zeros(len(y_pairs), dtype=int)
    all_probs = np.zeros((len(y_pairs), 2), dtype=np.float64)

    lgbm_params = {
        "n_estimators"    : 500,
        "learning_rate"   : 0.03,
        "max_depth"       : 5,
        "num_leaves"      : 20,
        "min_child_samples": 3,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha"       : 0.1,
        "reg_lambda"      : 0.1,
        "random_state"    : RANDOM_STATE,
        "n_jobs"          : -1,
        "verbose"         : -1,
        "class_weight"    : "balanced",
    }

    print("[LightGBM] Running 5-Fold Stratified CV on pairwise similarity …")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_pairs, y_pairs)):
        X_tr, X_te = X_pairs[tr_idx], X_pairs[te_idx]
        y_tr, y_te = y_pairs[tr_idx], y_pairs[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])

        all_preds[te_idx] = model.predict(X_te)
        all_probs[te_idx] = model.predict_proba(X_te)

        fold_acc = (all_preds[te_idx] == y_te).mean()
        print(f"  Fold {fold+1}/5  test_acc={fold_acc:.4f}  "
              f"(pos={y_te.sum()}, neg={len(y_te)-y_te.sum()})")

    # ── Metrics ───────────────────────────────────────────────────────────────
    label_names = ["Different Group", "Same Group"]
    metrics = compute_metrics(y_pairs, all_preds)
    print_metrics(metrics, "LightGBM — Port-Group Similarity (5-Fold CV)")
    print_classification_report(y_pairs, all_preds, label_names)

    plot_confusion_matrix(
        y_pairs, all_preds, label_names,
        "LightGBM — Confusion Matrix\n(Same vs Different Port Group)",
        os.path.join(RESULTS_DIR, "lgbm_confusion_matrix.png")
    )
    mean_auc = plot_roc_curves(
        y_pairs, all_probs, label_names,
        "LightGBM — ROC Curves (Same vs Different Port Group)",
        os.path.join(RESULTS_DIR, "lgbm_roc_curves.png")
    )
    metrics["mean_auc"] = mean_auc

    # ── Final model on ALL pairs ──────────────────────────────────────────────
    print("\n[LightGBM] Training final model on all pairs …")
    final_scaler = StandardScaler()
    X_all = final_scaler.fit_transform(X_pairs)
    final_model = lgb.LGBMClassifier(**lgbm_params)
    final_model.fit(X_all, y_pairs)

    joblib.dump(final_model,  os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    joblib.dump(final_scaler, os.path.join(MODEL_DIR, "lgbm_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "lgbm_feature_cols.pkl"))
    print(f"[LightGBM] Final model saved → {MODEL_DIR}")

    return metrics
