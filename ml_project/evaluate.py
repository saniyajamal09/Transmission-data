"""
Shared evaluation utilities — all required metrics:
  Accuracy, Precision, Recall, F1-Score,
  Confusion Matrix, ROC Curve, AUC
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return dict with Accuracy, Precision, Recall, F1 (weighted)."""
    return {
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred,
                                           average="weighted", zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred,
                                        average="weighted", zero_division=0)),
        "f1_score" : float(f1_score(y_true, y_pred,
                                    average="weighted", zero_division=0)),
    }


def print_metrics(metrics, model_name):
    bar = "=" * 52
    print(f"\n{bar}")
    print(f"  {model_name} — Evaluation Results")
    print(bar)
    for k, v in metrics.items():
        print(f"  {k:<18s}: {v:.4f}")
    print(bar)


def print_classification_report(y_true, y_pred, label_names):
    print("\nPer-class Classification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=label_names,
                                 zero_division=0))


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names, title, save_path):
    """Plot and save a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    n  = len(label_names)
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.4, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    ax.set_title(title,              fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,             fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Confusion matrix  → {save_path}")


# ── ROC Curve + AUC ───────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_prob, label_names, title, save_path):
    """
    One-vs-Rest ROC curves for every class.
    y_prob : (n_samples, n_classes) probability array.
    Returns mean AUC across all classes.
    """
    n_classes = len(label_names)
    classes   = list(range(n_classes))

    # label_binarize needs at least 2 classes in y_true for multi-class
    present = sorted(set(y_true))

    # If only 1 class present in test set, skip ROC
    if len(present) < 2:
        print("  [ROC] Skipped — fewer than 2 classes in test set.")
        return 0.0

    # Binary case: label_binarize returns shape (n,1) — handle separately
    is_binary = (n_classes == 2)
    if is_binary:
        y_bin = np.array(y_true)
    else:
        y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap    = plt.cm.get_cmap("tab20", max(n_classes, 2))
    aucs    = []

    for i, name in enumerate(label_names):
        if i not in present:
            continue
        try:
            if is_binary:
                col = (y_bin == i).astype(int)
            else:
                col = y_bin[:, i]
            fpr, tpr, _ = roc_curve(col, y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, color=cmap(i), lw=1.5,
                    label=f"{name} (AUC={roc_auc:.2f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6, ncol=3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"  ROC curves        → {save_path}  (mean AUC = {mean_auc:.4f})")
    return mean_auc
