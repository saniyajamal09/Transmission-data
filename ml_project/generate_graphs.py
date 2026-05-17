"""
Generate comparison bar charts for Accuracy and Precision
of LightGBM vs EfficientNetV2
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
models     = ["LightGBM", "EfficientNetV2"]
accuracy   = [83.97, 87.80]
precision  = [84.36, 77.10]
recall     = [83.97, 87.80]
f1_score   = [84.16, 82.10]
auc        = [83.66, 53.54]

colors_lgbm   = "#2196F3"   # blue
colors_effnet = "#FF5722"   # orange
bar_colors    = [colors_lgbm, colors_effnet]

x = np.arange(len(models))
bar_width = 0.45

# ── 1. Accuracy Bar Chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(x, accuracy, width=bar_width, color=bar_colors,
              edgecolor="white", linewidth=1.2)
ax.set_title("Model Comparison — Accuracy", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_accuracy.png"), dpi=150)
plt.close()
print("  Saved → comparison_accuracy.png")

# ── 2. Precision Bar Chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(x, precision, width=bar_width, color=bar_colors,
              edgecolor="white", linewidth=1.2)
ax.set_title("Model Comparison — Precision", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Precision (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
for bar, val in zip(bars, precision):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_precision.png"), dpi=150)
plt.close()
print("  Saved → comparison_precision.png")

# ── 3. All Metrics Grouped Bar Chart ─────────────────────────────────────────
metrics      = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
lgbm_vals    = [83.97, 84.36, 83.97, 84.16, 83.66]
effnet_vals  = [87.80, 77.10, 87.80, 82.10, 53.54]

x2 = np.arange(len(metrics))
w  = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x2 - w/2, lgbm_vals,   width=w, label="LightGBM",
            color=colors_lgbm,   edgecolor="white", linewidth=1.0)
b2 = ax.bar(x2 + w/2, effnet_vals, width=w, label="EfficientNetV2",
            color=colors_effnet, edgecolor="white", linewidth=1.0)

ax.set_title("LightGBM vs EfficientNetV2 — All Metrics", fontsize=14,
             fontweight="bold", pad=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_xticks(x2)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 105)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=11)

for bar, val in zip(b1, lgbm_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(b2, effnet_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_all_metrics.png"), dpi=150)
plt.close()
print("  Saved → comparison_all_metrics.png")

# ── 4. Fold-wise Accuracy Line Chart ─────────────────────────────────────────
folds         = [1, 2, 3, 4, 5]
lgbm_folds    = [86.71, 87.21, 84.88, 86.63, 74.42]
effnet_folds  = [87.86, 87.79, 87.79, 87.79, 87.79]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(folds, lgbm_folds,   marker="o", linewidth=2, markersize=8,
        color=colors_lgbm,   label="LightGBM")
ax.plot(folds, effnet_folds, marker="s", linewidth=2, markersize=8,
        color=colors_effnet, label="EfficientNetV2")
ax.axhline(y=83.97, color=colors_lgbm,   linestyle="--", alpha=0.5, label="LightGBM Avg")
ax.axhline(y=87.80, color=colors_effnet, linestyle="--", alpha=0.5, label="EfficientNetV2 Avg")
ax.set_title("Fold-wise Accuracy — 5-Fold Cross-Validation", fontsize=13,
             fontweight="bold", pad=12)
ax.set_xlabel("Fold Number", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xticks(folds)
ax.set_ylim(60, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "foldwise_accuracy.png"), dpi=150)
plt.close()
print("  Saved → foldwise_accuracy.png")

print("\nAll graphs saved to:", RESULTS_DIR)
