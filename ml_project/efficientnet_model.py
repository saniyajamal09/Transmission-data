"""
EfficientNetV2-S on spectral images — FAST version.

Strategy: Extract features ONCE using frozen pretrained backbone,
then train a lightweight classifier on the feature pairs.
This runs in ~30 seconds on CPU instead of 15+ minutes.
"""
import os
import numpy as np
from itertools import combinations

from config import (IMG_SIZE, RANDOM_STATE, MODEL_DIR, RESULTS_DIR, DROPOUT_RATE)
from evaluate import (compute_metrics, print_metrics,
                      plot_confusion_matrix, plot_roc_curves,
                      print_classification_report)

PORT_GROUPS = {
    "S12": 1, "S13": 1, "S14": 1, "S15": 1, "S16": 1, "S17": 1,
    "S21": 2, "S23": 2, "S24": 2, "S25": 2, "S26": 2, "S27": 2,
    "S31": 3, "S32": 3, "S34": 3, "S35": 3, "S36": 3, "S37": 3,
    "S41": 4, "S42": 4, "S43": 4, "S45": 4, "S46": 4, "S47": 4,
    "S51": 5, "S52": 5, "S53": 5, "S54": 5, "S56": 5, "S57": 5,
    "S61": 6, "S62": 6, "S63": 6, "S64": 6, "S65": 6, "S67": 6,
    "S71": 7, "S72": 7, "S73": 7, "S74": 7, "S75": 7, "S76": 7,
}

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Step 1: Extract deep features from all 42 images (done ONCE) ─────────────

def extract_all_features(img_df, device):
    """
    Run all 42 images through frozen EfficientNetV2-S backbone.
    Returns feature matrix of shape (42, 1280).
    """
    weights  = EfficientNet_V2_S_Weights.DEFAULT
    backbone = efficientnet_v2_s(weights=weights)
    # Remove final classifier — keep feature extractor + adaptive pool
    encoder  = nn.Sequential(*list(backbone.children())[:-1])
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats = []
    paths = img_df["image_path"].tolist()
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x   = tf(img).unsqueeze(0).to(device)
            f   = encoder(x).squeeze()   # shape (1280,)
            feats.append(f.cpu().numpy())

    return np.array(feats, dtype=np.float32)   # (42, 1280)


# ── Step 2: Build pairwise feature dataset ────────────────────────────────────

def build_pair_features(feat_matrix, img_df):
    """
    For each pair (i,j): feature = |fi - fj| (absolute difference, 1280-dim).
    Label = 1 if same port group, else 0.
    """
    labels_str = img_df["label_str"].tolist()
    n = len(labels_str)

    X_pairs, y_pairs = [], []
    for i, j in combinations(range(n), 2):
        diff = np.abs(feat_matrix[i] - feat_matrix[j])
        X_pairs.append(diff)
        gi = PORT_GROUPS.get(labels_str[i], -1)
        gj = PORT_GROUPS.get(labels_str[j], -1)
        y_pairs.append(1 if (gi == gj and gi != -1) else 0)

    return np.array(X_pairs, dtype=np.float32), np.array(y_pairs, dtype=int)


# ── Step 3: Lightweight MLP classifier on pair features ──────────────────────

class PairMLP(nn.Module):
    def __init__(self, in_dim=1280, dropout=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, device, epochs=30, lr=1e-3):
    model     = PairMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_tr).to(device)
    y_t = torch.tensor(y_tr, dtype=torch.long).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    return model


def eval_mlp(model, X_te, device):
    model.eval()
    X_t = torch.tensor(X_te).to(device)
    with torch.no_grad():
        out   = model(X_t)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
    return preds, probs


# ── Main entry ────────────────────────────────────────────────────────────────

def run_efficientnet(img_df, label_map):
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not TORCH_AVAILABLE:
        print("\n[EfficientNetV2] torch not installed. Skipping.\n")
        return {"accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1_score": 0.0, "mean_auc": 0.0}

    from sklearn.model_selection import StratifiedKFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EfficientNetV2] Device: {device}")

    # ── Extract features ONCE (the slow part — ~10s on CPU) ──────────────────
    print("[EfficientNetV2] Extracting deep features from 42 images (once) …")
    feat_matrix = extract_all_features(img_df, device)
    print(f"  Feature matrix: {feat_matrix.shape}")

    # ── Build pairwise dataset ────────────────────────────────────────────────
    X_pairs, y_pairs = build_pair_features(feat_matrix, img_df)
    n_pos = y_pairs.sum()
    n_neg = len(y_pairs) - n_pos
    print(f"  Pairs: {len(y_pairs)}  (same={n_pos}, diff={n_neg})")

    # ── 5-Fold CV on MLP (very fast — pure tensor ops) ───────────────────────
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_preds = np.zeros(len(y_pairs), dtype=int)
    all_probs = np.zeros((len(y_pairs), 2), dtype=np.float64)

    print("[EfficientNetV2] Running 5-Fold CV on MLP classifier …")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_pairs, y_pairs)):
        model = train_mlp(X_pairs[tr_idx], y_pairs[tr_idx], device)
        preds, probs = eval_mlp(model, X_pairs[te_idx], device)
        all_preds[te_idx] = preds
        all_probs[te_idx] = probs
        fold_acc = (preds == y_pairs[te_idx]).mean()
        print(f"  Fold {fold+1}/5  test_acc={fold_acc:.4f}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    label_names = ["Different Group", "Same Group"]
    metrics = compute_metrics(y_pairs, all_preds)
    print_metrics(metrics, "EfficientNetV2 (Frozen Backbone + MLP, 5-Fold CV)")
    print_classification_report(y_pairs, all_preds, label_names)

    plot_confusion_matrix(
        y_pairs, all_preds, label_names,
        "EfficientNetV2 — Confusion Matrix\n(Same vs Different Port Group)",
        os.path.join(RESULTS_DIR, "effnet_confusion_matrix.png")
    )
    mean_auc = plot_roc_curves(
        y_pairs, all_probs, label_names,
        "EfficientNetV2 — ROC Curves (Same vs Different Port Group)",
        os.path.join(RESULTS_DIR, "effnet_roc_curves.png")
    )
    metrics["mean_auc"] = mean_auc

    # ── Save final model ──────────────────────────────────────────────────────
    final_model = train_mlp(X_pairs, y_pairs, device, epochs=30)
    torch.save(final_model.state_dict(),
               os.path.join(MODEL_DIR, "efficientnet_mlp_head.pth"))
    np.save(os.path.join(MODEL_DIR, "efficientnet_features.npy"), feat_matrix)
    print(f"[EfficientNetV2] Model saved → {MODEL_DIR}")

    return metrics
