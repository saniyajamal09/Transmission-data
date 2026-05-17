"""
Parse all .s2p Touchstone files and extract:
  - Numerical features (for LightGBM)
  - Spectral images (for EfficientNetV2)

Dataset reality: 42 files, 1 sample per class → we treat each file as a sample
and the port-pair (S12, S13 …) as the class label.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA_DIR, IMAGE_DIR


# ── S2P parser ────────────────────────────────────────────────────────────────

def parse_s2p(filepath):
    """
    Read a Touchstone .s2p file (DB/angle format).
    Returns (freqs, data) where data.shape = (N, 8):
      S11_dB, S11_ph, S21_dB, S21_ph, S12_dB, S12_ph, S22_dB, S22_ph
    """
    freqs, rows = [], []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) == 9:
                try:
                    freqs.append(float(vals[0]))
                    rows.append([float(v) for v in vals[1:]])
                except ValueError:
                    continue
    return np.array(freqs, dtype=np.float64), np.array(rows, dtype=np.float64)


def extract_label(filename):
    """Extract port-pair label, e.g. 'S12', 'S34' from filename."""
    basename = os.path.basename(filename)
    match = re.search(r"S(\d{2})", basename)
    return ("S" + match.group(1)) if match else "UNKNOWN"


# ── Feature engineering ───────────────────────────────────────────────────────

def _stats(arr):
    """9 statistical descriptors of a 1-D array."""
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 90)),
        float(np.median(arr)),
    ]

STAT_SUFFIXES = ["mean", "std", "min", "max", "p10", "p25", "p75", "p90", "median"]


def extract_numerical_features(freqs, data):
    """
    Build a rich flat feature vector from S-parameter data.
    Includes: time-domain stats, FFT stats, band-averaged values,
    slope (linear trend), peak location.
    """
    col_names = ["S11_dB", "S11_ph", "S21_dB", "S21_ph",
                 "S12_dB", "S12_ph", "S22_dB", "S22_ph"]
    feats, feat_names = [], []

    freq_ghz = freqs / 1e9
    n = len(freqs)

    for i, name in enumerate(col_names):
        col = data[:, i]

        # Basic stats
        for val, suf in zip(_stats(col), STAT_SUFFIXES):
            feats.append(val)
            feat_names.append(f"{name}_{suf}")

        # Peak location (normalised index)
        feats.append(float(np.argmax(np.abs(col))) / max(n - 1, 1))
        feat_names.append(f"{name}_peak_loc")

        # Linear slope (trend)
        if n > 1:
            slope = float(np.polyfit(np.arange(n), col, 1)[0])
        else:
            slope = 0.0
        feats.append(slope)
        feat_names.append(f"{name}_slope")

        # FFT magnitude stats (dB columns only — phase is circular)
        if "dB" in name:
            fft_mag = np.abs(np.fft.rfft(col - col.mean()))
            for val, suf in zip(_stats(fft_mag), STAT_SUFFIXES):
                feats.append(val)
                feat_names.append(f"{name}_fft_{suf}")
            # Dominant frequency bin
            feats.append(float(np.argmax(fft_mag)) / max(len(fft_mag) - 1, 1))
            feat_names.append(f"{name}_fft_dom_bin")

        # Band-averaged values (split into 4 equal bands)
        band_size = max(n // 4, 1)
        for b in range(4):
            seg = col[b * band_size: (b + 1) * band_size]
            feats.append(float(np.mean(seg)))
            feat_names.append(f"{name}_band{b}_mean")

    # Cross-parameter features
    s21_db = data[:, 2]
    s12_db = data[:, 4]
    feats.append(float(np.mean(np.abs(s21_db - s12_db))))   # reciprocity error
    feat_names.append("reciprocity_error_mean")

    feats.append(float(np.max(s21_db) - np.min(s21_db)))    # S21 dynamic range
    feat_names.append("S21_dynamic_range")

    feats.append(float(freq_ghz[-1] - freq_ghz[0]))          # freq span
    feat_names.append("freq_span_GHz")

    return np.array(feats, dtype=np.float64), feat_names


# ── Image generation ──────────────────────────────────────────────────────────

def generate_image(freqs, data, label, filename, out_dir):
    """
    4-panel spectral image: S11, S21, S12, S22 magnitude (dB) vs frequency.
    Saved as PNG for EfficientNetV2 input.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    titles  = ["S11 (dB)", "S21 (dB)", "S12 (dB)", "S22 (dB)"]
    db_cols = [0, 2, 4, 6]
    freq_ghz = freqs / 1e9

    for ax, title, ci in zip(axes.flat, titles, db_cols):
        ax.plot(freq_ghz, data[:, ci], linewidth=0.7, color="steelblue")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("GHz", fontsize=7)
        ax.set_ylabel("dB",  fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(label, fontsize=10, fontweight="bold")
    plt.tight_layout()
    stem = os.path.splitext(filename)[0]
    img_path = os.path.join(out_dir, f"{stem}.png")
    plt.savefig(img_path, dpi=100)
    plt.close(fig)
    return img_path


# ── Main loader ───────────────────────────────────────────────────────────────

def load_dataset(data_dir=DATA_DIR, image_dir=IMAGE_DIR, generate_images=True):
    """
    Load all .s2p files.
    Returns:
      df_num    : DataFrame  [features … , label (int), label_str]
      img_df    : DataFrame  [image_path, label (int), label_str]
      label_map : dict {label_str -> int}
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.s2p")))
    if not files:
        raise FileNotFoundError(f"No .s2p files found in: {data_dir}")

    print(f"\nFound {len(files)} .s2p files in {data_dir}")

    all_feats, all_labels, all_img_paths = [], [], []
    feat_names = None

    for fp in files:
        label  = extract_label(fp)
        freqs, data = parse_s2p(fp)

        if data.shape[0] == 0:
            print(f"  [SKIP] {os.path.basename(fp)} — empty")
            continue

        feats, feat_names = extract_numerical_features(freqs, data)
        all_feats.append(feats)
        all_labels.append(label)

        if generate_images:
            img_path = generate_image(freqs, data, label,
                                      os.path.basename(fp), image_dir)
            all_img_paths.append(img_path)
        else:
            all_img_paths.append(None)

        print(f"  {os.path.basename(fp):45s}  label={label}  pts={data.shape[0]}")

    # Integer-encode labels
    unique_labels = sorted(set(all_labels))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = [label_map[l] for l in all_labels]

    df_num = pd.DataFrame(all_feats, columns=feat_names)
    df_num["label"]     = y
    df_num["label_str"] = all_labels

    img_df = pd.DataFrame({
        "image_path": all_img_paths,
        "label"     : y,
        "label_str" : all_labels,
    })

    print(f"\nDataset summary: {len(df_num)} samples | "
          f"{len(feat_names)} features | {len(unique_labels)} classes")
    return df_num, img_df, label_map


if __name__ == "__main__":
    df, img, lmap = load_dataset(generate_images=False)
    print(df.head())
    print("Label map:", lmap)
