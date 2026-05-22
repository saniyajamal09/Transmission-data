"""
Streamlit Web App — AI-Based Breast Cancer Detection
Using Microwave S-Parameter Analysis
"""
import os
import sys
import re
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

# ── Port groups ───────────────────────────────────────────────────────────────
PORT_GROUPS = {
    "S12": 1, "S13": 1, "S14": 1, "S15": 1, "S16": 1, "S17": 1,
    "S21": 2, "S23": 2, "S24": 2, "S25": 2, "S26": 2, "S27": 2,
    "S31": 3, "S32": 3, "S34": 3, "S35": 3, "S36": 3, "S37": 3,
    "S41": 4, "S42": 4, "S43": 4, "S45": 4, "S46": 4, "S47": 4,
    "S51": 5, "S52": 5, "S53": 5, "S54": 5, "S56": 5, "S57": 5,
    "S61": 6, "S62": 6, "S63": 6, "S64": 6, "S65": 6, "S67": 6,
    "S71": 7, "S72": 7, "S73": 7, "S74": 7, "S75": 7, "S76": 7,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_s2p(filepath):
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
    return np.array(freqs), np.array(rows)


def extract_label(filename):
    match = re.search(r"S(\d{2})", os.path.basename(filename))
    return ("S" + match.group(1)) if match else "UNKNOWN"


def _stats(arr):
    return [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)),
            float(np.max(arr)), float(np.percentile(arr, 10)),
            float(np.percentile(arr, 25)), float(np.percentile(arr, 75)),
            float(np.percentile(arr, 90)), float(np.median(arr))]


def extract_features(freqs, data):
    col_names = ["S11_dB","S11_ph","S21_dB","S21_ph","S12_dB","S12_ph","S22_dB","S22_ph"]
    feats = []
    n = len(freqs)
    for i, name in enumerate(col_names):
        col = data[:, i]
        feats.extend(_stats(col))
        feats.append(float(np.argmax(np.abs(col))) / max(n-1, 1))
        feats.append(float(np.polyfit(np.arange(n), col, 1)[0]) if n > 1 else 0.0)
        if "dB" in name:
            fft_mag = np.abs(np.fft.rfft(col - col.mean()))
            feats.extend(_stats(fft_mag))
            feats.append(float(np.argmax(fft_mag)) / max(len(fft_mag)-1, 1))
        band_size = max(n // 4, 1)
        for b in range(4):
            seg = col[b*band_size:(b+1)*band_size]
            feats.append(float(np.mean(seg)))
    s21, s12 = data[:, 2], data[:, 4]
    feats.append(float(np.mean(np.abs(s21 - s12))))
    feats.append(float(np.max(s21) - np.min(s21)))
    feats.append(float((freqs[-1] - freqs[0]) / 1e9))
    return np.array(feats, dtype=np.float64)


def generate_spectral_image(freqs, data, label):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    titles  = ["S11 (dB)", "S21 (dB)", "S12 (dB)", "S22 (dB)"]
    db_cols = [0, 2, 4, 6]
    freq_ghz = freqs / 1e9
    for ax, title, ci in zip(axes.flat, titles, db_cols):
        ax.plot(freq_ghz, data[:, ci], linewidth=0.8, color="steelblue")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("GHz", fontsize=8)
        ax.set_ylabel("dB",  fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"S-Parameter Spectral Plot — {label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


@st.cache_resource
def load_lgbm_model():
    model  = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "lgbm_scaler.pkl"))
    return model, scaler


@st.cache_resource
def load_effnet_features():
    feat_path = os.path.join(MODEL_DIR, "efficientnet_features.npy")
    if os.path.exists(feat_path):
        return np.load(feat_path)
    return None


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔬 AI-Based Breast Cancer Detection")
st.markdown("### Microwave S-Parameter Analysis using LightGBM & EfficientNetV2")
st.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/cancer-ribbon.png", width=80)
st.sidebar.title("About")
st.sidebar.info(
    "This app classifies VNA S-parameter measurements to detect "
    "electromagnetic tissue signatures using two ML models:\n\n"
    "- **LightGBM** — Numerical features\n"
    "- **EfficientNetV2** — Spectral images"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Evaluation Results:**")
st.sidebar.metric("LightGBM Accuracy",    "83.97%")
st.sidebar.metric("EfficientNetV2 Accuracy", "87.80%")
st.sidebar.metric("LightGBM AUC",         "83.66%")

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload & Predict", "📊 Model Results", "ℹ️ About"])

# ── Tab 1: Upload & Predict ───────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a .s2p File for Classification")
    st.markdown("Upload any two VNA `.s2p` files to compare and classify them.")

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload File 1 (.s2p)", type=["s2p"], key="f1")
    with col2:
        file2 = st.file_uploader("Upload File 2 (.s2p)", type=["s2p"], key="f2")

    if file1 and file2:
        with st.spinner("Analyzing S-parameter data..."):

            # Save temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".s2p") as t1:
                t1.write(file1.read()); path1 = t1.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".s2p") as t2:
                t2.write(file2.read()); path2 = t2.name

            # Parse
            freqs1, data1 = parse_s2p(path1)
            freqs2, data2 = parse_s2p(path2)
            label1 = extract_label(file1.name)
            label2 = extract_label(file2.name)

            if data1.shape[0] == 0 or data2.shape[0] == 0:
                st.error("Could not parse one or both files. Please check the format.")
            else:
                # Spectral plots
                st.markdown("#### Spectral Images")
                c1, c2 = st.columns(2)
                with c1:
                    fig1 = generate_spectral_image(freqs1, data1, label1)
                    st.pyplot(fig1)
                    plt.close(fig1)
                with c2:
                    fig2 = generate_spectral_image(freqs2, data2, label2)
                    st.pyplot(fig2)
                    plt.close(fig2)

                # Feature extraction
                feat1 = extract_features(freqs1, data1)
                feat2 = extract_features(freqs2, data2)
                diff  = np.abs(feat1 - feat2).reshape(1, -1)

                # LightGBM prediction
                st.markdown("#### Prediction Results")
                try:
                    model, scaler = load_lgbm_model()
                    diff_scaled   = scaler.transform(diff)
                    lgbm_pred     = model.predict(diff_scaled)[0]
                    lgbm_prob     = model.predict_proba(diff_scaled)[0]
                    lgbm_label    = "✅ Same Port Group" if lgbm_pred == 1 else "❌ Different Port Group"
                    lgbm_conf     = lgbm_prob[lgbm_pred] * 100

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**🌿 LightGBM**")
                        st.success(lgbm_label) if lgbm_pred == 1 else st.error(lgbm_label)
                        st.metric("Confidence", f"{lgbm_conf:.1f}%")
                        st.progress(int(lgbm_conf))
                except Exception as e:
                    st.warning(f"LightGBM model not found: {e}")

                # Port group info
                g1 = PORT_GROUPS.get(label1, "?")
                g2 = PORT_GROUPS.get(label2, "?")
                st.markdown("---")
                st.markdown("#### File Information")
                info_df = pd.DataFrame({
                    "File"        : [file1.name, file2.name],
                    "Label"       : [label1, label2],
                    "Port Group"  : [f"Port {g1}", f"Port {g2}"],
                    "Freq Points" : [data1.shape[0], data2.shape[0]],
                    "Freq Range"  : [f"{freqs1[0]/1e6:.0f}–{freqs1[-1]/1e6:.0f} MHz",
                                     f"{freqs2[0]/1e6:.0f}–{freqs2[-1]/1e6:.0f} MHz"],
                })
                st.dataframe(info_df, use_container_width=True)

                # Cleanup
                os.unlink(path1); os.unlink(path2)

    else:
        st.info("👆 Please upload two .s2p files to get predictions.")

# ── Tab 2: Model Results ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Evaluation Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌿 LightGBM")
        lgbm_metrics = {
            "Accuracy" : "83.97%", "Precision": "84.36%",
            "Recall"   : "83.97%", "F1-Score" : "84.16%",
            "AUC"      : "83.66%", "Train Time": "~2 sec"
        }
        for k, v in lgbm_metrics.items():
            st.metric(k, v)

    with col2:
        st.markdown("#### 🧠 EfficientNetV2")
        effnet_metrics = {
            "Accuracy" : "87.80%", "Precision": "77.10%",
            "Recall"   : "87.80%", "F1-Score" : "82.10%",
            "AUC"      : "53.54%", "Train Time": "~6 sec"
        }
        for k, v in effnet_metrics.items():
            st.metric(k, v)

    st.markdown("---")
    st.markdown("#### Comparison Table")
    comp_df = pd.DataFrame({
        "Metric"        : ["Accuracy","Precision","Recall","F1-Score","AUC"],
        "LightGBM"      : ["83.97%","84.36%","83.97%","84.16%","83.66%"],
        "EfficientNetV2": ["87.80%","77.10%","87.80%","82.10%","53.54%"],
        "Winner"        : ["EfficientNetV2","LightGBM","EfficientNetV2","LightGBM","LightGBM"],
    })
    st.dataframe(comp_df, use_container_width=True)

    # Show saved graphs
    st.markdown("---")
    st.markdown("#### Saved Evaluation Graphs")
    graph_files = {
        "All Metrics Comparison"  : "comparison_all_metrics.png",
        "Accuracy Comparison"     : "comparison_accuracy.png",
        "Precision Comparison"    : "comparison_precision.png",
        "Fold-wise Accuracy"      : "foldwise_accuracy.png",
        "LightGBM Confusion Matrix"    : "lgbm_confusion_matrix.png",
        "LightGBM ROC Curve"           : "lgbm_roc_curves.png",
        "EfficientNetV2 Confusion Matrix": "effnet_confusion_matrix.png",
        "EfficientNetV2 ROC Curve"     : "effnet_roc_curves.png",
    }
    cols = st.columns(2)
    for idx, (title, fname) in enumerate(graph_files.items()):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            with cols[idx % 2]:
                st.markdown(f"**{title}**")
                st.image(fpath, use_container_width=True)

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    ## AI-Based Breast Cancer Detection Using Microwave S-Parameter Analysis

    Breast cancer is one of the leading causes of death among women globally.
    This project builds an automated ML pipeline on VNA-generated `.s2p` files
    to classify electromagnetic S-parameter signatures — with application to
    microwave-based breast cancer screening.

    ### Models Used
    | Model | Input | Accuracy | AUC |
    |---|---|---|---|
    | **LightGBM** | 163 numerical features | 83.97% | 83.66% |
    | **EfficientNetV2** | Spectral images | 87.80% | 53.54% |

    ### Dataset
    - 42 VNA `.s2p` files (7-port RF network)
    - 1568 frequency points per file (1 MHz – 555 MHz)
    - 861 pairwise combinations for binary classification
    - 5-Fold Stratified Cross-Validation

    ### Evaluation Metrics
    Accuracy · Precision · Recall · F1-Score · Confusion Matrix · ROC Curve · AUC

    ### Tech Stack
    `Python` · `LightGBM` · `PyTorch` · `EfficientNetV2` · `Scikit-learn` · `Streamlit`
    """)
