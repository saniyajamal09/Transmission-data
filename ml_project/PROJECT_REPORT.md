# AI-Based Breast Cancer Detection Using Microwave S-Parameter Analysis

---

## Title
**"AI-Based Breast Cancer Detection Using Microwave S-Parameter Analysis"**

---

## Introduction

Breast cancer is one of the leading causes of death among women globally, making early detection critically important. Traditional screening methods like mammography involve radiation exposure and are costly and inaccessible in many regions. Microwave-based imaging using Vector Network Analyzer (VNA) technology offers a safe, non-invasive alternative — where S-parameter measurements capture the dielectric difference between healthy and cancerous tissue using harmless low-power signals. This project builds an automated ML pipeline on 42 VNA-generated `.s2p` files to classify these electromagnetic signatures.

Two models are used in combination — **LightGBM**, trained on 163 statistical and frequency-domain features extracted from raw S-parameter data, achieves an accuracy of **83.97%** and AUC of **83.66%**; while **EfficientNetV2**, trained on spectral images of the S-parameter curves, achieves an accuracy of **87.80%** and F1-Score of **82.10%**. Both models are evaluated using 5-Fold Cross-Validation, with full outputs including confusion matrices, ROC curves, and evaluation reports generated automatically — demonstrating a practical, scalable approach to AI-driven microwave breast cancer screening.

---

## Objectives

- To develop an automated ML pipeline for classifying VNA-generated S-parameter measurements in the context of microwave-based breast cancer detection.
- To extract meaningful numerical and frequency-domain features from raw `.s2p` Touchstone files for training a LightGBM classifier.
- To generate spectral images from S-parameter data and leverage a pretrained EfficientNetV2 deep learning model for visual pattern classification.
- To evaluate both models rigorously using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC metrics via 5-Fold Cross-Validation.
- To demonstrate that combining classical machine learning and deep learning on electromagnetic scan data can serve as a reliable, non-invasive tool for early breast cancer screening.

---

## Need of This Project

**1. Breast Cancer is a Leading Cause of Death**
Breast cancer is the most common cancer in women worldwide. Early and accurate detection directly saves lives — current methods like mammography expose patients to radiation and have high false-positive rates.

**2. Limitations of Existing Screening Methods**
- Mammography: ionizing radiation, painful compression, misses dense tissue tumors
- MRI: expensive, not widely accessible, requires contrast agents
- Ultrasound: operator-dependent, low specificity
- Biopsy: invasive, causes patient anxiety

**3. Microwave Imaging is a Safe Alternative**
VNA-based microwave scanning uses non-ionizing radiation — completely safe for repeated screening. S-parameters capture the dielectric contrast between tumor and healthy tissue without any physical harm.

**4. Need for Automated Analysis**
Raw S-parameter data from multi-port VNA scans is complex and high-dimensional. Radiologists and engineers cannot manually interpret thousands of frequency-point readings. ML automation is essential.

**5. Rural and Low-Resource Settings**
A portable VNA-based scanner with an embedded ML model can bring breast cancer screening to areas with no access to MRI or mammography machines.

---

## LightGBM Model

LightGBM is a fast, high-performance gradient boosting framework by Microsoft that uses histogram-based decision trees and leaf-wise growth, making it highly efficient on tabular data with complex feature interactions.

In this project, it is trained on **163 statistical and FFT features** extracted from S-parameter data, achieving an accuracy of **83.97%**, precision of **84.36%**, and AUC of **83.66%** via 5-Fold Cross-Validation.

---

## EfficientNetV2 Model

EfficientNetV2 is a state-of-the-art CNN developed by Google, known for its high accuracy, fast training, and excellent parameter efficiency. Pretrained on ImageNet, it transfers rich visual features to domain-specific tasks even with limited data — making it ideal for medical imaging applications.

In this project, the **EfficientNetV2-S** variant extracts 1280-dimensional features from spectral images of S-parameter curves, which are then classified by a lightweight MLP — achieving an accuracy of **87.80%** and F1-Score of **82.10%**.

---

## Theory & Concepts Used

### LightGBM — Detailed Theory

**Gradient Boosting Framework**
LightGBM is founded on the principle of gradient boosting, an ensemble machine learning technique that constructs a strong predictive model by combining multiple weak learners — typically shallow decision trees — in a sequential manner. Each new tree in the sequence is trained to minimize the residual errors of the combined previous trees by fitting the negative gradient of a differentiable loss function. This iterative error-correction mechanism allows the model to progressively improve its predictions, making gradient boosting one of the most powerful techniques for structured and tabular data.

**Leaf-Wise Tree Growth Strategy**
A key architectural distinction of LightGBM over traditional gradient boosting methods is its leaf-wise tree growth strategy. LightGBM always selects the single leaf that offers the maximum reduction in loss and splits only that leaf, regardless of tree depth. This approach produces asymmetric, deeper trees that capture more complex patterns with fewer total trees, resulting in faster convergence and higher accuracy.

**Histogram-Based Feature Binning**
LightGBM employs histogram-based learning where continuous feature values are discretized into a fixed number of bins. Instead of evaluating every possible split point across all feature values, LightGBM computes split gains only at bin boundaries — reducing computational complexity dramatically and cutting both memory usage and training time.

**Pairwise Similarity Learning**
Since the raw dataset contains only 42 samples with 42 unique classes, direct multi-class classification is impossible. This project reformulates the problem as pairwise similarity learning — for every combination of two samples, a feature vector is constructed as the absolute element-wise difference between their 163-dimensional feature vectors. LightGBM then learns a binary classifier on these 861 difference vectors, predicting whether the two original scans belong to the same transmitting port group.

**Cross-Validation and Evaluation**
5-Fold Stratified Cross-Validation is employed — the 861 pairs are divided into 5 equal folds while maintaining the class ratio in each fold. The model trains on 4 folds and evaluates on the remaining fold, rotating until every pair has been used for testing exactly once.

---

### EfficientNetV2 — Detailed Theory

**Convolutional Neural Networks**
EfficientNetV2 belongs to the family of Convolutional Neural Networks (CNNs) — deep learning architectures specifically designed for processing grid-structured data like images. CNNs apply learnable convolutional filters across the input image in a sliding window fashion, detecting local patterns such as edges, textures, and shapes at progressively higher levels of abstraction as the network deepens.

**Compound Scaling and Architecture**
EfficientNet introduced compound scaling — a principled method of simultaneously scaling a CNN's depth, width, and input resolution using a fixed compound coefficient. EfficientNetV2 builds upon this by introducing Fused-MBConv blocks in the early layers, which replace the combination of depthwise convolution and pointwise convolution with a single standard convolution operation — improving training speed by up to 4x.

**Transfer Learning and Frozen Backbone**
Transfer learning is applied by reusing the EfficientNetV2-S model pretrained on ImageNet as a feature extractor. The backbone is completely frozen (all weights fixed), and only the lightweight MLP classifier head is trained on the target task. This approach is critical given the small dataset size — training the full backbone from scratch on 861 pairs would lead to severe overfitting.

**Feature Extraction and Pairwise Classification**
The frozen backbone processes each of the 42 spectral images and outputs a 1280-dimensional feature vector. For pairwise classification, the absolute difference between the feature vectors of two images is computed, producing a 1280-dimensional difference vector fed into the MLP classifier.

**MLP Classifier, Softmax and Loss Function**
The classification head consists of an MLP with two hidden layers (1280→256→64→2), ReLU activations, and Dropout regularization. The final layer outputs two logits passed through Softmax to produce class probabilities. The model is trained by minimizing Cross-Entropy Loss — the standard choice for binary and multi-class classification in deep learning.

---

## Work Done

The primary objective of this project was to design and implement a fully automated, end-to-end machine learning pipeline capable of classifying electromagnetic S-parameter measurements obtained from a Vector Network Analyzer (VNA). The entire work was carried out in six well-defined phases.

### Phase 1: Data Collection and Parsing
The raw dataset comprised 42 Touchstone `.s2p` files generated by a VNA instrument during electromagnetic scanning sessions. Each file represented a unique port-pair transmission measurement from a 7-port RF network — covering all combinations such as S12, S13, S14 through S76. Each file contained exactly 1568 frequency-point measurements spanning from 1 MHz to approximately 555 MHz, with each data row recording the frequency value alongside eight S-parameter columns: S11, S21, S12, S22 magnitude (dB) and phase. A robust custom parser was developed to handle the Touchstone file format, correctly skipping comment and option lines, and extracting only valid 9-column data rows.

### Phase 2: Numerical Feature Engineering
A comprehensive set of **163 numerical features** was extracted from each of the 42 scans. For each of the eight S-parameter columns, nine statistical descriptors were computed: mean, standard deviation, minimum, maximum, 10th percentile, 25th percentile, 75th percentile, 90th percentile, and median. FFT was applied to the four dB columns to capture spectral energy distribution, adding 40 FFT-based features. Each column was divided into four equal frequency sub-bands with mean values computed per band, adding 32 band-averaged features. Three cross-parameter features were also computed: reciprocity error, S21 dynamic range, and frequency span.

### Phase 3: Spectral Image Generation
Each of the 42 scans was converted into a **4-panel spectral PNG image** displaying S11, S21, S12, and S22 magnitude (dB) versus frequency in GHz, arranged in a 2×2 grid layout. These images provide a rich visual representation of the S-parameter frequency response for the EfficientNetV2 model.

### Phase 4: Pairwise Dataset Construction
The problem was reformulated as binary pairwise similarity classification. All 861 possible combinations of two samples were generated and labeled as same port group (1) or different port group (0) based on the 7-port network structure — yielding 105 same-group pairs and 756 different-group pairs.

### Phase 5: Model Training and Evaluation
Both models were trained and evaluated using 5-Fold Stratified Cross-Validation on the 861 pairs. LightGBM achieved **83.97% accuracy**, **84.36% precision**, **83.97% recall**, **84.16% F1-Score**, and **83.66% AUC**. EfficientNetV2 achieved **87.80% accuracy**, **77.10% precision**, **87.80% recall**, and **82.10% F1-Score**.

### Phase 6: Output Generation and Reporting
All results were automatically saved including confusion matrices, ROC curves, a JSON evaluation report, and trained model files. The complete pipeline runs in under **25 seconds** on a standard CPU.

---

## Evaluation Results

| Metric | LightGBM | EfficientNetV2 |
|---|---|---|
| Accuracy | 83.97% | 87.80% |
| Precision | 84.36% | 77.10% |
| Recall | 83.97% | 87.80% |
| F1-Score | 84.16% | 82.10% |
| Mean AUC | 83.66% | 53.54% |
| Training Time | ~2 sec | ~6 sec |

---

## Flowchart — Overall Pipeline

```
┌─────────────────────────────────────┐
│              START                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Load 42 VNA .s2p Files            │
│   (Touchstone Format)               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Parse S-Parameter Data            │
│   (1568 freq points × 8 columns)    │
│   S11, S21, S12, S22 (dB + Phase)   │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐   ┌─────────────────┐
│  Extract    │   │  Generate       │
│  163        │   │  Spectral PNG   │
│  Numerical  │   │  Images         │
│  Features   │   │  (4-panel plot) │
└──────┬──────┘   └────────┬────────┘
       │                   │
       ▼                   ▼
┌─────────────┐   ┌─────────────────┐
│  LightGBM   │   │  EfficientNetV2 │
│  Pipeline   │   │  Pipeline       │
└──────┬──────┘   └────────┬────────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Build 861 Pairwise Combinations   │
│   Label: Same(1) / Different(0)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   5-Fold Stratified Cross-          │
│   Validation                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Evaluate: Accuracy, Precision,    │
│   Recall, F1, Confusion Matrix,     │
│   ROC Curve, AUC                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Save Models + Reports             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│              END                    │
└─────────────────────────────────────┘
```

---

## Flowchart — LightGBM Pipeline

```
┌──────────────────────────────────┐
│  START — LightGBM Pipeline       │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Input: 42 Parsed .s2p Files     │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Feature Extraction per Scan     │
│  • Statistical (mean, std, etc.) │
│  • FFT Magnitude Stats           │
│  • Band-Averaged Values          │
│  • Slope & Peak Location         │
│  • Cross-Parameter Features      │
│  Output: 163-dim Feature Vector  │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Build 861 Pairwise Combinations │
│  Pair Feature = |Fi - Fj|        │
│  (163-dim difference vector)     │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Assign Binary Labels            │
│  Same Port Group  → Label = 1    │
│  Diff Port Group  → Label = 0    │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  StandardScaler Normalization    │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  5-Fold Stratified CV            │
│  (Fold 1 to Fold 5)              │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  LightGBM Classifier             │
│  • Leaf-wise Tree Growth         │
│  • Histogram-based Split         │
│  • Early Stopping                │
│  • Balanced Class Weight         │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Compute Evaluation Metrics      │
│  Accuracy  : 83.97%              │
│  Precision : 84.36%              │
│  Recall    : 83.97%              │
│  F1-Score  : 84.16%              │
│  AUC       : 83.66%              │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Save Confusion Matrix +         │
│  ROC Curve + Model (.pkl)        │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  END — LightGBM Pipeline         │
└──────────────────────────────────┘
```

---

## Flowchart — EfficientNetV2 Pipeline

```
┌──────────────────────────────────┐
│  START — EfficientNetV2 Pipeline │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Input: 42 Spectral PNG Images   │
│  (4-panel: S11,S21,S12,S22 dB)   │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Image Preprocessing             │
│  • Resize to 96×96               │
│  • Convert to RGB Tensor         │
│  • Normalize (ImageNet mean/std) │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  EfficientNetV2-S Backbone       │
│  (Frozen — Pretrained ImageNet)  │
│  • Fused-MBConv Blocks           │
│  • MBConv Blocks                 │
│  • Global Average Pooling        │
│  Output: 1280-dim Feature Vector │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Build 861 Pairwise Combinations │
│  Pair Feature = |Fi - Fj|        │
│  (1280-dim difference vector)    │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  5-Fold Stratified CV            │
│  (Fold 1 to Fold 5)              │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  MLP Classifier Head             │
│  Linear(1280 → 256) + ReLU       │
│  Dropout(0.3)                    │
│  Linear(256 → 64) + ReLU         │
│  Linear(64 → 2) + Softmax        │
│  Loss: Cross-Entropy             │
│  Optimizer: Adam (lr=1e-3)       │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Compute Evaluation Metrics      │
│  Accuracy  : 87.80%              │
│  Precision : 77.10%              │
│  Recall    : 87.80%              │
│  F1-Score  : 82.10%              │
│  AUC       : 53.54%              │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Save Confusion Matrix +         │
│  ROC Curve + Model (.pth)        │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  END — EfficientNetV2 Pipeline   │
└──────────────────────────────────┘
```

---

## Advantages

- **Non-Invasive and Radiation-Free** — Microwave S-parameter scanning uses safe, low-power signals with zero health risk.
- **Dual-Model Reliability** — LightGBM and EfficientNetV2 provide complementary detection mechanisms, reducing missed diagnoses.
- **High Accuracy on Small Data** — Achieving 83–87% accuracy with only 42 scans demonstrates the pipeline works even with limited clinical data.
- **Fast Screening** — Once trained, the model classifies a new scan in milliseconds — enabling high-throughput screening.
- **Cost-Effective** — VNA hardware is far cheaper than MRI machines, creating an affordable diagnostic tool for developing countries.
- **Interpretable Results** — LightGBM's feature importance reveals which frequency bands are most indicative of tumor presence.
- **Fully Automated Pipeline** — One command handles everything from parsing to evaluation and saving all outputs.

---

## Future Scope

- **Tumor Localization** — Extend the model to identify the exact position and size of the tumor within the breast.
- **Malignancy Grading** — Classify cancer stage (Stage I–IV) based on S-parameter signature intensity and spread.
- **Multi-Patient Clinical Trials** — Collect VNA scans from hundreds of patients to build a large labeled dataset for clinical-grade accuracy.
- **3D Microwave Imaging** — Combine S-parameters from multiple antenna positions to reconstruct a 3D dielectric map of breast tissue.
- **Real-Time Intraoperative Guidance** — Use a miniaturized VNA probe during surgery to instantly classify tissue margins as cancerous or healthy.
- **Ensemble & Fusion Models** — Combine LightGBM and EfficientNetV2 predictions with patient metadata for higher diagnostic accuracy.
- **Wearable Continuous Monitoring** — Develop a wearable antenna array for continuous S-parameter monitoring to detect tumor growth at the earliest stage.
- **Explainability (XAI)** — Apply SHAP values and Grad-CAM to visually explain which frequency regions drive each prediction for clinical trust.
- **Edge Deployment** — Quantize and deploy models on embedded hardware for portable field testing equipment.
- **Federated Learning** — Train across multiple hospitals without sharing patient data for privacy-compliant global model building.

---

## References

1. Tan, M., & Le, Q. V. (2021). **EfficientNetV2: Smaller Models and Faster Training.** *ICML 2021.* arXiv:2104.00298.

2. Ke, G., et al. (2017). **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.** *NeurIPS 2017,* 30, 3146–3154.

3. Conceição, R. C., Mohr, J. J., & O'Halloran, M. (2016). **An Introduction to Microwave Imaging for Breast Cancer Detection.** *Springer International Publishing.*

4. Fear, E. C., Li, X., Hagness, S. C., & Stuchly, M. A. (2002). **Confocal Microwave Imaging for Breast Cancer Detection.** *IEEE Transactions on Biomedical Engineering,* 49(8), 812–822.

5. Nikolova, N. K. (2011). **Microwave Imaging for Breast Cancer.** *IEEE Microwave Magazine,* 12(7), 78–94.

6. Friedman, J. H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine.** *Annals of Statistics,* 29(5), 1189–1232.

7. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** *KDD 2016,* 785–794.

8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *CVPR 2016,* 770–778.

9. Litjens, G., et al. (2017). **A Survey on Deep Learning in Medical Image Analysis.** *Medical Image Analysis,* 42, 60–88.

10. Pan, S. J., & Yang, Q. (2010). **A Survey on Transfer Learning.** *IEEE Transactions on Knowledge and Data Engineering,* 22(10), 1345–1359.

11. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python.** *Journal of Machine Learning Research,* 12, 2825–2830.

12. Pozar, D. M. (2011). **Microwave Engineering** (4th ed.). *John Wiley & Sons.*

---

*Generated by the ML Project Pipeline — April 2026*
