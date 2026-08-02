# NitinsaiKaruturi-x596c589-CS898BA-Project

**Author:** Nitinsai Karuturi
**Course:** CS 898BA – Image Analysis and Computer Vision
**Project:** Recognizing Vehicles in Radar Surveillance Images
**Institution:** Wichita State University — Summer 2026

---

## Project Overview

This project implements a Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) pipeline that classifies military ground vehicles from radar imagery. SAR imaging enables all-weather, day-and-night ground surveillance but presents unique classification challenges — multiplicative speckle noise, aspect-angle-dependent target signatures, and high inter-class visual similarity between vehicle types.

The system combines classical SAR image processing (Lee adaptive filtering, CFAR segmentation, morphological cleanup) with a lightweight convolutional neural network to classify three vehicle types from the MSTAR benchmark dataset. Hyperparameter optimization was performed via grid search across 12 configurations to identify the best training setup.

---

## Setup & Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Pillow, scikit-learn, scikit-image, Matplotlib, PyTorch, torchvision

### Install Dependencies

```bash
pip install numpy pillow scipy scikit-learn scikit-image matplotlib torch torchvision
```

---

## Execution

Run scripts in this order from the project root directory:

```bash
python src/load_data.py
python src/preprocess.py
python src/baseline_hog_svm.py
python src/train_cnn.py
python src/hyperparameter_tuning.py
python src/final_evaluation.py
```

All outputs are saved automatically to the `outputs/` folder.

> **Note:** The dataset (~28 MB) is downloaded automatically when you run `src/load_data.py`. No manual download required.

---

## File Descriptions

| File | Purpose |
|---|---|
| `src/load_data.py` | Downloads MSTAR NPZ file, verifies class distribution, saves sample chips per class |
| `src/preprocess.py` | Applies Lee adaptive filter, CFAR segmentation, morphological cleanup, and 64×64 centre crop |
| `src/baseline_hog_svm.py` | Extracts HOG features and trains RBF-SVM classifier as the classical baseline |
| `src/train_cnn.py` | Defines and trains SARNet CNN for 10 epochs on CPU, saves weights and training curves |
| `src/hyperparameter_tuning.py` | Grid search across 12 configurations (LR × Batch × Dropout), saves best config and comparison charts |
| `src/final_evaluation.py` | Trains and evaluates all three pipeline variants, produces full comparison table and confusion matrices |

---

## Dataset

| Property | Value |
|---|---|
| Name | MSTAR (Moving and Stationary Target Acquisition and Recognition) |
| Classes | BMP2 (infantry vehicle), BTR70 (armored carrier), T72 (main battle tank) |
| Train samples | 696 (232 per class, 17° depression angle) |
| Test samples | 822 (274 per class, 15° depression angle) |
| Image size | 32×32 grayscale SAR chips |
| Preprocessed size | 64×64 after pipeline |
| Source | Downloaded automatically by `src/load_data.py` |

---

## Pipeline

```
Raw SAR Chip (32×32)
       │
       ▼
[1] Lee Adaptive Filter       — adaptive smoothing for multiplicative speckle noise
       │
       ▼
[2] CFAR Segmentation         — 95th percentile threshold isolates target return
       │
       ▼
[3] Morphological Cleanup     — binary opening/closing removes stray noise pixels
       │
       ▼
[4] Standardised Centre Crop  — 64×64 crop centred on target region
       │
       ├──▶ [5a] HOG + RBF-SVM    (classical baseline)
       └──▶ [5b] SARNet CNN       (deep learning classifier)
```

---

## Model Architecture — SARNet

```
Input:        (N, 1, 64, 64)
Conv Block 1: Conv2d(1→32,   3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
Conv Block 2: Conv2d(32→64,  3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
Conv Block 3: Conv2d(64→128, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
Flatten:      128 × 8 × 8 = 8192
FC1:          Linear(8192→512) → ReLU → Dropout(0.3)
FC2:          Linear(512→128)  → ReLU → Dropout(0.3)
FC3:          Linear(128→3)    → softmax output

Optimizer:    Adam (lr=0.0001, StepLR ×0.5 every 5 epochs)
Loss:         CrossEntropyLoss | Batch size: 16 | Epochs: 15
```

---

## Hyperparameter Optimization

Grid search across 12 configurations (LR × Batch Size × Dropout), 10 epochs each:

| # | LR | Batch | Dropout | Best Acc |
|---|---|---|---|---|
| 1 | 0.01 | 16 | 0.3 | 81.3% |
| 2 | 0.01 | 16 | 0.5 | 81.6% |
| 3 | 0.01 | 32 | 0.3 | 86.7% |
| 4 | 0.01 | 32 | 0.5 | 75.7% |
| 5 | 0.001 | 16 | 0.3 | 96.8% |
| 6 | 0.001 | 16 | 0.5 | 97.8% |
| 7 | 0.001 | 32 | 0.3 | 95.9% |
| 8 | 0.001 | 32 | 0.5 | 95.7% |
| **9** | **0.0001** | **16** | **0.3** | **98.4% ✓ Best** |
| 10 | 0.0001 | 16 | 0.5 | 98.2% |
| 11 | 0.0001 | 32 | 0.3 | 95.6% |
| 12 | 0.0001 | 32 | 0.5 | 95.3% |

**Best configuration:** LR=0.0001, Batch=16, Dropout=0.3

**Key finding:** Lower learning rate with small batch size produced the most stable convergence on this small dataset.

---

## Results

### Final Comparison — All Three Pipeline Variants

| Model | Test Accuracy | BMP2 F1 | BTR70 F1 | T72 F1 |
|---|---|---|---|---|
| HOG + SVM (baseline) | 91.4% | 0.86 | 0.89 | 0.99 |
| CNN — No Pipeline | 94.9% | 0.92 | 0.93 | 1.00 |
| CNN — Pipeline + Tuned | **99.5%** | **0.99** | **1.00** | **1.00** |

### Final CNN Training Curve (Tuned Model, 15 epochs)

| Epoch | Test Acc |
|---|---|
| 1 | 40.5% |
| 4 | 96.7% |
| 8 | 98.7% |
| 13 | 99.3% |
| 15 | **99.5%** |

---

## Evaluation & Analysis

**HOG+SVM (91.4%):** HOG captures structural edge patterns from despeckled chips. The RBF-SVM separates classes well overall but cannot distinguish BMP2 from BTR70 reliably — 60 of 274 BMP2 test chips were misclassified as BTR70. Both are wheeled armored vehicles with overlapping SAR cross-section profiles at certain aspect angles, making the feature distributions inseparable in HOG space.

**CNN — No Pipeline (94.9%):** Same SARNet architecture trained on raw chips without Lee filter or CFAR preprocessing. Achieves 94.9% — better than HOG+SVM but 4.6% below the tuned pipeline model. This confirms that the architecture alone is not sufficient; domain-specific preprocessing is a meaningful contributor to final accuracy.

**CNN — Pipeline + Tuned (99.5%):** Full pipeline preprocessing combined with best hyperparameters (LR=0.0001, Batch=16, Dropout=0.3) trained for 15 epochs. BMP2↔BTR70 confusion dropped from 60 misclassifications to just 2. BTR70 F1 reached 1.00, completely resolving the dominant error pattern from the baseline.

**Impact of preprocessing:** The pipeline contributed +4.6% accuracy improvement over a raw-chip CNN using the identical architecture — proving Lee filtering and CFAR segmentation are decisive contributors, not optional preprocessing steps.

**T72 performance:** T72 achieves F1 = 0.99–1.00 across all models. The tank's distinctive turret shape produces a SAR signature clearly different from both wheeled vehicles, making it reliably separable regardless of classifier.

---

## Output Files

| File | Description |
|---|---|
| `outputs/sample_BMP2.png` | Sample chip — BMP2 infantry vehicle |
| `outputs/sample_BTR70.png` | Sample chip — BTR70 armored carrier |
| `outputs/sample_T72.png` | Sample chip — T72 main battle tank |
| `outputs/pipeline_BMP2.png` | 4-panel preprocessing demo — BMP2 |
| `outputs/pipeline_BTR70.png` | 4-panel preprocessing demo — BTR70 |
| `outputs/pipeline_T72.png` | 4-panel preprocessing demo — T72 |
| `outputs/confusion_hog_svm.png` | HOG+SVM confusion matrix |
| `outputs/confusion_cnn.png` | CNN confusion matrix (initial 10 epochs) |
| `outputs/cnn_training_curves.png` | Loss and accuracy curves — initial CNN |
| `outputs/sarnet_weights.pt` | Saved SARNet model weights (initial) |
| `outputs/hyperparameter_comparison.png` | Bar chart of all 12 hyperparameter configurations |
| `outputs/top3_learning_curves.png` | Learning curves for top 3 configurations |
| `outputs/hyperparameter_results.json` | Full grid search results in JSON format |
| `outputs/sarnet_best_weights.pt` | Best model weights from hyperparameter tuning |
| `outputs/final_cm_hog_svm.png` | Final HOG+SVM confusion matrix |
| `outputs/final_cm_cnn_raw.png` | Final CNN no-pipeline confusion matrix |
| `outputs/final_cm_cnn_tuned.png` | Final CNN tuned confusion matrix |
| `outputs/final_comparison_curves.png` | Learning curves comparing all three variants |
| `outputs/final_results.json` | Final evaluation results in JSON format |

---

## References

1. Lee, J.S. (1980). Digital image enhancement and noise filtering by use of local statistics. *IEEE TPAMI*.
2. Chen, S. et al. (2016). Target classification using deep convolutional networks for SAR images (AConvNet). *IEEE TGRS*.
3. Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple target situations. *IEEE TAES*.
4. Hu, J. et al. (2018). Squeeze-and-excitation networks. *IEEE CVPR*.

---