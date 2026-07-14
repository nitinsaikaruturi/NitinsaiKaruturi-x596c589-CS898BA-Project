# NitinsaiKaruturi-x596c589-CS898BA-Project

**Author:** Nitinsai Karuturi
**Course:** CS 898BA – Image Analysis and Computer Vision
**Project:** Recognizing Vehicles in Radar Surveillance Images

---

## Project Overview

This project implements a Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) pipeline that classifies military ground vehicles from radar imagery. The system combines classical SAR image processing (Lee adaptive filtering, CFAR segmentation, morphological cleanup) with a lightweight convolutional neural network to classify three vehicle types from the MSTAR benchmark dataset.

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

## Run the Project

Run scripts in this order from the project directory:

```bash
python load_data.py
python preprocess.py
python baseline_hog_svm.py
python train_cnn.py
```

---

## File Descriptions

| File | Purpose |
|---|---|
| `load_data.py` | Downloads MSTAR NPZ file, verifies class distribution, saves sample chips per class |
| `preprocess.py` | Applies Lee adaptive filter, CFAR segmentation, morphological cleanup, and 64×64 centre crop |
| `baseline_hog_svm.py` | Extracts HOG features and trains RBF-SVM classifier as the classical baseline |
| `train_cnn.py` | Defines and trains SARNet CNN for 10 epochs on CPU, saves weights and training curves |

---

## Pipeline
Raw SAR Chip (32×32)
│
▼
[1] Lee Adaptive Filter       — removes multiplicative speckle noise
│
▼
[2] CFAR Segmentation         — percentile-based threshold isolates target return
│
▼
[3] Morphological Cleanup     — binary opening/closing removes stray noise pixels
│
▼
[4] Standardised Centre Crop  — 64×64 crop centred on target region
│
├──▶ [5a] HOG + RBF-SVM    (classical baseline)
└──▶ [5b] SARNet CNN       (deep learning classifier)

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

---

## Results

| Model | Test Accuracy | BMP2 F1 | BTR70 F1 | T72 F1 |
|---|---|---|---|---|
| HOG + SVM | 91.4% | 0.86 | 0.89 | 0.99 |
| SARNet CNN (10 epochs) | 90.1% | 0.84 | 0.87 | 0.99 |

### CNN Training Curve

| Epoch | Train Acc | Test Acc |
|---|---|---|
| 1 | 46.6% | 33.3% |
| 4 | 86.2% | 59.6% |
| 6 | 94.8% | 91.2% |
| 9 | 96.8% | 95.6% (peak) |
| 10 | 97.8% | 90.1% |

---

## Model Architecture — SARNet
Input:        (N, 1, 64, 64)
Conv Block 1: Conv2d(1→32,   3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
Conv Block 2: Conv2d(32→64,  3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
Conv Block 3: Conv2d(64→128, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
Flatten:      128 × 8 × 8 = 8192
FC1:          Linear(8192→512) → ReLU → Dropout(0.5)
FC2:          Linear(512→128)  → ReLU → Dropout(0.5)
FC3:          Linear(128→3)    → softmax output
Optimizer:    Adam (lr=0.001, StepLR ×0.5 every 5 epochs)
Loss:         CrossEntropyLoss | Batch size: 32 | Epochs: 10

---

## Evaluation & Analysis

**HOG+SVM (91.4%):** HOG captures structural edge patterns from despeckled chips. The RBF-SVM separates classes well overall but cannot distinguish BMP2 from BTR70 reliably — 60 of 274 BMP2 test chips were misclassified as BTR70. Both are wheeled armored vehicles with overlapping SAR cross-section profiles at certain aspect angles, making the feature distributions inseparable in HOG space.

**SARNet CNN (90.1%):** The CNN learns hierarchical features directly from preprocessed chips. Performance at 10 epochs is slightly below HOG+SVM, which is expected for small datasets — CNNs require more epochs and hyperparameter tuning to surpass classical baselines. Peak test accuracy of 95.6% was achieved at epoch 9, after which the final-epoch drop indicates that early stopping or learning rate decay adjustment would improve generalisation. The same BMP2↔BTR70 confusion pattern persists, confirming this is a dataset-level challenge rather than a model-specific failure.

**T72 performance:** T72 achieves F1 = 0.99 in both models. The tank's distinctive turret shape and track width produce a SAR signature clearly different from both wheeled vehicles, making it reliably separable regardless of classifier.

**Impact of preprocessing:** Lee filtering and CFAR segmentation are necessary for viable accuracy in both classifiers. The adaptive despeckling preserves bright target returns while suppressing background clutter, directly improving HOG feature quality and CNN gradient stability during early training.

**Best performing method:** HOG+SVM at 91.4% narrowly leads the CNN at 10 epochs. With extended training and hyperparameter tuning, the CNN is expected to surpass this baseline — consistent with the AConvNet literature reporting high-90s accuracy on the same dataset.

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
| `outputs/confusion_cnn.png` | CNN confusion matrix |
| `outputs/cnn_training_curves.png` | Loss and accuracy curves over 10 epochs |
| `outputs/sarnet_weights.pt` | Saved SARNet model weights |

---

## References

1. Lee, J.S. (1980). Digital image enhancement and noise filtering by use of local statistics. *IEEE TPAMI*.
2. Chen, S. et al. (2016). Target classification using deep convolutional networks for SAR images (AConvNet). *IEEE TGRS*.
3. Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple target situations. *IEEE TAES*.
4. Hu, J. et al. (2018). Squeeze-and-excitation networks. *IEEE CVPR*.

---

## AI Usage

See AI_Log.md for full AI usage tracking.