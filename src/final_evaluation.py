"""
final_evaluation.py
===================
CS-898BA — Recognizing Vehicles in Radar Surveillance Images
Nitinsai Karuturi | X596C589

Final evaluation comparing three pipeline variants:
  1. HOG + SVM (classical baseline)
  2. CNN without preprocessing pipeline (raw chips)
  3. CNN with full preprocessing pipeline (best tuned config)

Produces: comparison table, confusion matrices, per-class metrics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage import uniform_filter, binary_opening, binary_closing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

OUTPUT_DIR  = "outputs"
CLASS_NAMES = ["BMP2", "BTR70", "T72"]
SEED        = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Best hyperparameters from tuning ─────────────────────────────────────────
BEST_LR       = 0.0001
BEST_BATCH    = 16
BEST_DROPOUT  = 0.3
BEST_EPOCHS   = 15   # more epochs with best config for final model

# ── Dataset class ─────────────────────────────────────────────────────────────
class MSTARDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ── SARNet model ──────────────────────────────────────────────────────────────
class SARNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

# ── Preprocessing pipeline ────────────────────────────────────────────────────
def lee_filter(chip, window=5):
    chip   = chip.astype(np.float64)
    mean   = uniform_filter(chip, window)
    sq_mean= uniform_filter(chip**2, window)
    local_var  = sq_mean - mean**2
    noise_var  = max(np.var(chip), 1e-8)
    weight     = local_var / (local_var + noise_var)
    return mean + weight * (chip - mean)

def cfar_segment(chip, percentile=95):
    threshold = np.percentile(chip, percentile)
    return (chip >= threshold).astype(np.uint8)

def morph_cleanup(mask):
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=1)
    return mask.astype(np.uint8)

def centre_crop(chip, size=64):
    h, w = chip.shape
    cy, cx = h//2, w//2
    half = size//2
    r0, r1 = max(cy-half, 0), min(cy+half, h)
    c0, c1 = max(cx-half, 0), min(cx+half, w)
    cropped = chip[r0:r1, c0:c1]
    if cropped.shape != (size, size):
        out = np.zeros((size, size), dtype=chip.dtype)
        out[:cropped.shape[0], :cropped.shape[1]] = cropped
        return out
    return cropped

def preprocess_chip(raw_chip, crop_size=64):
    chip = raw_chip.squeeze().astype(np.float64)
    chip = lee_filter(chip)
    chip = centre_crop(chip, size=crop_size)
    a, b = chip.min(), chip.max()
    return (chip - a) / (b - a + 1e-8)

def preprocess_raw_only(raw_chip, crop_size=64):
    """No Lee filter — just resize raw chip for CNN-no-pipeline variant."""
    chip = raw_chip.squeeze().astype(np.float64)
    chip = centre_crop(chip, size=crop_size)
    a, b = chip.min(), chip.max()
    return (chip - a) / (b - a + 1e-8)

# ── Train CNN ─────────────────────────────────────────────────────────────────
def train_cnn(x_tr, y_tr, x_te, y_te, lr, batch, dropout, epochs, label):
    loader_tr = DataLoader(MSTARDataset(x_tr, y_tr), batch_size=batch, shuffle=True)
    loader_te = DataLoader(MSTARDataset(x_te, y_te), batch_size=batch, shuffle=False)
    model     = SARNet(dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    curves    = []

    print(f"\n  Training {label} ({epochs} epochs) ...")
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in loader_tr:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader_te:
                preds.extend(model(xb).argmax(1).numpy())
                labels.extend(yb.numpy())
        acc = accuracy_score(labels, preds) * 100
        curves.append(acc)
        print(f"    Epoch {epoch:02d}/{epochs}  test_acc={acc:.1f}%")

    return model, curves

# ── Evaluate and return metrics ───────────────────────────────────────────────
def evaluate(model_or_clf, x_te, y_te, is_svm=False, batch=32):
    if is_svm:
        y_pred = model_or_clf.predict(x_te)
    else:
        loader = DataLoader(MSTARDataset(x_te, y_te), batch_size=batch, shuffle=False)
        model_or_clf.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in loader:
                preds.extend(model_or_clf(xb).argmax(1).numpy())
        y_pred = np.array(preds)

    acc  = accuracy_score(y_te, y_pred) * 100
    cm   = confusion_matrix(y_te, y_pred)
    rep  = classification_report(y_te, y_pred, target_names=CLASS_NAMES, output_dict=True)
    return acc, cm, rep, y_pred

# ── Confusion matrix figure ───────────────────────────────────────────────────
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B1F3A")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_xlabel("Predicted", color="#2FE6C7", fontsize=12)
    ax.set_ylabel("Actual",    color="#2FE6C7", fontsize=12)
    ax.set_title(title, color="white", fontsize=13)
    for i in range(3):
        for j in range(3):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="black" if val > 50 else "white")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
    plt.close()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load raw data ─────────────────────────────────────────────────────────
    print("Loading data ...")
    d = np.load("data.npz")
    x_train_raw, y_train = d["x_train"], d["y_train"]
    x_test_raw,  y_test  = d["x_test"],  d["y_test"]

    # ── Prepare three variants ────────────────────────────────────────────────
    print("Preparing datasets ...")

    # Variant A — HOG features from preprocessed chips
    def make_hog(x):
        feats = []
        for chip in x:
            p = preprocess_chip(chip)
            f = hog(p, orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), transform_sqrt=True, feature_vector=True)
            feats.append(f)
        return np.array(feats)

    print("  Extracting HOG features ...")
    X_hog_train = make_hog(x_train_raw)
    X_hog_test  = make_hog(x_test_raw)
    scaler = StandardScaler()
    X_hog_train = scaler.fit_transform(X_hog_train)
    X_hog_test  = scaler.transform(X_hog_test)

    # Variant B — CNN on raw chips (no pipeline)
    def make_raw_dataset(x, size=64):
        out = np.zeros((len(x), size, size, 1), dtype=np.float32)
        for i, chip in enumerate(x):
            out[i,:,:,0] = preprocess_raw_only(chip, size)
        return out

    print("  Preparing raw chips for CNN ...")
    x_raw_tr = make_raw_dataset(x_train_raw)
    x_raw_te = make_raw_dataset(x_test_raw)

    # Variant C — CNN on preprocessed chips (full pipeline, best config)
    print("  Loading preprocessed chips ...")
    dp = np.load("data_preprocessed.npz")
    x_pre_tr, x_pre_te = dp["x_train"], dp["x_test"]

    # ── MODEL A: HOG + SVM ────────────────────────────────────────────────────
    print("\n[1/3] HOG + SVM ...")
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              decision_function_shape="ovo", random_state=SEED)
    svm.fit(X_hog_train, y_train)
    acc_hog, cm_hog, rep_hog, _ = evaluate(svm, X_hog_test, y_test, is_svm=True)
    print(f"  HOG+SVM accuracy: {acc_hog:.1f}%")
    plot_cm(cm_hog, f"HOG+SVM  ({acc_hog:.1f}%)",
            os.path.join(OUTPUT_DIR, "final_cm_hog_svm.png"))

    # ── MODEL B: CNN no pipeline ──────────────────────────────────────────────
    print("\n[2/3] CNN — no preprocessing pipeline ...")
    cnn_raw, curves_raw = train_cnn(
        x_raw_tr, y_train, x_raw_te, y_test,
        lr=BEST_LR, batch=BEST_BATCH, dropout=BEST_DROPOUT,
        epochs=BEST_EPOCHS, label="CNN (raw chips)"
    )
    acc_raw, cm_raw, rep_raw, _ = evaluate(cnn_raw, x_raw_te, y_test, batch=BEST_BATCH)
    print(f"  CNN (raw) accuracy: {acc_raw:.1f}%")
    plot_cm(cm_raw, f"CNN No Pipeline  ({acc_raw:.1f}%)",
            os.path.join(OUTPUT_DIR, "final_cm_cnn_raw.png"))

    # ── MODEL C: CNN full pipeline best config ────────────────────────────────
    print("\n[3/3] CNN — full pipeline + best hyperparameters ...")
    cnn_best, curves_best = train_cnn(
        x_pre_tr, y_train, x_pre_te, y_test,
        lr=BEST_LR, batch=BEST_BATCH, dropout=BEST_DROPOUT,
        epochs=BEST_EPOCHS, label="CNN (pipeline + tuned)"
    )
    acc_best, cm_best, rep_best, _ = evaluate(cnn_best, x_pre_te, y_test, batch=BEST_BATCH)
    print(f"  CNN (tuned) accuracy: {acc_best:.1f}%")
    plot_cm(cm_best, f"CNN Pipeline + Tuned  ({acc_best:.1f}%)",
            os.path.join(OUTPUT_DIR, "final_cm_cnn_tuned.png"))

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    models = [
        ("HOG + SVM",              acc_hog,  rep_hog),
        ("CNN — No Pipeline",      acc_raw,  rep_raw),
        ("CNN — Pipeline + Tuned", acc_best, rep_best),
    ]
    print(f"{'Model':<28} {'Acc%':<8} {'BMP2 F1':<10} {'BTR70 F1':<10} {'T72 F1'}")
    print("-"*60)
    for name, acc, rep in models:
        b  = rep["BMP2"]["f1-score"]
        bt = rep["BTR70"]["f1-score"]
        t  = rep["T72"]["f1-score"]
        print(f"{name:<28} {acc:<8.1f} {b:<10.2f} {bt:<10.2f} {t:.2f}")

    # ── Learning curves comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0B1F3A")
    ax.plot(range(1, BEST_EPOCHS+1), curves_raw,
            color="#F0A500", linewidth=2, linestyle="--", label="CNN — No Pipeline")
    ax.plot(range(1, BEST_EPOCHS+1), curves_best,
            color="#2FE6C7", linewidth=2, label="CNN — Pipeline + Tuned")
    ax.axhline(y=acc_hog, color="#CADCFC", linewidth=1.5,
               linestyle=":", label=f"HOG+SVM baseline ({acc_hog:.1f}%)")
    ax.set_xlabel("Epoch", color="#CADCFC", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", color="#CADCFC", fontsize=12)
    ax.set_title("Final Model Comparison — Learning Curves",
                 color="white", fontsize=14)
    ax.legend(facecolor="#16314F", labelcolor="white", fontsize=10)
    ax.set_facecolor("#16314F")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#44546A")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "final_comparison_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
    plt.close()
    print(f"\nLearning curves saved: {path}")

    # ── Save final results JSON ───────────────────────────────────────────────
    final_results = {
        "hog_svm":       {"accuracy": round(acc_hog, 1),
                          "BMP2_f1":  round(rep_hog["BMP2"]["f1-score"], 2),
                          "BTR70_f1": round(rep_hog["BTR70"]["f1-score"], 2),
                          "T72_f1":   round(rep_hog["T72"]["f1-score"], 2)},
        "cnn_no_pipeline": {"accuracy": round(acc_raw, 1),
                            "BMP2_f1":  round(rep_raw["BMP2"]["f1-score"], 2),
                            "BTR70_f1": round(rep_raw["BTR70"]["f1-score"], 2),
                            "T72_f1":   round(rep_raw["T72"]["f1-score"], 2)},
        "cnn_tuned":     {"accuracy": round(acc_best, 1),
                          "BMP2_f1":  round(rep_best["BMP2"]["f1-score"], 2),
                          "BTR70_f1": round(rep_best["BTR70"]["f1-score"], 2),
                          "T72_f1":   round(rep_best["T72"]["f1-score"], 2)},
        "best_hyperparams": {
            "lr": BEST_LR, "batch_size": BEST_BATCH, "dropout": BEST_DROPOUT
        }
    }
    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved: outputs/final_results.json")
    print("\nfinal_evaluation.py complete.\n")
