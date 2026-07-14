"""
baseline_hog_svm.py
===================
CS-898BA — Recognizing Vehicles in Radar Surveillance Images
Nitinsai Karuturi | X596C589

Step 3: Classical baseline — HOG features + RBF-SVM classifier.
Runs on preprocessed chips (output of preprocess.py).
Prints accuracy, per-class F1, and saves a confusion matrix figure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

OUTPUT_DIR  = "outputs"
CLASS_NAMES = ["BMP2", "BTR70", "T72"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
print("Loading preprocessed data …")
d = np.load("data_preprocessed.npz")
x_train, y_train = d["x_train"], d["y_train"]
x_test,  y_test  = d["x_test"],  d["y_test"]

# ── Extract HOG features ──────────────────────────────────────────────────────
def extract_hog(x):
    feats = []
    for chip in x:
        img = chip.squeeze()
        f = hog(img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                transform_sqrt=True,
                feature_vector=True)
        feats.append(f)
    return np.array(feats)

print("Extracting HOG features from training set …")
X_train_hog = extract_hog(x_train)
print("Extracting HOG features from test set …")
X_test_hog  = extract_hog(x_test)
print(f"  HOG feature vector length: {X_train_hog.shape[1]}")

# ── Normalise features ────────────────────────────────────────────────────────
scaler      = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog  = scaler.transform(X_test_hog)

# ── Train SVM ─────────────────────────────────────────────────────────────────
print("\nTraining RBF-SVM …")
svm = SVC(kernel="rbf", C=10, gamma="scale",
          decision_function_shape="ovo", random_state=42)
svm.fit(X_train_hog, y_train)
print("  Done.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred    = svm.predict(X_test_hog)
acc       = accuracy_score(y_test, y_pred)
report    = classification_report(y_test, y_pred,
                                  target_names=CLASS_NAMES)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n=== HOG + SVM Results ===")
print(f"Overall accuracy : {acc * 100:.1f}%")
print(f"\nClassification report:\n{report}")
print(f"Confusion matrix:\n{cm}")

# ── Save confusion matrix figure ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B1F3A")
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=100)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
ax.set_xlabel("Predicted", color="#2FE6C7", fontsize=12)
ax.set_ylabel("Actual",    color="#2FE6C7", fontsize=12)
ax.set_title(f"HOG+SVM Confusion Matrix\n(Overall Acc: {acc*100:.1f}%)",
             color="white", fontsize=13)

for i in range(3):
    for j in range(3):
        val = cm_norm[i, j]
        ax.text(j, i, f"{val:.0f}%",
                ha="center", va="center", fontsize=14,
                color="black" if val > 50 else "white", fontweight="bold")

plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="white")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "confusion_hog_svm.png")
plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
plt.close()
print(f"\nConfusion matrix saved: {path}")

# ── Save summary for midterm deck update ─────────────────────────────────────
summary = {
    "hog_svm_accuracy": round(acc * 100, 1),
    "hog_svm_cm": cm.tolist()
}
np.save(os.path.join(OUTPUT_DIR, "hog_svm_results.npy"), summary)
print(f"\nHOG+SVM accuracy: {acc*100:.1f}%")
print("\nbaseline_hog_svm.py complete.\n")
