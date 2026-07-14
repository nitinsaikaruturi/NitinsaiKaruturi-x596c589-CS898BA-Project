"""
train_cnn.py
============
CS-898BA — Recognizing Vehicles in Radar Surveillance Images
Nitinsai Karuturi | X596C589

Step 4: Lightweight CNN classifier for MSTAR 3-class recognition.
Architecture: 3 conv blocks + 2 FC layers (inspired by AConvNet).
Trains on CPU — runs in ~8-15 min for 10 epochs at 64x64 input.
Saves training curves, confusion matrix, and final accuracy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

OUTPUT_DIR  = "outputs"
CLASS_NAMES = ["BMP2", "BTR70", "T72"]
EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 0.001
DROPOUT     = 0.5
SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class MSTARDataset(Dataset):
    def __init__(self, x, y):
        # x: (N, H, W, 1) float32 in [0,1]
        self.x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ── Model ─────────────────────────────────────────────────────────────────────
class SARNet(nn.Module):
    """
    Lightweight CNN for SAR chip classification (AConvNet-inspired).
    Input: (N, 1, 64, 64)
    """
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                             # → 32x32
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                             # → 16x16
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                             # → 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading preprocessed data …")
d = np.load("data_preprocessed.npz")
x_train, y_train = d["x_train"], d["y_train"]
x_test,  y_test  = d["x_test"],  d["y_test"]

train_loader = DataLoader(MSTARDataset(x_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(MSTARDataset(x_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── Train ─────────────────────────────────────────────────────────────────────
device = torch.device("cpu")
model  = SARNet(num_classes=3, dropout=DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining SARNet for {EPOCHS} epochs on CPU …")
print(f"  Batch size: {BATCH_SIZE}  |  LR: {LR}  |  Dropout: {DROPOUT}")
print(f"  Train samples: {len(x_train)}  |  Test samples: {len(x_test)}\n")

train_losses, train_accs, test_accs = [], [], []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total   += len(yb)

    scheduler.step()
    train_loss = running_loss / total
    train_acc  = correct / total * 100

    # Test accuracy
    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            tc += (out.argmax(1) == yb.to(device)).sum().item()
            tt += len(yb)
    test_acc = tc / tt * 100

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    elapsed = time.time() - t0
    print(f"  Epoch {epoch:02d}/{EPOCHS}  "
          f"loss={train_loss:.4f}  "
          f"train_acc={train_acc:.1f}%  "
          f"test_acc={test_acc:.1f}%  "
          f"({elapsed:.0f}s)")

# ── Final evaluation ──────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

final_acc = accuracy_score(all_labels, all_preds)
report    = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
cm        = confusion_matrix(all_labels, all_preds)

print(f"\n=== CNN Final Results ===")
print(f"Final test accuracy : {final_acc * 100:.1f}%")
print(f"\nClassification report:\n{report}")
print(f"Confusion matrix:\n{cm}")

# ── Save training curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0B1F3A")

epochs_x = range(1, EPOCHS + 1)
ax1.plot(epochs_x, train_losses, color="#2FE6C7", linewidth=2)
ax1.set_title("Training Loss", color="white", fontsize=13)
ax1.set_xlabel("Epoch", color="#CADCFC")
ax1.set_ylabel("Loss",  color="#CADCFC")
ax1.set_facecolor("#16314F")
ax1.tick_params(colors="white")
for sp in ax1.spines.values(): sp.set_color("#44546A")

ax2.plot(epochs_x, train_accs, color="#2FE6C7",  linewidth=2, label="Train")
ax2.plot(epochs_x, test_accs,  color="#F0A500",  linewidth=2, label="Test",  linestyle="--")
ax2.set_title("Accuracy", color="white", fontsize=13)
ax2.set_xlabel("Epoch",   color="#CADCFC")
ax2.set_ylabel("Acc %",   color="#CADCFC")
ax2.legend(facecolor="#16314F", labelcolor="white")
ax2.set_facecolor("#16314F")
ax2.tick_params(colors="white")
for sp in ax2.spines.values(): sp.set_color("#44546A")

plt.suptitle("SARNet — Training History", color="white", fontsize=14)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "cnn_training_curves.png")
plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
plt.close()
print(f"Training curves saved: {path}")

# ── Save CNN confusion matrix ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B1F3A")
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=100)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
ax.set_xlabel("Predicted", color="#2FE6C7", fontsize=12)
ax.set_ylabel("Actual",    color="#2FE6C7", fontsize=12)
ax.set_title(f"CNN Confusion Matrix\n(Overall Acc: {final_acc*100:.1f}%)",
             color="white", fontsize=13)
for i in range(3):
    for j in range(3):
        val = cm_norm[i, j]
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=14,
                color="black" if val > 50 else "white", fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "confusion_cnn.png")
plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
plt.close()
print(f"CNN confusion matrix saved: {path}")

# ── Save model + summary ──────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "sarnet_weights.pt"))
np.save(os.path.join(OUTPUT_DIR, "cnn_results.npy"),
        {"cnn_accuracy": round(final_acc * 100, 1), "cnn_cm": cm.tolist()})

print(f"\nCNN accuracy: {final_acc*100:.1f}%")
print("\ntrain_cnn.py complete.\n")
