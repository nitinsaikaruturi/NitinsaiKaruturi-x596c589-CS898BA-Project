"""
hyperparameter_tuning.py
========================
CS-898BA — Recognizing Vehicles in Radar Surveillance Images
Nitinsai Karuturi | X596C589

Hyperparameter grid search for SARNet CNN.
Sweeps: LR x Batch Size x Dropout
Saves best model weights and full results table to outputs/.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time, json

OUTPUT_DIR  = "outputs"
CLASS_NAMES = ["BMP2", "BTR70", "T72"]
EPOCHS      = 10
SEED        = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameter Grid ───────────────────────────────────────────────────────
GRID = {
    "lr":          [0.01, 0.001, 0.0001],
    "batch_size":  [16, 32],
    "dropout":     [0.3, 0.5],
}

# ── Dataset ───────────────────────────────────────────────────────────────────
class MSTARDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ── Model ─────────────────────────────────────────────────────────────────────
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

# ── Train one config ──────────────────────────────────────────────────────────
def train_one(x_train, y_train, x_test, y_test, lr, batch_size, dropout):
    train_loader = DataLoader(MSTARDataset(x_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(MSTARDataset(x_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    model     = SARNet(dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    test_accs = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds.extend(model(xb).argmax(1).numpy())
                labels.extend(yb.numpy())
        acc = accuracy_score(labels, preds) * 100
        test_accs.append(acc)

    best_acc = max(test_accs)
    final_acc = test_accs[-1]
    return model, best_acc, final_acc, test_accs

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading preprocessed data ...")
    d = np.load("data_preprocessed.npz")
    x_train, y_train = d["x_train"], d["y_train"]
    x_test,  y_test  = d["x_test"],  d["y_test"]

    # Build all combinations
    configs = [
        {"lr": lr, "batch_size": bs, "dropout": do}
        for lr in GRID["lr"]
        for bs in GRID["batch_size"]
        for do in GRID["dropout"]
    ]

    total = len(configs)
    print(f"\nRunning {total} configurations ({EPOCHS} epochs each) ...\n")
    print(f"{'#':<4} {'LR':<8} {'Batch':<7} {'Drop':<6} {'Best%':<8} {'Final%':<8} {'Time'}")
    print("-" * 55)

    results = []
    best_acc   = 0
    best_cfg   = None
    best_model = None
    all_curves = []

    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        model, best, final, curve = train_one(
            x_train, y_train, x_test, y_test,
            lr=cfg["lr"], batch_size=cfg["batch_size"], dropout=cfg["dropout"]
        )
        elapsed = time.time() - t0

        print(f"{i:<4} {cfg['lr']:<8} {cfg['batch_size']:<7} {cfg['dropout']:<6} "
              f"{best:<8.1f} {final:<8.1f} {elapsed:.0f}s")

        results.append({
            "config": i,
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "dropout": cfg["dropout"],
            "best_acc": round(best, 2),
            "final_acc": round(final, 2)
        })
        all_curves.append((cfg, curve))

        if best > best_acc:
            best_acc   = best
            best_cfg   = cfg
            best_model = model

    # ── Save best model ───────────────────────────────────────────────────────
    torch.save(best_model.state_dict(),
               os.path.join(OUTPUT_DIR, "sarnet_best_weights.pt"))

    # ── Save results JSON ─────────────────────────────────────────────────────
    summary = {
        "best_config": best_cfg,
        "best_accuracy": round(best_acc, 2),
        "all_results": results
    }
    with open(os.path.join(OUTPUT_DIR, "hyperparameter_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("BEST CONFIGURATION")
    print("=" * 55)
    print(f"  Learning rate : {best_cfg['lr']}")
    print(f"  Batch size    : {best_cfg['batch_size']}")
    print(f"  Dropout       : {best_cfg['dropout']}")
    print(f"  Best accuracy : {best_acc:.1f}%")

    # ── Plot: all configs bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0B1F3A")
    labels  = [f"LR={r['lr']}\nB={r['batch_size']}\nD={r['dropout']}"
               for r in results]
    accs    = [r["best_acc"] for r in results]
    colors  = ["#2FE6C7" if a == max(accs) else "#16314F" for a in accs]

    bars = ax.bar(range(len(accs)), accs, color=colors,
                  edgecolor="#2FE6C7", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color="white", fontsize=8)
    ax.set_ylabel("Best Test Accuracy (%)", color="#CADCFC", fontsize=11)
    ax.set_title("Hyperparameter Grid Search — All Configurations",
                 color="white", fontsize=13)
    ax.set_facecolor("#16314F")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 105)
    for sp in ax.spines.values(): sp.set_color("#44546A")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom",
                color="white", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "hyperparameter_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
    plt.close()
    print(f"\nBar chart saved: {path}")

    # ── Plot: top 3 learning curves ───────────────────────────────────────────
    sorted_results = sorted(results, key=lambda r: r["best_acc"], reverse=True)
    top3_idxs = [r["config"] - 1 for r in sorted_results[:3]]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0B1F3A")
    colors_top = ["#2FE6C7", "#F0A500", "#CADCFC"]
    for rank, idx in enumerate(top3_idxs):
        cfg, curve = all_curves[idx]
        lbl = f"LR={cfg['lr']} B={cfg['batch_size']} D={cfg['dropout']}"
        ax.plot(range(1, EPOCHS+1), curve,
                color=colors_top[rank], linewidth=2,
                label=lbl, linestyle=["-","--","-."][rank])

    ax.set_xlabel("Epoch", color="#CADCFC", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", color="#CADCFC", fontsize=11)
    ax.set_title("Top 3 Configurations — Learning Curves",
                 color="white", fontsize=13)
    ax.legend(facecolor="#16314F", labelcolor="white", fontsize=9)
    ax.set_facecolor("#16314F")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#44546A")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "top3_learning_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0B1F3A")
    plt.close()
    print(f"Top 3 curves saved: {path}")
    print("\nhyperparameter_tuning.py complete.\n")
