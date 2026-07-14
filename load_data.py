"""
load_data.py
============
CS-898BA — Recognizing Vehicles in Radar Surveillance Images
Nitinsai Karuturi | X596C589

Step 1: Load and verify the MSTAR 3-class dataset.
Downloads the pre-packaged NPZ file if not already present.
Prints class distribution, image stats, and saves sample chips.
"""

import os, urllib.request
import numpy as np
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
DATA_URL   = "https://raw.githubusercontent.com/amirhoseinoveis/GradCAM-with-MSTAR/main/mstar_3cl_32by32.npz"
DATA_FILE  = "data.npz"
OUTPUT_DIR = "outputs"
CLASS_NAMES = ["BMP2", "BTR70", "T72"]   # alphabetical — matches label 0/1/2

# ── Download ─────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    print("Downloading MSTAR dataset …")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("  Done.")
else:
    print(f"Dataset file already exists: {DATA_FILE}")

# ── Load ─────────────────────────────────────────────────────────────────────
d = np.load(DATA_FILE)
x_train, y_train = d["x_train"], d["y_train"]
x_test,  y_test  = d["x_test"],  d["y_test"]

# ── Report ───────────────────────────────────────────────────────────────────
print("\n=== Dataset Summary ===")
print(f"Train images : {x_train.shape}  (N, H, W, C)")
print(f"Test  images : {x_test.shape}")
print(f"Pixel range  : {x_train.min():.3f} – {x_train.max():.3f}  (already normalised 0-1)")

print("\nPer-class counts:")
for c, name in enumerate(CLASS_NAMES):
    tr = (y_train == c).sum()
    te = (y_test  == c).sum()
    print(f"  {name:6s}  train={tr:4d}   test={te:4d}")

# ── Save one sample chip per class ───────────────────────────────────────────
def save_chip(arr, path, size=256):
    a = arr.squeeze().astype(np.float64)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    Image.fromarray((a * 255).astype(np.uint8), mode="L") \
         .resize((size, size), Image.LANCZOS) \
         .save(path)

for c, name in enumerate(CLASS_NAMES):
    idx = np.where(y_train == c)[0][0]
    path = os.path.join(OUTPUT_DIR, f"sample_{name}.png")
    save_chip(x_train[idx], path)
    print(f"  Saved sample chip: {path}")

print("\nload_data.py complete.\n")
