"""
preprocess.py
=============
CS-898BA — Recognizing Vehicles in Radar Surveillance Images
Nitinsai Karuturi | X596C589

Step 2: Domain-engineering preprocessing pipeline.
  1. Lee adaptive filter  — removes multiplicative speckle noise
  2. CFAR segmentation    — isolates target from background clutter
  3. Morphological cleanup — removes stray noise pixels from mask
  4. Standardised crop    — centres and crops to fixed 64x64 region

Saves before/after visualisations and returns cleaned arrays.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, binary_opening, binary_closing, label
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["BMP2", "BTR70", "T72"]

# ── 1. Lee Adaptive Filter ────────────────────────────────────────────────────
def lee_filter(chip, window=5):
    """
    Adaptive Lee filter for multiplicative SAR speckle.
    Smoothing strength adapts per-pixel based on local variance:
      filtered = mean + weight * (pixel - mean)
    where weight = local_var / (local_var + noise_var)
    """
    chip   = chip.astype(np.float64)
    mean   = uniform_filter(chip, window)
    sq_mean= uniform_filter(chip ** 2, window)
    local_var  = sq_mean - mean ** 2
    noise_var  = max(np.var(chip), 1e-8)
    weight     = local_var / (local_var + noise_var)
    return mean + weight * (chip - mean)

# ── 2. Adaptive CFAR Segmentation ─────────────────────────────────────────────
def cfar_segment(chip, percentile=95):
    """
    Percentile-based constant false-alarm rate threshold.
    Pixels above the threshold are labelled as 'target'.
    Using percentile (not fixed multiplier) makes it robust
    across chips with very different clutter-to-target ratios.
    """
    threshold = np.percentile(chip, percentile)
    return (chip >= threshold).astype(np.uint8)

# ── 3. Morphological Cleanup ──────────────────────────────────────────────────
def morph_cleanup(mask):
    """
    Binary opening removes stray isolated noise pixels.
    Binary closing fills small gaps in the target mask.
    """
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=1)
    return mask.astype(np.uint8)

# ── 4. Standardised Centre Crop ──────────────────────────────────────────────
def centre_crop(chip, size=64):
    """
    Centre-crop to (size x size) around the image centre.
    Removes peripheral clutter while keeping the target region.
    """
    h, w = chip.shape
    cy, cx = h // 2, w // 2
    half = size // 2
    r0, r1 = max(cy - half, 0), min(cy + half, h)
    c0, c1 = max(cx - half, 0), min(cx + half, w)
    cropped = chip[r0:r1, c0:c1]
    # Pad if chip was too small
    if cropped.shape != (size, size):
        out = np.zeros((size, size), dtype=chip.dtype)
        out[:cropped.shape[0], :cropped.shape[1]] = cropped
        return out
    return cropped

# ── Full pipeline ─────────────────────────────────────────────────────────────
def preprocess_chip(raw_chip, crop_size=64):
    chip = raw_chip.squeeze().astype(np.float64)
    chip = lee_filter(chip)
    mask = cfar_segment(chip)
    mask = morph_cleanup(mask)
    chip = centre_crop(chip, size=crop_size)
    a    = chip.min(); b = chip.max()
    chip = (chip - a) / (b - a + 1e-8)
    return chip

def preprocess_dataset(x, crop_size=64):
    out = np.zeros((len(x), crop_size, crop_size, 1), dtype=np.float32)
    for i, chip in enumerate(x):
        out[i, :, :, 0] = preprocess_chip(chip, crop_size)
    return out

# ── Visualisation helpers ─────────────────────────────────────────────────────
def norm_display(arr):
    a = arr.astype(np.float64).squeeze()
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def save_pipeline_figure(raw_chip, out_path):
    """Save a 4-panel before→after figure for one chip."""
    raw  = raw_chip.squeeze().astype(np.float64)
    lee  = lee_filter(raw)
    mask = morph_cleanup(cfar_segment(lee))
    crop = centre_crop(lee)
    crop = (crop - crop.min()) / (crop.max() - crop.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5),
                             facecolor="#0B1F3A")
    titles = ["1. Raw chip\n(speckle noise)",
              "2. Lee filter\n(despeckling)",
              "3. CFAR mask\n(segmentation)",
              "4. Cropped & normalised\n(model input)"]
    images = [norm_display(raw), norm_display(lee),
              mask.astype(float), crop]
    cmaps  = ["gray", "gray", "hot", "gray"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, color="#2FE6C7", fontsize=10, pad=6)
        ax.axis("off")

    plt.suptitle("Preprocessing Pipeline — MSTAR Chip",
                 color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#0B1F3A")
    plt.close()
    print(f"  Saved pipeline figure: {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    d = np.load("data.npz")
    x_train, y_train = d["x_train"], d["y_train"]
    x_test,  y_test  = d["x_test"],  d["y_test"]

    # Save one pipeline figure per class
    for c, name in enumerate(CLASS_NAMES):
        idx = np.where(y_train == c)[0][0]
        save_pipeline_figure(x_train[idx],
                             os.path.join(OUTPUT_DIR, f"pipeline_{name}.png"))

    # Run full preprocessing on train + test sets
    print("\nPreprocessing training set …")
    x_train_p = preprocess_dataset(x_train, crop_size=64)
    print("Preprocessing test set …")
    x_test_p  = preprocess_dataset(x_test,  crop_size=64)

    # Save preprocessed arrays for use by downstream scripts
    np.savez("data_preprocessed.npz",
             x_train=x_train_p, y_train=y_train,
             x_test=x_test_p,   y_test=y_test)

    print(f"\nPreprocessed shapes:")
    print(f"  x_train : {x_train_p.shape}")
    print(f"  x_test  : {x_test_p.shape}")
    print("\npreprocess.py complete.\n")
