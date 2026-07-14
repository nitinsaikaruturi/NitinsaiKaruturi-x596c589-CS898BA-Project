# AI Usage Log

**Course:** CS-898BA — Image Analysis and Computer Vision
**Institution:** Wichita State University — Summer 2026
**Student:** Nitinsai Karuturi | WSU ID: X596C589

---

## Project Pitch Session — June 10, 2026

### Entry 1: Pipeline Design and Literature Review

**Tool:** Claude (claude.ai) **Prompt:** Asked Claude to define the full preprocessing pipeline, propose an appropriate CNN architecture, outline a hyperparameter tuning plan, and map the existing SAR ATR literature from classical to deep learning approaches. **Response synopsis:** Claude proposed a 4-step domain-engineering pipeline: (1) Lee adaptive filter for multiplicative speckle noise removal, (2) CFAR segmentation using percentile-based thresholding to isolate target returns from background clutter, (3) morphological opening/closing to clean the binary mask, (4) standardised 64×64 centre crop. Architecture: AConvNet-inspired 3-block CNN. Literature structured across 4 eras — template matching (1990s), HOG/SVM classifiers (2000s), AConvNet-style CNNs (2016+), and attention-based CNNs and ViTs (2020s). Hyperparameter grid: LR ∈ {0.01, 0.001, 0.0001}, batch ∈ {16, 32}, dropout ∈ {0.3, 0.5}. **Design change:** Established full technical pipeline and architecture. Literature review used in proposal slides.

---

### Entry 2: Alternative Design Justification

**Tool:** Claude (claude.ai) **Prompt:** Asked Claude to evaluate alternative architectures (HOG+SVM, attention-based CNN, Vision Transformer) against the chosen lightweight CNN and produce a formal justification. **Response synopsis:** HOG+SVM is interpretable but accuracy-capped by handcrafted features — retained as classical baseline only. Attention CNN (EMC2A-Net style) is technically stronger but adds multi-branch complexity disproportionate to a CPU-only, small-dataset setting. ViT requires large data and pretraining to converge — unsuitable for 232 samples/class on CPU. Custom lightweight CNN balances accuracy, CPU-feasibility, and dataset size. **Design change:** Used as Choice Justification and Alternative Design Options sections in the proposal presentation.

---

## Midterm Development Session — June 21, 2026

### Entry 3: Implementation of Four Pipeline Scripts

**Tool:** Claude (claude.ai) **Prompt:** Asked Claude to implement load_data.py, preprocess.py, baseline_hog_svm.py, and train_cnn.py for the full MSTAR ATR pipeline with inline documentation and output figures. **Response synopsis:** Claude implemented all four scripts. `load_data.py` downloads the MSTAR NPZ file and verifies 696 train / 822 test samples (232 train, 274 test per class). `preprocess.py` applies Lee adaptive filter (window=5, local variance weighting), percentile-based CFAR threshold (95th percentile per chip), binary morphological opening then closing (1 iteration each), and 64×64 centre crop; saves 4-panel pipeline figures per class. `baseline_hog_svm.py` extracts HOG features (9 orientations, 8×8 pixels/cell, 2×2 cells/block) and trains an RBF-SVM (C=10, gamma=scale, one-vs-one). `train_cnn.py` defines SARNet (3 Conv-BN-ReLU-MaxPool blocks → FC 512 → FC 128 → FC 3), trains with Adam (lr=0.001) and StepLR decay (γ=0.5 every 5 epochs), saves weights and training curves. **Design change:** All four scripts added to repository via four separate incremental commits.

---

### Entry 4: Script Execution and Results

**Tool:** Claude (claude.ai) **Prompt:** Executed all four scripts and reported real accuracy metrics for inclusion in the midterm presentation. **Response synopsis:** HOG+SVM: 91.4% overall test accuracy (BMP2 F1=0.86, BTR70 F1=0.89, T72 F1=0.99); primary failure mode is BMP2↔BTR70 confusion (60 of 274 BMP2 test chips misclassified as BTR70 due to similar radar cross-sections at overlapping aspect angles). SARNet CNN at 10 epochs: 90.1% overall (BMP2 F1=0.84, BTR70 F1=0.87, T72 F1=0.99); peak test accuracy 95.6% at epoch 9 before final-epoch drop indicating early stopping would improve generalisation. Outputs saved: confusion matrices, training curves (loss + accuracy), model weights (sarnet_weights.pt). **Design change:** Midterm slide deck updated with real accuracy numbers. Confusion matrix values updated from estimated to actual.

---

### Entry 5: Midterm Presentation Generation

**Tool:** Claude (claude.ai) **Prompt:** Asked Claude to generate a 10-slide midterm progress report presentation covering all four required CS-898BA sections using real MSTAR chip images and matching the proposal visual theme. **Response synopsis:** Claude built a 10-slide pptxgenjs deck: title slide with real MSTAR montage background and radar reticle, agenda, condensed lit review (Lee filter, AConvNet, CFAR retained; ViT dropped with justification), 4-step pipeline overview with flow bar, pipeline demo slide with real chip before/after despeckling and all 3 class samples, HOG+SVM results (91.4% overall, per-class F1), SARNet CNN results (90.1% overall, epoch training history), confusion matrix analysis with BMP2↔BTR70 error pattern highlighted, roadblocks and pivots (CPU training time, dataset label ambiguity, CFAR threshold sensitivity), and next steps summary. **Design change:** Submitted as Midterm Progress Report presentation with all placeholder numbers replaced by real executed results.

---

### Entry 6: Documentation Generation

**Tool:** Claude (claude.ai) **Prompt:** Asked Claude to write a comprehensive README.md and technical AI_Log.md following the CS-898BA documentation standard. **Response synopsis:** README produced covering: problem statement, dataset specification table, pipeline architecture diagram, installation instructions, step-by-step execution guide, results table with per-class F1 scores, CNN epoch-by-epoch training history, full SARNet architecture spec (layer dimensions, optimizer, loss, scheduler), repository file structure, and references. AI_Log restructured in session/entry narrative format with technical prompt descriptions, response synopses, and specific design changes only. **Design change:** README.md and AI_Log.md committed to repository.