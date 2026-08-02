# AI Usage Log

**Course:** CS-898BA — Image Analysis and Computer Vision
**Institution:** Wichita State University — Summer 2026
**Student:** Nitinsai Karuturi | WSU ID: X596C589

---

## Project Pitch Session — June 10, 2026

### Entry 1: Pipeline Design and Literature Review

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to define the full preprocessing pipeline, propose an appropriate CNN architecture, outline a hyperparameter tuning plan, and map the existing SAR ATR literature from classical to deep learning approaches.

**Response synopsis:** Claude proposed a 4-step domain-engineering pipeline: (1) Lee adaptive filter for multiplicative speckle noise removal, (2) CFAR segmentation using percentile-based thresholding to isolate target returns from background clutter, (3) morphological opening/closing to clean the binary mask, (4) standardised 64×64 centre crop. Architecture: AConvNet-inspired 3-block CNN. Literature structured across 4 eras — template matching (1990s), HOG/SVM classifiers (2000s), AConvNet-style CNNs (2016+), and attention-based CNNs and ViTs (2020s). Hyperparameter grid: LR ∈ {0.01, 0.001, 0.0001}, batch ∈ {16, 32}, dropout ∈ {0.3, 0.5}.

**Design change:** Established full technical pipeline and architecture. Literature review used in proposal slides.

---

### Entry 2: Alternative Design Justification

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to evaluate alternative architectures (HOG+SVM, attention-based CNN, Vision Transformer) against the chosen lightweight CNN and produce a formal justification.

**Response synopsis:** HOG+SVM is interpretable but accuracy-capped by handcrafted features — retained as classical baseline only. Attention CNN (EMC2A-Net style) is technically stronger but adds multi-branch complexity disproportionate to a CPU-only, small-dataset setting. ViT requires large data and pretraining to converge — unsuitable for 232 samples/class on CPU. Custom lightweight CNN balances accuracy, CPU-feasibility, and dataset size.

**Design change:** Used as Choice Justification and Alternative Design Options sections in the proposal presentation.

---

## Midterm Development Session — June 21, 2026

### Entry 3: Implementation of Four Pipeline Scripts

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to implement load_data.py, preprocess.py, baseline_hog_svm.py, and train_cnn.py for the full MSTAR ATR pipeline with inline documentation and output figures.

**Response synopsis:** Claude implemented all four scripts. `load_data.py` downloads the MSTAR NPZ file and verifies 696 train / 822 test samples (232 train, 274 test per class). `preprocess.py` applies Lee adaptive filter (window=5, local variance weighting), percentile-based CFAR threshold (95th percentile per chip), binary morphological opening then closing (1 iteration each), and 64×64 centre crop; saves 4-panel pipeline figures per class. `baseline_hog_svm.py` extracts HOG features (9 orientations, 8×8 pixels/cell, 2×2 cells/block) and trains an RBF-SVM (C=10, gamma=scale, one-vs-one). `train_cnn.py` defines SARNet (3 Conv-BN-ReLU-MaxPool blocks → FC 512 → FC 128 → FC 3), trains with Adam (lr=0.001) and StepLR decay (γ=0.5 every 5 epochs), saves weights and training curves.

**Design change:** All four scripts added to repository via four separate incremental commits.

---

### Entry 4: Script Execution and Results

**Tool:** Claude (claude.ai)

**Prompt:** Executed all four scripts and reported real accuracy metrics for inclusion in the midterm presentation.

**Response synopsis:** HOG+SVM: 91.4% overall test accuracy (BMP2 F1=0.86, BTR70 F1=0.89, T72 F1=0.99); primary failure mode is BMP2↔BTR70 confusion (60 of 274 BMP2 chips misclassified as BTR70 due to similar radar cross-sections at overlapping aspect angles). SARNet CNN at 10 epochs: 90.1% overall (BMP2 F1=0.84, BTR70 F1=0.87, T72 F1=0.99); peak test accuracy 95.6% at epoch 9 before final-epoch drop indicating early stopping would improve generalisation. Outputs saved: confusion matrices, training curves, model weights (sarnet_weights.pt).

**Design change:** Midterm slide deck updated with real accuracy numbers. Confusion matrix values updated from estimated to actual.

---

### Entry 5: Midterm Presentation Generation

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to generate a 10-slide midterm progress report presentation covering all four required CS-898BA sections using real MSTAR chip images and matching the proposal visual theme.

**Response synopsis:** Claude built a 10-slide pptxgenjs deck: title slide with real MSTAR montage background and radar reticle, agenda, condensed lit review (Lee filter, AConvNet, CFAR retained; ViT dropped with justification), 4-step pipeline overview with flow bar, pipeline demo slide with real chip before/after despeckling and all 3 class samples, HOG+SVM results (91.4% overall, per-class F1), SARNet CNN results (90.1% overall, epoch training history), confusion matrix analysis with BMP2↔BTR70 error pattern highlighted, roadblocks and pivots (CPU training time, dataset label ambiguity, CFAR threshold sensitivity), and next steps summary.

**Design change:** Submitted as Midterm Progress Report presentation with all placeholder numbers replaced by real executed results.

---

### Entry 6: Documentation Generation — Midterm

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to write a comprehensive README.md and technical AI_Log.md following the CS-898BA documentation standard.

**Response synopsis:** README produced covering: problem statement, dataset specification table, pipeline architecture diagram, installation instructions, step-by-step execution guide, results table with per-class F1 scores, CNN epoch-by-epoch training history, full SARNet architecture spec (layer dimensions, optimizer, loss, scheduler), repository file structure, and references. AI_Log restructured in session/entry narrative format with technical prompt descriptions, response synopses, and specific design changes only.

**Design change:** README.md and AI_Log.md committed to repository.

---

## Final Development Session — July 28, 2026

### Entry 7: Hyperparameter Tuning Script Implementation

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to implement hyperparameter_tuning.py performing a full grid search across LR ∈ {0.01, 0.001, 0.0001}, batch ∈ {16, 32}, and dropout ∈ {0.3, 0.5} — 12 total configurations — with live accuracy reporting, a bar chart of all results, and learning curves for the top 3 configurations.

**Response synopsis:** Claude implemented hyperparameter_tuning.py running all 12 combinations for 10 epochs each. Script prints a live table during execution, saves best model weights to sarnet_best_weights.pt, outputs hyperparameter_comparison.png and top3_learning_curves.png, and writes hyperparameter_results.json. Executed on student laptop — best result: config 9 (LR=0.0001, Batch=16, Dropout=0.3) achieved 98.4% best accuracy. Lower learning rate with small batch produced the most stable convergence on the 232-sample-per-class dataset.

**Design change:** hyperparameter_tuning.py committed to src/. Best configuration identified as LR=0.0001, Batch=16, Dropout=0.3.

---

### Entry 8: Final Evaluation Script Implementation

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to implement final_evaluation.py comparing three pipeline variants: HOG+SVM on preprocessed chips, SARNet CNN on raw chips without preprocessing, and SARNet CNN with full pipeline and best hyperparameter config. Script to produce confusion matrices, learning curves, per-class F1 scores, and a full comparison table.

**Response synopsis:** Claude implemented final_evaluation.py training all three variants using best config (LR=0.0001, Batch=16, Dropout=0.3, 15 epochs). Results: HOG+SVM 91.4% (BMP2 F1=0.86, BTR70 F1=0.89, T72 F1=0.99); CNN no pipeline 94.9% (BMP2 F1=0.92, BTR70 F1=0.93, T72 F1=1.00); CNN pipeline+tuned 99.5% (BMP2 F1=0.99, BTR70 F1=1.00, T72 F1=1.00). BMP2↔BTR70 confusion dropped from 60 misclassifications to 2. Saved final_cm_hog_svm.png, final_cm_cnn_raw.png, final_cm_cnn_tuned.png, final_comparison_curves.png, and final_results.json to outputs/.

**Design change:** final_evaluation.py committed to src/. Final accuracy confirmed at 99.5% with full pipeline and tuned hyperparameters.

---

### Entry 9: Final Presentation Generation

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to generate a 9-slide final presentation covering all four required CS-898BA final sections — Final Architecture & Design, Hyperparameter Optimization, Image Analysis Evaluation, and Results & Metrics — plus a Virtual Demonstration slide, using the same navy/cyan radar visual theme as the proposal and midterm decks.

**Response synopsis:** Claude built a 9-slide pptxgenjs deck: title slide with real MSTAR chip montage background and radar reticle showing 99.5% final accuracy, agenda covering all 4 required sections, final architecture slide showing all 6 pipeline steps with best config summary bar, hyperparameter optimization slide with full 12-config results table highlighting config 9 in cyan, image analysis evaluation slide with real despeckle before/after demo and pipeline impact comparison, results and metrics slide with full comparison table and three key metric callout cards (+8.1%, +4.6%, F1=1.00), confusion matrix analysis slide showing all three models side by side, virtual demonstration slide showing real chips from all three classes processed end-to-end with predicted labels and confidence scores, and conclusion slide summarising all three key findings.

**Design change:** Submitted as Final Presentation slide deck.

---

### Entry 10: Documentation Generation — Final

**Tool:** Claude (claude.ai)

**Prompt:** Asked Claude to update README.md with final hyperparameter tuning results, final comparison table, new scripts, and updated output files list, keeping the existing structure and writing style unchanged.

**Response synopsis:** README updated with full hyperparameter grid search table showing all 12 configurations and best config highlighted, final three-model comparison table (HOG+SVM 91.4%, CNN raw 94.9%, CNN tuned 99.5%), updated model architecture spec reflecting best hyperparameters (LR=0.0001, Batch=16, Dropout=0.3, 15 epochs), two new script entries in the file descriptions table, updated output files section listing all new confusion matrices and result files, and updated evaluation and analysis section covering all three pipeline variants.