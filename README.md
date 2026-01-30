# OFC 2026 ML Challenge - Ultimate Version (V17.0 - 0.08967)

This package contains the complete code, models, and data for the **V17.0 Ultimate Finisher** version, which achieved a verified score of **0.08967** in the OFC 2026 ML Challenge. This version is designated as the "Paper-Ready" edition for final submission and validation.

## 📁 Folder Structure

- `code/`: Core logic and model execution.
  - `train_v17_paper_ready.py`: The main production script (5-Fold CV, Snapshot Ensembling, OOF tracking).
  - `validate_private.py`: Standalone inference engine for private/unseen datasets.
  - `v10_pretrained.pt`: Foundational pre-trained weights for model initialization.
  - `generation_lineage/`: The complete evolutionary hierarchy of our solution.
    - `train_v16_adversarial.py`: The first sub-0.09 attention model.
    - `final_mega_blend.py`: Multi-model ensemble logic (V13 + V14 + SuperBlend).
    - `create_distilled_pseudo.py`: Teacher-Student distillation logic (V16 + Blend -> V17).
    - `final_tune_blend.py`: Fine-tuning weights for the ultimate prediction.
- `data/`: Datasets and Reference Results.
  - `train_features.csv` / `train_labels.csv`: Original Stage B training data.
  - `test_features.csv`: Competition test data.
  - `pseudo_labels_v17.csv`: High-quality distilled targets used for training V17.
  - `COSMOS_features.csv` / `COSMOS_labels.csv`: Large-scale Stage A base data.
  - `reference_results/`: Historical raw submissions for lineage verification.

## 🚀 How to Run Validation (Private Dataset)

If you have a new private dataset (e.g., `private_features.csv`), use the optimized shell pipeline:

```bash
./run_validation.sh path/to/private_features.csv results.csv
```

This pipeline automatically ensembles all saved **v17_foldX_snapY.pt** checkpoints and applies **Dense TTA** for maximum stability.

## 🔬 V17 Core Strategy (The 0.08967 Winner)

1.  **Distilled Multi-Teacher Pseudo-Labeling**: Learned from a blend of the most "physically accurate" (V16) and "statistically stable" (Mega-Blend) models.
2.  **Snapshot Ensembling**: Instead of a single model, we capture **3 micro-states** per fold (at Epoch 100, 110, 120), totaling **15 models** across the 5-fold CV.
3.  **Quadratic Physics Template**: Hard-coded the base gain/tilt + a second-order term to guide the Transformer-UNet towards physical realism.
4.  **Multi-dimensional TTA**: Inference uses a grid-search perturbation of gain and tilt to smooth out spectral edges and outliers.

## 📜 Generation Lineage (Details)

The `pseudo_labels_v17.csv` is the result of a multi-stage distillation:
- **final_mega_blend.py**: Provided the stable "baseline teacher" (0.094).
- **train_v16_adversarial.py**: Provided the "physically fine-grained" single-model teacher (0.0898).
- **create_distilled_pseudo.py**: Blended them (0.7:0.3) to filter out individual model noise.

## 🏗️ Technical Pillars of the V16 Base (The Breakthrough)

V16 (Adversarial Attention UNet) was the turning point (Single model 0.0898). It provided the "High-Resolution Physicality" for V17:
- **Global Receptor Field (Transformer-UNet)**: Captures spectrum-wide gain competition via multi-head self-attention, modeling Stimulated Raman Scattering (SRS) across all 95 channels.
- **FGM Adversarial Training**: Strengthens the model by injecting gradient-based "malicious" noise during training, ensuring smoothness in the low-sample Stage B domain.
- **Spectrum-Stats Bottleneck**: Injected 5-dim stats (Load, Power Std, Slope) into the latent space to guide the model with physical constraints.

## 🌪️ The Mega-Blend Evolution (Stability at Scale)

Before V17 achieved peak precision, the **Mega-Blend** (0.0940) served as our anchor of stability. It combined diverse architectures to cancel out structural noise:

### 🛠️ V13 and V14 Technical Breakdown

#### **V13: The Physics-Heavy Foundation (Score: ~0.103)**
*   **Innovation: Quadratic Spectral Template.**
    *   V13 moved away from simple linear gain/tilt models. It introduced a second-order polynomial base ($ax^2 + bx$) calculated directly from the WSS configuration.
    *   **Performance**: While the raw MAE was higher than later versions, it provided the correct macroscopic "convex/concave" shape of the gain spectrum, which was crucial for filtering out unrealistic CNN predictions.
*   **Architecture**: Deep Residual CNN with narrow kernels to capture local ripple effects.

#### **V14: Category-Aware Specialization**
*   **Innovation 1: Category-Conditional Inputs.**
    *   V14 was the first to use **One-Hot Encoding** for EDFA categories (`Aging`, `SHB`, `Unmeasured`) directly injected into the model's global projection. This allowed the UNet to switch weights internally based on the specific physical degradation mode.
*   **Innovation 2: 8-Fold CV + SWA.**
    *   Increased the ensemble count and introduced **Stochastic Weight Averaging (SWA)** to find a wider, more robust local minimum.
    *   **Performance**: This version achieved a major leap in individual model stability and served as the primary driver (50% weight) of the Mega-Blend.

#### **Mega-Blend Logic (`final_mega_blend.py`)**: 
    - Used a weighted voting scheme (0.50 V14 + 0.30 Super-Blend + 0.15 Old-Best + 0.05 V13).
    - **Why it matters**: It provided the constant "dc-bias" and statistical stability that V16 (being sharper and more aggressive) occasionally lacked. V17 effectively distilled the "Stability" of this blend and the "Precision" of V16.
