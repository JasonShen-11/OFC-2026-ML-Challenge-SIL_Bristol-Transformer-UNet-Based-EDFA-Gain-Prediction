# OFC 2026 ML Challenge - Optimal Version (V18.0 God Mode)

This folder contains the complete code and data for the most advanced submission in the OFC 2026 ML Challenge, aimed at achieving a score of 0.082.

## Folder Structure

- `code/`: Contains the training script and pre-trained weights.
  - `train_v18_godmode.py`: The main script (10-Fold CV, SWA, Wide Transformer-UNet).
  - `v10_pretrained.pt`: Pre-trained weights for model initialization.
  - `generation_lineage/`: Scripts used to generate `pseudo_labels_v17.csv`.
    - `create_distilled_pseudo.py`: Blends V16 and Mega Blend.
    - `train_v17_ultimate.py`: Trains the V17 model used for labeling.
    - `train_v16_adversarial.py`: The high-performance adversarial single model.
    - `final_mega_blend.py`: The comprehensive multi-model ensemble.
- `data/`: Contains all necessary datasets.
  - `train_features.csv` / `train_labels.csv`: Original training data.
  - `test_features.csv`: Original test data.
  - `pseudo_labels_v17.csv`: High-quality pseudo labels from our best previous iteration (0.089), used as hard targets for V18.
  - `COSMOS_features.csv` / `COSMOS_labels.csv`: Large-scale Stage A data.
  - `reference_results/`: Historical submission files for lineage verification.
    - `submission_v16.csv`: V16 single model result (0.0898).
    - `submission_mega_blend_v1.csv`: Mega Blend result (0.0940).
    - `submission_v17_ultimate.csv`: V17 result (0.0896).

## How to Run

1. **Environment Setup**:
   Ensure you have a Python 3.10+ environment with PyTorch and the following dependencies installed:
   ```bash
   pip install torch pandas numpy scikit-learn
   ```

2. **Run Training**:
   Navigate to the `code` directory and execute the script:
   ```bash
   cd code
   python train_v18_godmode.py
   ```

3. **Output**:
   The script will generate `submission_v18_godmode.csv` in the `code/` directory after 10-fold training and TTA inference.

## Key Strategies Implemented

- **Wide Transformer-UNet**: A 512-dimension bottleneck with 8-head self-attention for global spectrum context.
- **Learned Spline Template**: An adaptive physics-based template that learns the unique gain ripple patterns.
- **10-Fold CV + SWA**: Maximum data utilization and Stochastic Weight Averaging for extreme generalization.
- **Hard Pseudo-Labeling**: Leveraging distilled knowledge from our best 0.089 model with noise injection.
- **Dense TTA**: 8-point grid search perturbation for gain and tilt during inference.

## Pseudo Label Generation Lineage (Details)

The `pseudo_labels_v17.csv` file is the result of a multi-stage distillation process. To understand its origin or reproduce it, refer to the following scripts in `code/generation_lineage/`:

1.  **`final_mega_blend.py`**:
    *   **Function**: Combines early successful versions (V13, V14, V15).
    *   **Outcome**: Produced the initial 0.094 submission. It provides a stable "baseline teacher" that captures the broader dataset trends.

2.  **`train_v16_adversarial.py`**:
    *   **Function**: Implements the first iteration of the Adversarial Attention UNet.
    *   **Outcome**: Produced the 0.0898 single-model submission. It is the most "physically accurate" individual model, focusing on fine-grained spectrum details.

3.  **`create_distilled_pseudo.py`**:
    *   **Function**: Blends the results of `final_mega_blend.py` (weight 0.3) and `train_v16_adversarial.py` (weight 0.7).
    *   **Outcome**: Created a "Super-Teacher" label set. This blending filters out individual model errors and emphasizes points where both high-stability and high-accuracy models agree.

4.  **`train_v17_ultimate.py`**:
    *   **Function**: Trains a 5-Fold model using the labels from `create_distilled_pseudo.py`.
    *   **Outcome**: This script uses Snapshot Ensembling (collecting multiple checkpoints during the final stages of training) and TTA to produce the `pseudo_labels_v17.csv` used by V18. It is the most robust labeling tool in the lineage.
