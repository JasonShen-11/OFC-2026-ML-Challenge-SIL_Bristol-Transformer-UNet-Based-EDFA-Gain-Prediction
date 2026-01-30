#!/bin/bash
# OFC 2026 ML Challenge - Private Dataset Validation Pipeline
# Implementation: Antigravity AI

set -e

echo "----------------------------------------------------"
echo "OFC 2026 ML Challenge - Private Validation Pipeline"
echo "----------------------------------------------------"

# 1. Environment Check
echo "[1/3] Checking environment..."
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install it."
    exit 1
fi

# 2. Dependency Check (Minimum required)
echo "[2/3] Verifying dependencies..."
python3 -c "import torch; import pandas; import numpy; import sklearn" || (echo "Installing missing packages..."; pip install torch pandas numpy scikit-learn)

# 3. Running Inference
echo "[3/3] Running V18.3 God Mode Inference..."

# Default paths, can be overridden by env variables
INPUT_FEATURES=${1:-"data/test_features.csv"}
OUTPUT_CSV=${2:-"validation_results.csv"}
MODEL_DIR="./"

python3 code/validate_private.py --features "$INPUT_FEATURES" --output "$OUTPUT_CSV" --models "$MODEL_DIR"

echo "----------------------------------------------------"
echo "✅ Validation Complete! Results saved to: $OUTPUT_CSV"
echo "----------------------------------------------------"
