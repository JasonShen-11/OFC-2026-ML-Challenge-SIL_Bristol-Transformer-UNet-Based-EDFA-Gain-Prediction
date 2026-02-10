#!/bin/bash
# OFC 2026 ML Challenge - V17 Training Pipeline
# Implementation: Antigravity AI

set -e

echo "----------------------------------------------------"
echo "OFC 2026 ML Challenge - V17 Training Pipeline"
echo "----------------------------------------------------"

# 1. Environment Check
echo "[1/3] Checking environment..."
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install it."
    exit 1
fi

# 2. Dependency Check
echo "[2/3] Verifying dependencies..."
python3 -c "import torch; import pandas; import numpy; import sklearn" || (echo "Installing missing packages..."; pip install torch pandas numpy scikit-learn)

# 3. Running Training
echo "[3/3] Starting V17 Production Training..."
echo "Log: train_output.log"

nohup python3 -u code/train_v17_paper_ready.py > train_output.log 2>&1 &

echo "----------------------------------------------------"
echo "✅ Training Started in Background!"
echo "Use 'tail -f train_output.log' to monitor progress."
echo "----------------------------------------------------"
