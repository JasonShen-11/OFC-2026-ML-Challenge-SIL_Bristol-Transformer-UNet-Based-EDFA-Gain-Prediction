# OFC 2026 ML Challenge – SIL_Bristol Team

## 1. Inference Script

The latest inference code used to run our submitted model is available at:

https://github.com/JasonShen-11/OFC-2026-ML-Challenge-SIL_Bristol-Transformer-UNet-Based-EDFA-Gain-Prediction/blob/15739e6845299bc0bfef3c1dea1bd0948f2e4985/code/validate_private.py

This script loads the pretrained model and performs **inference only** (no training).

---

## 2. How to Run Inference

The inference script can be executed using the following command:

python test_inference.py --input_csv <PATH_TO_INPUT_CSV>

Example:
python test_inference.py --input_csv D:\PythonProject\OFC2026ML\OFC-2026-ML-Challenge-SIL_Bristol-Transformer-UNet-Based-EDFA-Gain-Prediction\data\test_features.csv


To run the script with your own input file, simply replace the path after `--input_csv` with the **absolute path** to the target Kaggle-format input CSV.

The script will read the input CSV and generate the corresponding **submission CSV**.

---

## 3. Environment Setup

Before running the inference script, please install the required Python dependencies listed in `requirements.txt`.

Example:
pip install -r requirements.txt

---

## 4. Model Artifacts

All required model artifacts are included in the repository, including:

- trained model weights  
- preprocessing components (e.g., standardizers)  
- reference tables used during inference  

The inference script automatically loads these artifacts when executed.

---

## 5. Notes

- The inference code uses the **exact model and weights submitted during the competition**.
- No training procedure is included in the inference script.


