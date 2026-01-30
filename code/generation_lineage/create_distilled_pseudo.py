import pandas as pd
import numpy as np

def create_ultimate_pseudo():
    p_v16 = "../Features/Test/submission_v16.csv"           # 0.08981
    p_mega = "../Features/Test/submission_mega_blend_v1.csv" # 0.09402
    
    df_v16 = pd.read_csv(p_v16)
    df_mega = pd.read_csv(p_mega)
    
    label_cols = [f"calculated_gain_spectra_{i:02d}" for i in range(95)]
    
    # Simple weighted blend for the new teacher
    # 0.7 from the better model, 0.3 from the ensemble
    distilled_data = 0.7 * df_v16[label_cols].values + 0.3 * df_mega[label_cols].values
    
    # Masking handled by the training script usually, but safe to keep here
    df_out = df_v16.copy()
    df_out[label_cols] = distilled_data
    out_path = "../Features/Test/pseudo_v17_distilled.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Ultimate Teacher Labels created at: {out_path}")

if __name__ == "__main__":
    create_ultimate_pseudo()
