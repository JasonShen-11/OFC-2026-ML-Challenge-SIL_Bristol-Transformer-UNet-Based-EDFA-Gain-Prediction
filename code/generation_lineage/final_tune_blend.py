import pandas as pd
import numpy as np
import os

def final_tune_blend():
    p_v15 = "../Features/Test/submission_v15_pseudo.csv"   # 0.09702
    p_mega = "../Features/Test/submission_mega_blend_v1.csv" # 0.09402
    
    if not os.path.exists(p_v15) or not os.path.exists(p_mega):
        print("Missing files")
        return

    df_v15 = pd.read_csv(p_v15)
    df_mega = pd.read_csv(p_mega)
    
    label_cols = [f"calculated_gain_spectra_{i:02d}" for i in range(95)]
    
    # Tiny weight for pseudo-labeling to see if it acts as a regularizer
    ens_data = 0.9 * df_mega[label_cols].values + 0.1 * df_v15[label_cols].values
    
    # Mask check
    mask_cols = [f"DUT_WSS_activated_channel_index_{i:02d}" for i in range(95)]
    test_f = pd.read_csv("../Features/Test/test_features.csv")
    mask = test_f[mask_cols].values
    ens_data[mask == 0] = -1.0
    
    ens_df = df_mega.copy()
    ens_df[label_cols] = ens_data
    out_path = "../Features/Test/submission_ultra_blend_v2.csv"
    ens_df.to_csv(out_path, index=False)
    print(f"Ultra Blend v2 saved to {out_path}")
    
    msg = "Ultra Blend v2: 0.9*Mega + 0.1*V15Pseudo"
    cmd = f"KAGGLE_CONFIG_DIR='/home/ubuntu/Desktop/ofc-competition' /home/ubuntu/Desktop/ofc-competition/.mmenv/bin/kaggle competitions submit -c ofc-2026-ml-challenge -f '{out_path}' -m '{msg}'"
    os.system(cmd)

if __name__ == "__main__":
    final_tune_blend()
