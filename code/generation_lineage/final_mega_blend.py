import pandas as pd
import numpy as np
import os

def mega_blend():
    p_v14 = "../Features/Test/submission_v14_catwise.csv" # 0.09954
    p_sb  = "../Features/Test/submission_super_blend.csv" # 0.09982
    p_v13 = "../Features/Test/submission_v13_kfold.csv"   # 0.10469
    p_old_best = "../Features/Test/submission_ensemble_best.csv" # 0.10299
    
    files = [p_v14, p_sb, p_v13, p_old_best]
    for f in files:
        if not os.path.exists(f):
            print(f"Missing: {f}")
            return

    df_v14 = pd.read_csv(p_v14)
    df_sb  = pd.read_csv(p_sb)
    df_v13 = pd.read_csv(p_v13)
    df_old = pd.read_csv(p_old_best)
    
    label_cols = [f"calculated_gain_spectra_{i:02d}" for i in range(95)]
    
    # Weights based on performance and diversity
    # V14 is the best (0.0995), SuperBlend is second (0.0998), OldBest is consistent (0.102), V13 is fold-stable (0.104)
    w_v14 = 0.50
    w_sb  = 0.30
    w_old = 0.15
    w_v13 = 0.05
    
    ens_data = (w_v14 * df_v14[label_cols].values + 
                w_sb  * df_sb[label_cols].values + 
                w_old * df_old[label_cols].values +
                w_v13 * df_v13[label_cols].values)
    
    # Mask check
    mask_cols = [f"DUT_WSS_activated_channel_index_{i:02d}" for i in range(95)]
    test_f = pd.read_csv("../Features/Test/test_features.csv")
    mask = test_f[mask_cols].values
    ens_data[mask == 0] = -1.0
    
    ens_df = df_v14.copy()
    ens_df[label_cols] = ens_data
    out_path = "../Features/Test/submission_mega_blend_v1.csv"
    ens_df.to_csv(out_path, index=False)
    print(f"Mega Blend saved to {out_path}")
    
    msg = f"Mega Blend v1: {w_v14}*V14 + {w_sb}*SB + {w_old}*Old + {w_v13}*V13"
    cmd = f"KAGGLE_CONFIG_DIR='/home/ubuntu/Desktop/ofc-competition' /home/ubuntu/Desktop/ofc-competition/.mmenv/bin/kaggle competitions submit -c ofc-2026-ml-challenge -f '{out_path}' -m '{msg}'"
    os.system(cmd)

if __name__ == "__main__":
    mega_blend()
