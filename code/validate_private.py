# -*- coding: utf-8 -*-
"""
OFC 2026 ML Challenge - Private Dataset Inference Engine (V17.0 - 0.08967)
Optimized for final validation on unseen/private data.
- Loads 15 Snapshot Models (5-Fold x 3-Snap).
- Uses Quadratic Physics Template for spectral alignment.
- Dense Grid-Search TTA.
"""

import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- V17 Model Architecture ---
class TransformerUNet(nn.Module):
    def __init__(self, nch, global_dim, n_dev):
        super().__init__()
        self.nch = nch; self.dev_emb = nn.Embedding(n_dev, 16)
        self.dev_proj = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Conv1d(3, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(True))
        self.bottleneck_conv = nn.Sequential(nn.Conv1d(256+128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(True))
        self.bottleneck_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.g_proj = nn.Sequential(nn.Linear(global_dim+64, 128), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv1d(256+128, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv1d(128+64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(True))
        self.head = nn.Conv1d(64, 1, 3, 1, 1)
        self.register_buffer("pos", torch.linspace(-1, 1, nch).view(1, 1, nch))

    def forward(self, load, pow, cfg, dev_id):
        # Precise Quadratic Physics Template (V17 Signature)
        ax = torch.linspace(0.5, -0.5, self.nch).to(load.device)
        base = cfg[:, 0:1] + (cfg[:, 1:2] * 0.72) * ax + (cfg[:, 1:2] * 0.05) * (ax**2)
        
        g = self.g_proj(torch.cat([cfg[:, 2:], self.dev_proj(self.dev_emb(dev_id))], 1))
        x = torch.stack([load, pow, self.pos.expand(load.size(0),-1,-1).squeeze(1)], 1)
        e1 = self.enc1(x); p1 = nn.functional.max_pool1d(e1, 2, ceil_mode=True)
        e2 = self.enc2(p1); p2 = nn.functional.max_pool1d(e2, 2, ceil_mode=True); e3 = self.enc3(p2)
        b = self.bottleneck_conv(torch.cat([e3, g.unsqueeze(-1).expand(-1,-1,e3.size(2))], 1))
        b_at, _ = self.bottleneck_attn(b.transpose(1, 2), b.transpose(1, 2), b.transpose(1, 2))
        b = b_at.transpose(1, 2)
        d1 = self.dec1(torch.cat([nn.functional.interpolate(b, e2.size(2), mode='linear'), e2], 1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d1, e1.size(2), mode='linear'), e1], 1))
        return base + self.head(d2).squeeze(1) * 0.92

def run_v17_inference(feature_path, out_path, model_dir):
    print(f"🚀 V17.0 Engine Loading Private Data: {feature_path}")
    df = pd.read_csv(feature_path)
    
    m_cols = [f"DUT_WSS_activated_channel_index_{i:02d}" for i in range(95)]
    p_cols = [f"EDFA_input_spectra_{i:02d}" for i in range(95)]
    
    # --- V17 Hardcoded Normalization Stats (Aligned with Training) ---
    m_trunk = np.array([18.0, -0.0017683338, -11.2927885, 7.317869], dtype=np.float32)
    s_trunk = np.array([2.445172, 0.042014383, 8.613661, 7.9722786], dtype=np.float32)
    m_stat  = np.array([24.44598, 0.257322, -21.032595, 0.11534409], dtype=np.float32)
    s_stat  = np.array([28.92054, 0.30437034, 3.3155823, 0.15772969], dtype=np.float32)
    
    all_dk_list = ['booster_0', 'booster_1', 'preamp_0', 'preamp_1', 'preamp_6', 'preamp_7', 'preamp_2', 'preamp_3', 'preamp_4', 'preamp_5', 'booster_6', 'booster_7', 'booster_2', 'booster_3', 'booster_4', 'booster_5']

    k2id = {k:i for i,k in enumerate(all_dk_list)}
    dk = (df["EDFA_type"].str.lower() + "_" + df["edfa_index"].astype(str)).map(k2id).fillna(0).values.astype(int)
    
    cat_ids = df["Category"].str.lower().map({"aging":0,"shb":1,"unseen":2}).fillna(2).astype(int).values if "Category" in df.columns else np.full(len(df),2)
    oh = np.eye(3)[cat_ids].astype(np.float32)
    
    xl = df[m_cols].values.astype(np.float32)
    xp = df[p_cols].values.astype(np.float32)
    
    n = np.sum(xl>0.5, 1, keepdims=True)+1e-8; mean = np.sum(xl*xp, 1, keepdims=True)/n
    st = np.concatenate([n, n/95.0, mean, np.sqrt(np.sum(xl*(xp-mean)**2,1,keepdims=True)/n+1e-8)], 1).astype(np.float32)
    trunk = df[["target_gain","target_gain_tilt","EDFA_input_power_total","EDFA_output_power_total"]].values.astype(np.float32)
    raw = df[["target_gain","target_gain_tilt"]].values.astype(np.float32)
    etype = (df["EDFA_type"].str.lower()=="booster").values.astype(np.float32)[:,None]
    
    xc = np.concatenate([raw, (trunk-m_trunk)/s_trunk, (st-m_stat)/s_stat, oh, etype], 1).astype(np.float32)
    
    models_found = [f for f in os.listdir(model_dir) if f.endswith(".pt") and "v17_fold" in f]
    print(f"Found {len(models_found)} V17 snapshots for ensemble.")
    
    pred_sum = np.zeros((len(df), 95))
    lp_t, pp_t, cp_t, dp_t = [torch.from_numpy(x).to(DEVICE) for x in (xl, xp, xc, dk)]
    
    for m_file in models_found:
        model = TransformerUNet(95, xc.shape[1]-2, len(all_dk_list)).to(DEVICE)
        model.load_state_dict(torch.load(f"{model_dir}/{m_file}", map_location=DEVICE), strict=False)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(lp_t), 512):
                p_base = model(lp_t[i:i+512], pp_t[i:i+512], cp_t[i:i+512], dp_t[i:i+512])
                # TTA 2D Grid
                tta_acc = p_base * 0.52
                for d_g, d_t in [(0.25, 0), (-0.25, 0), (0, 0.15), (0, -0.15)]:
                    off = torch.cat([torch.tensor([d_g, d_t], device=DEVICE), torch.zeros(12, device=DEVICE)])
                    tta_acc += model(lp_t[i:i+512], pp_t[i:i+512], cp_t[i:i+512]+off, dp_t[i:i+512]) * 0.12
                pred_sum[i:i+512] += tta_acc.cpu().numpy()
                
    final_pred = np.clip(pred_sum / len(models_found), -15, 25)
    out_df = pd.DataFrame(final_pred, columns=[f"calculated_gain_spectra_{i:02d}" for i in range(95)])
    if "ID" in df.columns: 
        out_df.insert(0, "ID", df["ID"].astype(int).values)
    
    out_df.to_csv(out_path, index=False)
    print(f"✅ V17 Validation Finished -> {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/test_features.csv")
    parser.add_argument("--output", default="v17_results.csv")
    parser.add_argument("--models", default=".")
    args = parser.parse_args()
    run_v17_inference(args.features, args.output, args.models)
