# -*- coding: utf-8 -*-
"""
OFC 2026 ML Challenge - VERSION 16.2 (Adversarial Attention UNet)
Strategy:
1. Transformer Bottleneck + FGM Adv Training.
2. 5-Fold Semi-Supervised with 0.8 Pseudo weight.
3. Robust TTA Inference.
4. Fixed 95-channel alignment.
"""

import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

BASE_SEED = 2026; seed_all(BASE_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configs
TRAIN_FEATURES_CSV = "../Features/Train/train_features.csv"
TRAIN_LABELS_CSV   = "../Features/Train/train_labels.csv"
TEST_FEATURES_CSV  = "../Features/Test/test_features.csv"
PSEUDO_CSV         = "../Features/Test/submission_mega_blend_v1.csv"
COSMOS_FEATURES_CSV = "../Features/Train/COSMOS_features.csv"
COSMOS_LABELS_CSV   = "../Features/Train/COSMOS_labels.csv"
OUT_SUB_CSV        = "../Features/Test/submission_v16.csv"
PRETRAINED_PATH    = "v10_pretrained.pt"

BATCH_SIZE = 128; N_FOLDS = 5 # Back to 5 for speed + stability in pool training
EPOCHS_B1 = 50; EPOCHS_B2 = 120
LR_B1 = 1e-4; LR_B2 = 5e-5

FGM_EPS = 0.5; W_PSEUDO = 0.75; W_COSMOS = 0.15

# =========================
# 1) Network
# =========================
class FGM():
    def __init__(self, model):
        self.model = model; self.backup = {}
    def attack(self, epsilon=1.0, emb_name='dev_emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)
    def restore(self, emb_name='dev_emb.'):
        for name, param in self.model.named_parameters():
            if name in self.backup: param.data = self.backup[name]
        self.backup = {}

class TransformerUNet(nn.Module):
    def __init__(self, nch, global_dim, n_dev):
        super().__init__()
        self.nch = nch; self.dev_emb = nn.Embedding(n_dev, 16)
        self.dev_proj = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Conv1d(3, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(True))
        self.bottleneck_conv = nn.Sequential(nn.Conv1d(256+128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.ReLU(True))
        self.bottleneck_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.g_proj = nn.Sequential(nn.Linear(global_dim+64, 128), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv1d(256+128, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv1d(128+64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(True))
        self.head = nn.Conv1d(64, 1, 3, 1, 1)
        self.register_buffer("pos", torch.linspace(-1, 1, nch).view(1, 1, nch))

    def forward(self, load, pow, cfg, dev_id):
        base = cfg[:, 0:1] + (cfg[:, 1:2] * 0.72) * torch.linspace(0.5, -0.5, self.nch).to(DEVICE)
        g = self.g_proj(torch.cat([cfg[:, 2:], self.dev_proj(self.dev_emb(dev_id))], 1))
        x = torch.stack([load, pow, self.pos.expand(load.size(0),-1,-1).squeeze(1)], 1)
        e1 = self.enc1(x); p1 = nn.functional.max_pool1d(e1, 2, ceil_mode=True)
        e2 = self.enc2(p1); p2 = nn.functional.max_pool1d(e2, 2, ceil_mode=True)
        e3 = self.enc3(p2)
        b = self.bottleneck_conv(torch.cat([e3, g.unsqueeze(-1).expand(-1,-1,e3.size(2))], 1))
        b_at, _ = self.bottleneck_attn(b.transpose(1, 2), b.transpose(1, 2), b.transpose(1, 2))
        b = b_at.transpose(1, 2)
        d1 = self.dec1(torch.cat([nn.functional.interpolate(b, e2.size(2), mode='linear'), e2], 1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d1, e1.size(2), mode='linear'), e1], 1))
        return base + self.head(d2).squeeze(1) * 0.95

# =========================
# 2) Data
# =========================
def prep():
    print("V16.2 Precise Preprocessing...")
    tr_f, tr_l = pd.read_csv(TRAIN_FEATURES_CSV), pd.read_csv(TRAIN_LABELS_CSV)
    te_f, ps_l = pd.read_csv(TEST_FEATURES_CSV), pd.read_csv(PSEUDO_CSV)
    cs_f, cs_l = pd.read_csv(COSMOS_FEATURES_CSV), pd.read_csv(COSMOS_LABELS_CSV).fillna(0)
    
    label_cols = [f"calculated_gain_spectra_{i:02d}" for i in range(95)]
    m_cols = [f"DUT_WSS_activated_channel_index_{i:02d}" for i in range(95)]
    p_cols = [f"EDFA_input_spectra_{i:02d}" for i in range(95)]
    
    def get_dk(df): return df["EDFA_type"].str.lower() + "_" + df["edfa_index"].astype(str)
    all_dk = pd.Index(pd.concat([get_dk(tr_f), get_dk(te_f), get_dk(cs_f)]).unique()); k2id = {k:i for i,k in enumerate(all_dk)}
    
    def _prep(df, labels_df=None):
        l, p = df[m_cols].values.astype(np.float32), df[p_cols].values.astype(np.float32)
        dk = get_dk(df).map(k2id).values
        cat = df["Category"].str.lower().map({"aging":0,"shb":1,"unseen":2}).fillna(2).astype(int).values if "Category" in df.columns else np.full(len(df),2,dtype=int)
        oh = np.eye(3)[cat].astype(np.float32)
        n = np.sum(l>0.5, 1, keepdims=True)+1e-8; mean = np.sum(l*p, 1, keepdims=True)/n
        st = np.concatenate([n, n/95.0, mean, np.sqrt(np.sum(l*(p-mean)**2,1,keepdims=True)/n+1e-8)], 1).astype(np.float32)
        r = df[["target_gain","target_gain_tilt"]].values.astype(np.float32)
        t = df[["target_gain","target_gain_tilt","EDFA_input_power_total","EDFA_output_power_total"]].values.astype(np.float32)
        if labels_df is not None:
            y = labels_df[label_cols].values.astype(np.float32)
            vm = l * (y>0); left, right = np.zeros_like(l), np.zeros_like(l)
            left[:,1:] = np.abs(l[:,1:]-l[:,:-1]); right[:,:-1] = np.abs(l[:,:-1]-l[:,1:])
            chw = np.clip(1.0 + 1.2*np.clip(left+right,0,1), 1, 3)
            y_pk = np.concatenate([y, vm, l, chw], 1).astype(np.float32)
            return l, p, dk, r, t, st, oh, y_pk
        return l, p, dk, r, t, st, oh
    
    xl_tr, xp_tr, xd_tr, cr_tr, ct_tr, st_tr, co_tr, y_tr_p = _prep(tr_f, tr_l)
    xl_te, xp_te, xd_te, cr_te, ct_te, st_te, co_te, y_te_p = _prep(te_f, ps_l)
    xl_cs, xp_cs, xd_cs, cr_cs, ct_cs, st_cs, co_cs, y_cs_p = _prep(cs_f, cs_l)
    
    m_t, s_t = np.mean(np.vstack([ct_tr, ct_cs]),0), np.std(np.vstack([ct_tr, ct_cs]),0)+1e-8
    m_s, s_s = np.mean(np.vstack([st_tr, st_cs]),0), np.std(np.vstack([st_tr, st_cs]),0)+1e-8
    def _fin(r, t, s, o, et):
        etype = (et.str.lower()=="booster").values.astype(np.float32)[:,None]
        return np.concatenate([r, (t-m_t)/s_t, (s-m_s)/s_s, o, etype], 1).astype(np.float32)
    
    xc_tr, xc_te, xc_cs = _fin(cr_tr, ct_tr, st_tr, co_tr, tr_f["EDFA_type"]), _fin(cr_te, ct_te, st_te, co_te, te_f["EDFA_type"]), _fin(cr_cs, ct_cs, st_cs, co_cs, cs_f["EDFA_type"])
    w_tr = np.ones(len(tr_f), dtype=np.float32); cat = tr_f["Category"].str.lower().values
    w_tr[cat=="shb"] = 2.8; w_tr[cat=="unseen"] = 1.8; w_tr *= np.clip(1.0 + 0.5*np.sqrt(np.sum(xl_tr>0.5,1)/47.5), 1, 2)
    return (xl_tr, xp_tr, xc_tr, xd_tr, y_tr_p, w_tr), (xl_te, xp_te, xc_te, xd_te, y_te_p), (xl_cs, xp_cs, xc_cs, xd_cs, y_cs_p), te_f, all_dk

class EdfaDS(Dataset):
    def __init__(self, l, p, c, d, y, w):
        self.l,self.p,self.c,self.d,self.y,self.w = map(torch.from_numpy, (l,p,c,d,y,w))
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.l[i],self.p[i],self.c[i],self.d[i],self.y[i],self.w[i]

def masked_huber_loss(y_p, y_pk, w, delta=0.75):
    yt, vm, chw = y_pk[:,:95], y_pk[:,95:190], y_pk[:,285:]
    diff = (yt - y_p).abs(); huber = 0.5*torch.clamp(diff, max=delta)**2 + delta*(diff-torch.clamp(diff,max=delta))
    loss = (huber * vm * chw).sum(1) / (vm * chw).sum(1).clamp(1e-8)
    return (loss * w).mean()

# =========================
# 3) Run
# =========================
def main():
    (xl_tr, xp_tr, xc_tr, xd_tr, y_tr_p, w_tr), (xl_te, xp_te, xc_te, xd_te, y_te_p), (xl_cs, xp_cs, xc_cs, xd_cs, y_cs_p), te_f, all_dk = prep()
    l_pool, p_pool, c_pool, d_pool, y_pool, w_pool = np.vstack([xl_tr, xl_te]), np.vstack([xp_tr, xp_te]), np.vstack([xc_tr, xc_te]), np.concatenate([xd_tr, xd_te]), np.vstack([y_tr_p, y_te_p]), np.concatenate([w_tr, np.full(len(xl_te), W_PSEUDO)])
    
    pred_sum = np.zeros((len(xl_te), 95), dtype=np.float32)
    model = TransformerUNet(95, xc_tr.shape[1]-2, len(all_dk)).to(DEVICE)
    fgm = FGM(model); kf = KFold(5, shuffle=True, random_state=BASE_SEED)
    
    for f, (ti, vi) in enumerate(kf.split(l_pool), 1):
        print(f"FOLD {f}/5"); seed_all(BASE_SEED + f)
        if os.path.exists(PRETRAINED_PATH):
            model.load_state_dict({k:v for k,v in torch.load(PRETRAINED_PATH, map_location=DEVICE).items() if "template" not in k and "g_proj" not in k}, strict=False)
        
        idx = np.random.choice(len(xl_cs), 12000, False)
        xl_m, xp_m, xc_m, xd_m, y_m, w_m = np.vstack([l_pool[ti], xl_cs[idx]]), np.vstack([p_pool[ti], xp_cs[idx]]), np.vstack([c_pool[ti], xc_cs[idx]]), np.concatenate([d_pool[ti], xd_cs[idx]]), np.vstack([y_pool[ti], y_cs_p[idx]]), np.concatenate([w_pool[ti], np.full(12000, W_COSMOS)])
        
        opt = optim.AdamW(model.parameters(), lr=LR_B1, weight_decay=1e-5)
        for e in range(1, EPOCHS_B1+1):
            model.train(); t = 0
            for l, p, c, d, y, w in DataLoader(EdfaDS(xl_m, xp_m, xc_m, xd_m, y_m, w_m), BATCH_SIZE, True):
                l,p,c,d,y,w = [x.to(DEVICE) for x in (l,p,c,d,y,w)]
                opt.zero_grad(); loss = masked_huber_loss(model(l,p,c,d), y, w); loss.backward()
                fgm.attack(FGM_EPS); masked_huber_loss(model(l,p,c,d), y, w).backward()
                fgm.restore(); opt.step(); t += loss.item()
            if e % 25 == 0: print(f"  B1 E{e} L: {t/len(ti):.6f}")

        opt = optim.AdamW(model.parameters(), lr=LR_B2, weight_decay=2e-5)
        for e in range(1, EPOCHS_B2+1):
            model.train()
            for l,p,c,d,y,w in DataLoader(EdfaDS(l_pool[ti], p_pool[ti], c_pool[ti], d_pool[ti], y_pool[ti], w_pool[ti]), BATCH_SIZE, True):
                l,p,c,d,y,w = [x.to(DEVICE) for x in (l,p,c,d,y,w)]
                opt.zero_grad(); masked_huber_loss(model(l,p,c,d), y, w).backward(); opt.step()
        
        model.eval()
        with torch.no_grad():
            lp, pp, cp, dp = [torch.from_numpy(x).to(DEVICE) for x in (xl_te, xp_te, xc_te, xd_te)]
            for i in range(0, len(lp), 512):
                p_orig = model(lp[i:i+512], pp[i:i+512], cp[i:i+512], dp[i:i+512])
                # TTA: perturb gain and tilt
                p1 = model(lp[i:i+512], pp[i:i+512], cp[i:i+512] + torch.cat([torch.tensor([0.2, 0], device=DEVICE), torch.zeros(12, device=DEVICE)]), dp[i:i+512])
                p2 = model(lp[i:i+512], pp[i:i+512], cp[i:i+512] + torch.cat([torch.tensor([0, 0.1], device=DEVICE), torch.zeros(12, device=DEVICE)]), dp[i:i+512])
                pred_sum[i:i+512] += (p_orig * 0.7 + p1 * 0.15 + p2 * 0.15).cpu().numpy()

    res = np.clip(pred_sum / 5.0, -15, 25); res[xl_te == 0] = -1
    pd.DataFrame(res, columns=[f"calculated_gain_spectra_{i:02d}" for i in range(95)]).insert(0, "ID", te_f["ID"].values)
    out_df = pd.DataFrame(res, columns=[f"calculated_gain_spectra_{i:02d}" for i in range(95)])
    out_df.insert(0, "ID", te_f["ID"].values); out_df.to_csv(OUT_SUB_CSV, index=False)
    os.system(f"KAGGLE_CONFIG_DIR='/home/ubuntu/Desktop/ofc-competition' /home/ubuntu/Desktop/ofc-competition/.mmenv/bin/kaggle competitions submit -c ofc-2026-ml-challenge -f {OUT_SUB_CSV} -m 'V16.2: Precise 95-ch Adversarial'")

if __name__ == "__main__": main()
