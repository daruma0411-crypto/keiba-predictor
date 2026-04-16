"""
v2 予測エンジン — 強化版NN
- 4層構造 (128→128→64→32)
- ListNet風ランキング損失
- 学習データ量拡大対応 (30,000-50,000件)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.features_v2 import FEATURES_V2, CAT_FEATURES_V2


# ============================================================
# NNモデル v2 — 4層構造
# ============================================================
class HorseRaceModelV2(nn.Module):
    def __init__(self, nf, ed=None):
        super().__init__()
        self.embs = nn.ModuleDict()
        te = 0
        if ed:
            for n, (v, d) in ed.items():
                self.embs[n] = nn.Embedding(v, d)
                te += d
        self.net = nn.Sequential(
            nn.Linear(nf + te, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.25),  # 追加層
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.mu = nn.Linear(32, 1)
        self.sig = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x, c=None):
        p = [x]
        if c:
            for n, i in c.items():
                if n in self.embs:
                    p.append(self.embs[n](i))
        h = self.net(torch.cat(p, 1))
        return self.mu(h), self.sig(h) + 0.1


# ============================================================
# ランキング損失 (ListNet風)
# ============================================================
def listnet_loss(mu, sigma, y_true, race_ids, alpha_rank=0.3):
    """
    Gaussian NLL + ListNet風ランキング損失のハイブリッド

    Parameters
    ----------
    mu, sigma : Tensor (n, 1)
    y_true : Tensor (n, 1) - 正規化着順 (0-1)
    race_ids : Tensor or None - レースID（同一レース内で順位比較）
    alpha_rank : float - ランキング損失の重み
    """
    # Gaussian NLL (従来)
    nll = torch.mean(0.5 * torch.log(sigma ** 2) + 0.5 * ((y_true - mu) / sigma) ** 2)

    # Race-level ListNet
    if race_ids is None:
        # fallback to batch-global
        p_true = F.softmax(-y_true.squeeze(), dim=0)
        p_pred = F.softmax(-mu.squeeze(), dim=0)
        rank_loss = -torch.sum(p_true * torch.log(p_pred + 1e-8))
    else:
        rank_loss = torch.tensor(0.0, device=mu.device)
        unique_ids = torch.unique(race_ids)
        count = 0
        for rid in unique_ids:
            mask = (race_ids == rid)
            if mask.sum() < 2:
                continue
            p_true_r = F.softmax(-y_true[mask].squeeze(), dim=0)
            p_pred_r = F.softmax(-mu[mask].squeeze(), dim=0)
            rank_loss = rank_loss + (-torch.sum(p_true_r * torch.log(p_pred_r + 1e-8)))
            count += 1
        if count > 0:
            rank_loss = rank_loss / count

    return nll + alpha_rank * rank_loss


# ============================================================
# Predictor v2
# ============================================================
class PredictorV2:
    """
    v2予測エンジン
    - 4層NN
    - Gaussian NLL + ListNetハイブリッド損失
    - 学習データ30,000-50,000件対応
    """

    def __init__(self, numeric_features=None, cat_features=None):
        self.nf = numeric_features or FEATURES_V2
        self.cf = cat_features or ['kisyu_code', 'chokyosi_code', 'sire', 'banusi_name']
        self.sc = StandardScaler()
        self.les = {}
        self.m = None
        self.d = torch.device('cpu')
        self._med = None

    def _prepare(self, df, fit=False):
        """DataFrameからTensor群を準備"""
        X = df[self.nf].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')

        if fit:
            self._med = X.median()
            X = X.fillna(self._med)
            Xs = self.sc.fit_transform(X)
        else:
            X = X.fillna(self._med)
            Xs = self.sc.transform(X)

        Xt = torch.FloatTensor(Xs).to(self.d)

        ct = {}
        for c in self.cf:
            if c in df.columns:
                v = df[c].fillna('unknown').astype(str)
                if fit:
                    le = LabelEncoder()
                    le.fit(v)
                    self.les[c] = le
                else:
                    le = self.les[c]
                    v = v.map(lambda x, _le=le: x if x in _le.classes_ else _le.classes_[0])
                ct[c] = torch.LongTensor(le.transform(v)).to(self.d)

        return Xt, ct

    def train(self, df, ep=80, lr=0.003, bs=256, seed=42, alpha_rank=0.3, patience=5):
        """
        学習（Gaussian NLL + ListNet ハイブリッド損失）

        Parameters
        ----------
        ep : int - エポック数 (デフォルト80に増加)
        alpha_rank : float - ランキング損失の重み
        patience : int - early stopping patience (epochs)
        """
        import copy

        torch.manual_seed(seed)
        np.random.seed(seed)

        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float)
        h = df['heads'].values.astype(float)
        h[h == 0] = 16
        yt = torch.FloatTensor((y - 1) / (h - 1)).unsqueeze(1).to(self.d)

        # race_ids for ranking loss — encode as integer tensor
        if 'race_id' in df.columns:
            race_id_le = LabelEncoder()
            race_ids_t = torch.LongTensor(race_id_le.fit_transform(df['race_id'].values)).to(self.d)
        else:
            race_ids_t = None

        Xt, ct = self._prepare(df, True)

        ed = {}
        for c in self.cf:
            if c in self.les:
                vs = len(self.les[c].classes_)
                ed[c] = (vs, min(50, max(4, (vs + 1) // 2)))

        self.m = HorseRaceModelV2(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)

        # Early stopping: last 10% by index (data assumed sorted by date) as validation
        n = len(Xt)
        val_size = max(1, int(n * 0.1))
        train_size = n - val_size
        train_idx = torch.arange(train_size)
        val_idx = torch.arange(train_size, n)

        best_val_loss = float('inf')
        best_weights = None
        no_improve = 0

        self.m.train()

        for e in range(ep):
            # --- Training ---
            perm = train_idx[torch.randperm(train_size)]
            epoch_loss = 0
            batches = 0
            for i in range(0, train_size, bs):
                bi = perm[i:i + bs]
                rid_batch = race_ids_t[bi] if race_ids_t is not None else None
                mu, s = self.m(Xt[bi], {c: t[bi] for c, t in ct.items()})

                # ハイブリッド損失
                loss = listnet_loss(mu, s, yt[bi], rid_batch, alpha_rank)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()
                batches += 1

            scheduler.step()

            # --- Validation ---
            self.m.eval()
            with torch.no_grad():
                rid_val = race_ids_t[val_idx] if race_ids_t is not None else None
                mu_v, s_v = self.m(Xt[val_idx], {c: t[val_idx] for c, t in ct.items()})
                val_loss = listnet_loss(mu_v, s_v, yt[val_idx], rid_val, alpha_rank).item()
            self.m.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.m.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Restore best model
        if best_weights is not None:
            self.m.load_state_dict(best_weights)

    def predict(self, df):
        """各馬の能力値(μ, σ)を推論"""
        self.m.eval()
        Xt, ct = self._prepare(df, False)
        with torch.no_grad():
            mu, s = self.m(Xt, ct)
        return pd.DataFrame({
            'mu': mu.squeeze().numpy(),
            'sigma': s.squeeze().numpy(),
            'horse_name': df['horse_name'].values,
            'ketto_num': df['ketto_num'].values if 'ketto_num' in df.columns else '',
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })
