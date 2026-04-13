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
    race_ids : array-like - レースID（同一レース内で順位比較）
    alpha_rank : float - ランキング損失の重み
    """
    # Gaussian NLL (従来)
    nll = torch.mean(0.5 * torch.log(sigma ** 2) + 0.5 * ((y_true - mu) / sigma) ** 2)

    # ListNet: 同一レース内での順位確率分布の比較
    # (バッチ内でレースをグループ化するのは重いので、近似として全体でソフトマックス比較)
    # P_true = softmax(-y_true) (着順が小さいほど高確率)
    # P_pred = softmax(-mu)
    p_true = F.softmax(-y_true.squeeze(), dim=0)
    p_pred = F.softmax(-mu.squeeze(), dim=0)
    rank_loss = -torch.sum(p_true * torch.log(p_pred + 1e-8))

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
        self.cf = cat_features or ['kisyu_code', 'chokyosi_code', 'sire_type']
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

    def train(self, df, ep=80, lr=0.003, bs=256, seed=42, alpha_rank=0.3):
        """
        学習（Gaussian NLL + ListNet ハイブリッド損失）

        Parameters
        ----------
        ep : int - エポック数 (デフォルト80に増加)
        alpha_rank : float - ランキング損失の重み
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float)
        h = df['heads'].values.astype(float)
        h[h == 0] = 16
        yt = torch.FloatTensor((y - 1) / (h - 1)).unsqueeze(1).to(self.d)

        # race_ids for ranking loss
        race_ids = df['race_id'].values if 'race_id' in df.columns else None

        Xt, ct = self._prepare(df, True)

        ed = {}
        for c in self.cf:
            if c in self.les:
                vs = len(self.les[c].classes_)
                ed[c] = (vs, min(50, max(4, (vs + 1) // 2)))

        self.m = HorseRaceModelV2(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)

        self.m.train()
        n = len(Xt)

        for e in range(ep):
            idx = torch.randperm(n)
            epoch_loss = 0
            batches = 0
            for i in range(0, n, bs):
                bi = idx[i:i + bs]
                mu, s = self.m(Xt[bi], {c: t[bi] for c, t in ct.items()})

                # ハイブリッド損失
                loss = listnet_loss(mu, s, yt[bi], None, alpha_rank)

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                batches += 1

            scheduler.step()

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
