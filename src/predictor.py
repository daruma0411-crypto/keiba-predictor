"""
v9b ベースNNモデル（共通モジュール）
全レース共通で使用する学習・予測エンジン
各スクリプトからコピペされていたNN/Predictor/特徴量定義をここに集約
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================================
# 特徴量定義
# ============================================================
FEATURES_V9 = [
    'wakuban', 'futan', 'bataijyu', 'zogen_sa', 'heads',
    'past_count', 'ema_time_zscore', 'ema_finish',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'avg_jyuni_3c', 'avg_jyuni_4c',
    'prev_race_class', 'log_prize_money',
    'weighted_ema_finish',
    # v9特徴量
    'ema_agari',         # 末脚指標
    'long_stretch_avg',  # 直線長コース着順
    'prev_dist_diff',    # 前走距離差
]

CAT_FEATURES = ['kisyu_code', 'chokyosi_code', 'banusi_code', 'sire_code']


# ============================================================
# NNモデル
# ============================================================
class HorseRaceModel(nn.Module):
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
# Predictor（学習・推論ラッパー）
# ============================================================
class Predictor:
    """
    汎用の学習・推論クラス
    どのレースでも同一のインターフェースで使える

    使い方:
        pred = Predictor()
        pred.train(train_df, ep=50, lr=0.003)
        preds = pred.predict(race_df)
    """

    def __init__(self, numeric_features=None, cat_features=None):
        self.nf = numeric_features or FEATURES_V9
        self.cf = cat_features or CAT_FEATURES
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

    def train(self, df, ep=50, lr=0.003, bs=256, seed=42):
        """Gaussian NLLでNNを学習"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float)
        h = df['heads'].values.astype(float)
        h[h == 0] = 16
        yt = torch.FloatTensor((y - 1) / (h - 1)).unsqueeze(1).to(self.d)

        Xt, ct = self._prepare(df, True)

        ed = {}
        for c in self.cf:
            if c in self.les:
                vs = len(self.les[c].classes_)
                ed[c] = (vs, min(50, max(4, (vs + 1) // 2)))

        self.m = HorseRaceModel(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr)
        self.m.train()
        n = len(Xt)

        for e in range(ep):
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                bi = idx[i:i + bs]
                mu, s = self.m(Xt[bi], {c: t[bi] for c, t in ct.items()})
                loss = torch.mean(0.5 * torch.log(s ** 2) + 0.5 * ((yt[bi] - mu) / s) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()

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
            'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })
