"""
V2 フェーズ1バックテスト — 原点回帰
Gemini分析に基づく構成:
  - 損失: 純Gaussian NLL（ListNetオフ）
  - NN: 3層 (128→64→32) — V1と同じ
  - 学習: 20,000件, 50epoch固定, early stoppingなし
  - カテゴリ: kisyu_code, chokyosi_code (コードベースのみ)
  - 特徴量: V2の44特徴量をそのまま使用（まず損失関数の影響を分離）
"""
import sys
import io
import os
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import qmc, norm

np.random.seed(42)


# ============================================================
# V1互換NN (3層, 純Gaussian NLL)
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


class PredictorPhase1:
    """V1互換Predictor + V2特徴量"""

    def __init__(self, numeric_features, cat_features):
        self.nf = numeric_features
        self.cf = cat_features
        self.sc = StandardScaler()
        self.les = {}
        self.m = None
        self.d = torch.device('cpu')
        self._med = None

    def _prepare(self, df, fit=False):
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
        """純Gaussian NLL — V1と同じ損失関数"""
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
                # 純Gaussian NLL — ListNetなし
                loss = torch.mean(0.5 * torch.log(s ** 2) + 0.5 * ((yt[bi] - mu) / s) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, df):
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


def run():
    from src.features_v2 import FEATURES_V2

    print('=' * 60)
    print('  V2 Phase1: 原点回帰バックテスト')
    print('  損失=純GaussianNLL, NN=3層, ep=50, データ=20K')
    print('  カテゴリ=kisyu_code+chokyosi_code (コードベースのみ)')
    print('=' * 60)

    cache_path = 'data/features_v2_cache.pkl'
    if not os.path.exists(cache_path):
        print('ERROR: features_v2_cache.pkl not found. Run simulate_v2_backtest.py --cache first.')
        return

    feat = pd.read_pickle(cache_path)
    print(f'Loaded: {len(feat):,} rows, {feat["date"].min()} ~ {feat["date"].max()}')

    # 桜花賞特定
    sakura_mask = feat['class_name'].str.contains('桜花賞', na=False)
    g1_mask = (feat['grade_cd'] == 'A') | feat['class_name'].str.contains('G1', na=False)
    sakura_all = feat[sakura_mask & g1_mask]
    sakura_ids = {}
    for rid in sakura_all['race_id'].unique():
        row = sakura_all[sakura_all['race_id'] == rid].iloc[0]
        sakura_ids[row['date'].year] = rid

    print(f'Sakura Sho: {sorted(sakura_ids.keys())}')

    # 特徴量: V2の44特徴量そのまま
    use_f = [f for f in FEATURES_V2 if f in feat.columns]
    # カテゴリ: コードベースのみ（V1互換）
    use_cat = ['kisyu_code', 'chokyosi_code']
    use_cat = [c for c in use_cat if c in feat.columns]

    print(f'Numeric: {len(use_f)}, Categorical: {use_cat}')

    # オッズ欠損フィルタ
    bad_odds = feat['odds'].isna() | (feat['odds'] <= 0)
    print(f'Odds missing/zero: {bad_odds.sum():,} rows ({bad_odds.mean()*100:.1f}%)')

    results = []
    for year in range(2016, 2026):
        if year not in sakura_ids:
            continue

        rid = sakura_ids[year]
        rf = feat[feat['race_id'] == rid].copy()
        if len(rf) == 0:
            continue

        rdate = rf['date'].iloc[0]

        # 学習データ: 20,000件（V1と同じ）, odds欠損を除外
        tr = feat[
            (feat['date'] < rdate) &
            (feat['is_turf'] == 1) &
            (feat['past_count'] > 0) &
            (feat['finish'] > 0)
        ].sort_values('date').tail(20000).copy()

        print(f'\n{year}年桜花賞 ({rdate.strftime("%Y-%m-%d")}):')
        print(f'  学習: {len(tr):,} rows | 出走: {len(rf)} heads')

        pred = PredictorPhase1(use_f, use_cat)
        pred.train(tr, ep=50, lr=0.003, seed=42)
        ps = pred.predict(rf)

        # μの分離度を表示
        mu_range = ps['mu'].max() - ps['mu'].min()
        mu_std = ps['mu'].std()
        print(f'  μ range={mu_range:.4f}, std={mu_std:.4f}')

        # QMC
        from src.qmc_courses import qmc_sim
        mc = qmc_sim(ps, race_features=rf, course='hanshin_turf_1600_outer', n=100000)

        # 評価
        act = rf.sort_values('finish').head(3)
        a3 = set(act['umaban'].astype(int))
        a1 = int(act.iloc[0]['umaban'])

        t5 = mc.head(5)
        t5u = set(t5['umaban'].astype(int))
        ov = len(t5u & a3)
        w = int(t5.iloc[0]['umaban']) == a1

        t5f = []
        print(f'  予測TOP5:')
        for rk, (_, r) in enumerate(t5.iterrows(), 1):
            u = int(r['umaban'])
            a = rf[rf['umaban'] == u]
            af = int(a.iloc[0]['finish']) if len(a) > 0 else 18
            t5f.append(af)
            h = '*' if u in a3 else ' '
            odds_val = r['odds'] if pd.notna(r.get('odds')) else 0.0
            print(f'    {h}{rk}位 [{u:2d}] {r["horse_name"]:16s} '
                  f'勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} '
                  f'μ={ps[ps["umaban"]==u]["mu"].values[0]:.4f} '
                  f'odds={odds_val:.1f} (実際{af}着)')

        print(f'  実際TOP3:')
        for _, r in act.iterrows():
            ninki_val = int(r['ninki']) if pd.notna(r.get('ninki')) else 0
            odds_val = r['odds'] if pd.notna(r.get('odds')) else 0.0
            print(f'    {int(r["finish"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]:16s} '
                  f'({ninki_val}人気 odds:{odds_val})')

        avg = np.mean(t5f)
        ana = any(
            r['odds'] >= 10 and int(r['umaban']) in a3
            for _, r in t5.iterrows() if pd.notna(r.get('odds'))
        )
        results.append({
            'year': year, 'win': w, 'overlap': ov,
            't5': a1 in t5u, 'avg': avg, 'ana': ana, 'p1f': t5f[0],
        })
        sys.stdout.flush()

    # KPIレポート
    if not results:
        print('\nNo results.')
        return

    n = len(results)
    print(f'\n{"="*60}')
    print(f'  Phase1 KPIレポート (純GaussianNLL + 3層NN + 20K)')
    print(f'{"="*60}')

    wn = sum(r['win'] for r in results)
    wr = wn / n * 100
    t5ok = sum(1 for r in results if r['avg'] <= 9)
    an = sum(r['ana'] for r in results)
    o2 = sum(1 for r in results if r['overlap'] >= 2)
    ovr = o2 / n * 100

    print(f'\n  ■ KPI 1: 1着的中率: {wn}/{n} ({wr:.0f}%)')
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5ok}/{n}')
    print(f'  ■ KPI 3: 穴馬検知: {an}/{n}年')
    print(f'  ■ KPI 4: 馬券内占有率: {o2}/{n} ({ovr:.0f}%)')

    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        a = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["p1f"]}着 '
              f'3着内{r["overlap"]}/3 TOP5平均{r["avg"]:.1f}着 {a}')

    # V2フルとの比較サマリ
    print(f'\n  === 比較 ===')
    print(f'  V2(ListNet+4層+40K): 的中0/10, 占有率10%')
    print(f'  Phase1(NLL+3層+20K): 的中{wn}/{n}, 占有率{ovr:.0f}%')


if __name__ == '__main__':
    run()
