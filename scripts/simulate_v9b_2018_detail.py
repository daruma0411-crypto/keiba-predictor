"""
2018年桜花賞 v9b詳細シミュレーション
"""
import sys
import io
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import qmc, norm

np.random.seed(42)

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
    'ema_agari',
    'long_stretch_avg',
    'prev_dist_diff',
]
CAT_FEATURES = ['kisyu_code', 'chokyosi_code', 'banusi_code', 'sire_code']

class HorseRaceModel(nn.Module):
    def __init__(self, nf, ed=None):
        super().__init__()
        self.embs = nn.ModuleDict()
        te = 0
        if ed:
            for n, (v, d) in ed.items():
                self.embs[n] = nn.Embedding(v, d); te += d
        self.net = nn.Sequential(
            nn.Linear(nf + te, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, 1)
        self.sig = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x, c=None):
        p = [x]
        if c:
            for n, i in c.items():
                if n in self.embs: p.append(self.embs[n](i))
        h = self.net(torch.cat(p, 1))
        return self.mu(h), self.sig(h) + 0.1

class Predictor:
    def __init__(self, nf, cf=None):
        self.nf, self.cf = nf, cf or []
        self.sc = StandardScaler(); self.les = {}; self.m = None; self.d = torch.device('cpu'); self._med = None

    def _p(self, df, fit=False):
        X = df[self.nf].copy()
        for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
        if fit:
            self._med = X.median(); X = X.fillna(self._med); Xs = self.sc.fit_transform(X)
        else:
            X = X.fillna(self._med); Xs = self.sc.transform(X)
        Xt = torch.FloatTensor(Xs).to(self.d)
        ct = {}
        for c in self.cf:
            if c in df.columns:
                v = df[c].fillna('unknown').astype(str)
                if fit: le = LabelEncoder(); le.fit(v); self.les[c] = le
                else:
                    le = self.les[c]; v = v.map(lambda x: x if x in le.classes_ else le.classes_[0])
                ct[c] = torch.LongTensor(le.transform(v)).to(self.d)
        return Xt, ct

    def train(self, df, ep=50, lr=0.003, bs=256, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float); h = df['heads'].values.astype(float); h[h==0]=16
        yt = torch.FloatTensor((y-1)/(h-1)).unsqueeze(1).to(self.d)
        Xt, ct = self._p(df, True)
        ed = {}
        for c in self.cf:
            if c in self.les: vs = len(self.les[c].classes_); ed[c] = (vs, min(50, max(4, (vs+1)//2)))
        self.m = HorseRaceModel(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr); self.m.train(); n = len(Xt)
        for e in range(ep):
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                bi = idx[i:i+bs]; mu, s = self.m(Xt[bi], {c: t[bi] for c, t in ct.items()})
                l = torch.mean(0.5*torch.log(s**2) + 0.5*((yt[bi]-mu)/s)**2)
                opt.zero_grad(); l.backward(); opt.step()

    def predict(self, df):
        self.m.eval(); Xt, ct = self._p(df, False)
        with torch.no_grad(): mu, s = self.m(Xt, ct)
        return pd.DataFrame({'mu': mu.squeeze().numpy(), 'sigma': s.squeeze().numpy(),
            'horse_name': df['horse_name'].values, 'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values, 'odds': df['odds'].values if 'odds' in df.columns else np.nan})

def qmc_sim(preds, rf=None, race_features=None, n=100000):
    rf = rf or race_features
    nh = len(preds); mu = preds['mu'].values.copy(); sig = preds['sigma'].values.copy()
    rs = np.full(nh, 2.5); wk = np.arange(1, nh+1)
    if rf is not None and len(rf) == nh:
        if 'avg_run_style' in rf.columns:
            r = rf['avg_run_style'].values; rs = np.where(np.isnan(r), 2.5, r)
        if 'wakuban' in rf.columns: wk = rf['wakuban'].values.astype(float)
    nn_ = np.sum(rs <= 1.5); pb = (nn_-1.5)*0.3
    nd = nh*4+1; s = qmc.Sobol(d=nd, scramble=True, seed=42)
    np2 = 2**int(np.ceil(np.log2(n))); sb = s.random(np2)[:n]
    sn = norm.ppf(np.clip(sb, 0.001, 0.999))
    j = 0
    ab = mu[np.newaxis,:] + sig[np.newaxis,:]*sn[:,j:j+nh]; j+=nh
    p = pb + 0.15*sn[:,j]; j+=1; lu = sb[:,j:j+nh]; j+=nh; bu = sb[:,j:j+nh]; j+=nh; pn = sn[:,j:j+nh]
    p2 = p[:,np.newaxis]
    sa = (rs>=3.0).astype(float); ni = (rs<=1.5).astype(float); se = ((rs>1.5)&(rs<=2.5)).astype(float)
    ab -= p2*0.04*sa[np.newaxis,:]; ab += p2*0.06*ni[np.newaxis,:]; ab += p2*0.02*se[np.newaxis,:]
    ab -= 0.01*((wk<=3)&(rs<=2.5)).astype(float)[np.newaxis,:]
    ab += 0.01*((wk>=6)&(rs>=3.5)).astype(float)[np.newaxis,:]
    ab += (lu<0.05).astype(float)*0.15
    is_ = ((wk<=3)&(rs>=3.0)).astype(float)
    ab += (bu<0.10).astype(float)*is_[np.newaxis,:]*0.10
    ab += 0.02*pn
    rk = np.argsort(np.argsort(ab, axis=1), axis=1)+1
    res = preds[['horse_name','umaban','odds']].copy()
    for pos in range(1, min(nh+1, 19)): res[f'prob_{pos}'] = (rk==pos).mean(axis=0)
    res['expected_rank'] = rk.mean(axis=0); res['win_prob'] = res['prob_1']
    res['top3_prob'] = res[['prob_1','prob_2','prob_3']].sum(axis=1)
    if 'odds' in res.columns: res['ev_win'] = res['win_prob']*res['odds']
    return res.sort_values('expected_rank')


def run_2018():
    from src.binary_parser import load_all_data, load_hanshin_data
    from src.um_parser import load_um_data

    print('Loading...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    df_all = load_all_data(years=range(2014, 2026))
    df_all = df_all[df_all['kakutei_jyuni'] > 0]
    um = load_um_data(years=range(2010, 2026))
    feat = pd.read_pickle('data/features_all_v4.pkl')

    cache_path = 'data/features_v9b_cache.pkl'
    if os.path.exists(cache_path):
        print('Loading cached v9b features...')
        feat = pd.read_pickle(cache_path)

    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    year = 2018
    if year not in sakura_ids:
        print(f'{year}年桜花賞が見つかりません')
        return

    rid = sakura_ids[year]
    rd = df_h[df_h['race_id'] == rid]
    rdate = rd['date'].iloc[0]
    rf = feat[feat['race_id'] == rid].copy()

    if len(rf) == 0:
        print('特徴量データが見つかりません')
        return

    use_f = [f for f in FEATURES_V9 if f in feat.columns]
    tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()

    print('Training model...')
    pred = Predictor(use_f, CAT_FEATURES)
    pred.train(tr, ep=50, lr=0.003, seed=42)
    ps = pred.predict(rf)
    mc = qmc_sim(ps, race_features=rf, n=100000)

    print(f'\n{"="*100}')
    print(f'  2018年桜花賞 v9b 詳細シミュレーション結果')
    print(f'{"="*100}')
    print(f'レースID: {rid}')
    print(f'日付: {rdate}')
    print(f'出走頭数: {len(rf)}')

    print(f'\n■ 全馬データ（v9b予測順位順）')
    print(f'-'*100)

    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        horse_data = rd[rd['umaban'] == u]
        feat_data = rf[rf['umaban'] == u]

        if len(horse_data) == 0 or len(feat_data) == 0:
            continue

        hd = horse_data.iloc[0]
        fd = feat_data.iloc[0]

        print(f'\n  【{rank}位】 {r["horse_name"]}')
        print(f'    馬番: {u}  枠番: {int(fd["wakuban"])}  オッズ: {r["odds"]:.1f}')
        print(f'    NN出力: mu={ps[ps["umaban"]==u]["mu"].values[0]:.4f}  sigma={ps[ps["umaban"]==u]["sigma"].values[0]:.4f}')
        print(f'    勝率: {r["win_prob"]:.2%}  複勝率: {r["top3_prob"]:.2%}  期待順位: {r["expected_rank"]:.2f}')

        print(f'    --- 能力指標 ---')
        print(f'    EMA着順: {fd["ema_finish"]:.3f}')
        print(f'    EMAタイム偏差値: {fd["ema_time_zscore"]:.3f}')
        print(f'    末脚指標(ema_agari): {fd["ema_agari"]:.3f}')
        print(f'    加重EMA着順: {fd["weighted_ema_finish"]:.3f}')

        print(f'    --- 適性指標 ---')
        ls_val = fd.get("long_stretch_avg")
        print(f'    直線長コース着順: {ls_val:.3f}' if pd.notna(ls_val) else f'    直線長コース着順: データなし')
        sd_val = fd.get("same_dist_finish")
        print(f'    同距離着順: {sd_val:.3f}' if pd.notna(sd_val) else f'    同距離着順: データなし')
        ss_val = fd.get("same_surface_finish")
        print(f'    同馬場着順: {ss_val:.3f}' if pd.notna(ss_val) else f'    同馬場着順: データなし')
        print(f'    前走距離差: {fd["prev_dist_diff"]:.0f}m')

        print(f'    --- 展開指標 ---')
        print(f'    脚質(avg_run_style): {fd["avg_run_style"]:.2f}')
        print(f'    3角平均順位: {fd["avg_jyuni_3c"]:.2f}')
        print(f'    4角平均順位: {fd["avg_jyuni_4c"]:.2f}')

        print(f'    --- コンディション ---')
        print(f'    斤量: {fd["futan"]:.1f}kg  馬体重: {fd["bataijyu"]:.0f}kg  増減: {fd["zogen_sa"]:.0f}kg')
        print(f'    間隔日数: {fd["interval_days"]:.0f}日  過去走数: {fd["past_count"]:.0f}')

        print(f'    --- 騎手・調教師 ---')
        print(f'    騎手勝率: {fd["jockey_win_rate"]:.3f}  騎手複勝率: {fd["jockey_top3_rate"]:.3f}')
        print(f'    調教師勝率: {fd["trainer_win_rate"]:.3f}  調教師複勝率: {fd["trainer_top3_rate"]:.3f}')

        print(f'    前走クラス: {fd["prev_race_class"]:.0f}  累積賞金(log): {fd["log_prize_money"]:.3f}')
        print(f'    勝率(実績): {fd["win_rate"]:.3f}  複勝率(実績): {fd["top3_rate"]:.3f}')

        probs = [f'{r.get(f"prob_{i}", 0):.3f}' for i in range(1, 7)]
        print(f'    着順確率(1-6着): {" / ".join(probs)}')

    sys.stdout.flush()

if __name__ == '__main__':
    run_2018()
