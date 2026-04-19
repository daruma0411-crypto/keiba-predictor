"""
2017年桜花賞 v9b詳細出力版
ディベート用に全馬の詳細データを出力
"""
import sys
import io
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

# stdout wrapping must happen before any import that also wraps it
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
    'ema_agari', 'long_stretch_avg', 'prev_dist_diff',
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

def run_2017_detail():
    from src.binary_parser import load_all_data, load_hanshin_data
    from src.um_parser import load_um_data

    print('Loading data...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    df_all = load_all_data(years=range(2014, 2026))
    df_all = df_all[df_all['kakutei_jyuni'] > 0]

    feat = pd.read_pickle('data/features_v9b_cache.pkl')

    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    use_f = [f for f in FEATURES_V9 if f in feat.columns]

    year = 2017
    if year not in sakura_ids:
        print(f'ERROR: {year}年の桜花賞データが見つかりません')
        print(f'利用可能な年: {sorted(sakura_ids.keys())}')
        return

    rid = sakura_ids[year]
    rd = df_h[df_h['race_id'] == rid]
    rdate = rd['date'].iloc[0]
    rf = feat[feat['race_id'] == rid].copy()

    tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()

    print(f'Training on {len(tr)} samples...')
    pred = Predictor(use_f, CAT_FEATURES)
    pred.train(tr, ep=50, lr=0.003, seed=42)
    ps = pred.predict(rf)
    mc = qmc_sim(ps, race_features=rf, n=100000)

    # 詳細出力
    print(f'\n{"="*80}')
    print(f'  2017年桜花賞 v9b 詳細シミュレーション結果')
    print(f'{"="*80}')

    # 全馬データ出力
    merged = mc.merge(rf[['umaban','ketto_num','ema_time_zscore','ema_finish','ema_agari',
                           'long_stretch_avg','prev_dist_diff','avg_run_style',
                           'wakuban','futan','past_count','win_rate','top3_rate',
                           'same_dist_finish','same_surface_finish','weighted_ema_finish',
                           'prev_race_class','log_prize_money','interval_days',
                           'jockey_win_rate','jockey_top3_rate']],
                      on='umaban', how='left')

    # 実際の着順も追加
    act_map = dict(zip(rd['umaban'].astype(int), rd['kakutei_jyuni'].astype(int)))
    ninki_map = dict(zip(rd['umaban'].astype(int), rd['ninki'].astype(int)))
    jockey_map = dict(zip(rd['umaban'].astype(int), rd['kisyu_name']))

    print(f'\n■ 全馬予測一覧（期待順位順）:')
    print(f'{"順":>2} {"馬番":>3} {"馬名":<16} {"勝率":>6} {"複勝率":>6} {"期待順":>6} {"odds":>6} {"EV勝":>5} '
          f'{"タイム偏差":>8} {"末脚EMA":>7} {"直線長":>6} {"距離差":>5} {"脚質":>4} {"経験":>3} {"騎手":<10} {"実着":>4}')
    print('-'*130)

    for rk, (_, r) in enumerate(merged.iterrows(), 1):
        u = int(r['umaban'])
        af = act_map.get(u, 99)
        nk = ninki_map.get(u, 99)
        jk = jockey_map.get(u, '')
        rs_label = {0:'不明', 1:'逃', 1.5:'逃先', 2:'先', 2.5:'先差', 3:'差', 3.5:'差追', 4:'追'}
        rs_v = r.get('avg_run_style', 2.5)
        rs_str = rs_label.get(round(rs_v*2)/2, f'{rs_v:.1f}')
        ev = r.get('ev_win', 0)
        ls = r.get('long_stretch_avg', np.nan)
        ls_str = f'{ls:.1f}' if pd.notna(ls) else 'N/A'

        print(f'{rk:>2} [{u:2d}] {r["horse_name"]:<16} {r["win_prob"]:5.1%} {r["top3_prob"]:5.1%} '
              f'{r["expected_rank"]:5.1f}  {r["odds"]:5.1f} {ev:5.2f} '
              f'{r.get("ema_time_zscore",0):7.3f}  {r.get("ema_agari",0):6.3f} {ls_str:>6} {r.get("prev_dist_diff",0):5.0f} '
              f'{rs_str:>4} {int(r.get("past_count",0)):>3} {jk:<10} {af:>3}着')

    # MCでの詳細確率分布
    print(f'\n■ MC確率分布（上位10頭）:')
    top10 = merged.head(10)
    print(f'{"馬番":>3} {"馬名":<16} {"1着":>5} {"2着":>5} {"3着":>5} {"4着":>5} {"5着":>5} {"6着~":>5} {"期待順":>6}')
    print('-'*70)
    for _, r in top10.iterrows():
        u = int(r['umaban'])
        p1 = r.get('prob_1', 0)
        p2 = r.get('prob_2', 0)
        p3 = r.get('prob_3', 0)
        p4 = r.get('prob_4', 0)
        p5 = r.get('prob_5', 0)
        p6_ = 1 - p1 - p2 - p3 - p4 - p5
        print(f'[{u:2d}] {r["horse_name"]:<16} {p1:4.1%} {p2:4.1%} {p3:4.1%} {p4:4.1%} {p5:4.1%} {p6_:4.1%} {r["expected_rank"]:5.1f}')

    # 脚質・展開分析
    print(f'\n■ 脚質分布:')
    for _, r in merged.iterrows():
        rs = r.get('avg_run_style', 2.5)
        if rs <= 1.5:
            label = '逃げ/先行'
        elif rs <= 2.5:
            label = '先行/差し'
        else:
            label = '差し/追込'
        u = int(r['umaban'])
        print(f'  [{u:2d}] {r["horse_name"]:<16} 脚質値={rs:.2f} ({label}) 枠={int(r.get("wakuban",0))} 斤量={r.get("futan",0):.0f}')

    # NNの生μ/σ出力
    print(f'\n■ NN生出力（μ/σ）:')
    ps_sorted = ps.sort_values('mu')
    for _, r in ps_sorted.iterrows():
        u = int(r['umaban'])
        print(f'  [{u:2d}] {r["horse_name"]:<16} μ={r["mu"]:.4f} σ={r["sigma"]:.4f} (μ小=高評価)')

    # 実際の結果
    print(f'\n■ 実際の結果:')
    act = rd.sort_values('kakutei_jyuni')
    for _, r in act.iterrows():
        print(f'  {int(r["kakutei_jyuni"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]:<16} ({int(r["ninki"])}人気 odds:{r["odds"]})')

    print(f'\n■ サマリー:')
    t5 = mc.head(5)
    a3 = set(rd.sort_values('kakutei_jyuni').head(3)['umaban'].astype(int))
    t5u = set(t5['umaban'].astype(int))
    ov = len(t5u & a3)
    print(f'  予測1位→実際{act_map.get(int(t5.iloc[0]["umaban"]),99)}着')
    print(f'  馬券内占有率: {ov}/3')

if __name__ == '__main__':
    run_2017_detail()
