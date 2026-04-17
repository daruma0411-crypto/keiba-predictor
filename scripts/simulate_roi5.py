"""
ROI分析: 合計5頭固定、堅実N+穴M の全パターン × 全券種
V1従来5頭と比較
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import qmc as sqmc, norm
from itertools import combinations
np.random.seed(42)

FEATURES_V9 = [
    'wakuban','futan','bataijyu','zogen_sa','heads',
    'past_count','ema_time_zscore','ema_finish',
    'win_rate','top3_rate','avg_run_style',
    'same_dist_finish','same_surface_finish','interval_days',
    'jockey_win_rate','jockey_top3_rate',
    'trainer_win_rate','trainer_top3_rate',
    'avg_jyuni_3c','avg_jyuni_4c',
    'prev_race_class','log_prize_money','weighted_ema_finish',
    'ema_agari','long_stretch_avg','prev_dist_diff',
]
CAT_FEATURES = ['kisyu_code','chokyosi_code','banusi_code','sire_code']

class HorseRaceModel(nn.Module):
    def __init__(self, nf, ed=None):
        super().__init__()
        self.embs = nn.ModuleDict()
        te = 0
        if ed:
            for n, (v, d) in ed.items():
                self.embs[n] = nn.Embedding(v, d); te += d
        self.net = nn.Sequential(
            nn.Linear(nf+te, 128), nn.ReLU(), nn.Dropout(0.3),
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
        self.sc = StandardScaler(); self.les = {}; self.m = None
        self.d = torch.device('cpu'); self._med = None
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
                if fit:
                    le = LabelEncoder(); le.fit(v); self.les[c] = le
                else:
                    le = self.les[c]
                    v = v.map(lambda x: x if x in le.classes_ else le.classes_[0])
                ct[c] = torch.LongTensor(le.transform(v)).to(self.d)
        return Xt, ct
    def train(self, df, ep=50, lr=0.003, bs=256, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        df = df[df['finish']>0].copy()
        y = df['finish'].values.astype(float); h = df['heads'].values.astype(float); h[h==0]=16
        yt = torch.FloatTensor((y-1)/(h-1)).unsqueeze(1).to(self.d)
        Xt, ct = self._p(df, True)
        ed = {}
        for c in self.cf:
            if c in self.les:
                vs = len(self.les[c].classes_); ed[c] = (vs, min(50, max(4,(vs+1)//2)))
        self.m = HorseRaceModel(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr); self.m.train(); n = len(Xt)
        for e in range(ep):
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                bi = idx[i:i+bs]; mu, s = self.m(Xt[bi], {c:t[bi] for c,t in ct.items()})
                l = torch.mean(0.5*torch.log(s**2)+0.5*((yt[bi]-mu)/s)**2)
                opt.zero_grad(); l.backward(); opt.step()
    def predict(self, df):
        self.m.eval(); Xt, ct = self._p(df, False)
        with torch.no_grad(): mu, s = self.m(Xt, ct)
        return pd.DataFrame({
            'mu': mu.squeeze().numpy(), 'sigma': s.squeeze().numpy(),
            'horse_name': df['horse_name'].values, 'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
            'ninki': df['ninki'].values if 'ninki' in df.columns else np.nan,
        })

def qmc_sim(preds, rf=None, n=100000):
    nh = len(preds); mu = preds['mu'].values.copy(); sig = preds['sigma'].values.copy()
    rs = np.full(nh, 2.5); wk = np.arange(1, nh+1)
    if rf is not None and len(rf) == nh:
        if 'avg_run_style' in rf.columns:
            r = rf['avg_run_style'].values; rs = np.where(np.isnan(r), 2.5, r)
        if 'wakuban' in rf.columns: wk = rf['wakuban'].values.astype(float)
    nn_ = np.sum(rs <= 1.5); pb = (nn_-1.5)*0.3
    nd = nh*4+1; s = sqmc.Sobol(d=nd, scramble=True, seed=42)
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
    res = preds[['horse_name','umaban','odds','ninki']].copy()
    for pos in range(1, min(nh+1, 19)):
        res[f'prob_{pos}'] = (rk==pos).mean(axis=0)
    res['expected_rank'] = rk.mean(axis=0); res['win_prob'] = res['prob_1']
    res['top3_prob'] = res[['prob_1','prob_2','prob_3']].sum(axis=1)
    return res.sort_values('expected_rank')


def run():
    from src.binary_parser import load_hanshin_data
    print('Loading...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    feat = pd.read_pickle('data/features_v9b_cache.pkl')
    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}
    use_f = [f for f in FEATURES_V9 if f in feat.columns]

    mc_cache = {}
    for year in range(2016, 2026):
        if year not in sakura_ids: continue
        rid = sakura_ids[year]
        rd = df_h[df_h['race_id'] == rid]; rdate = rd['date'].iloc[0]
        rf = feat[feat['race_id'] == rid].copy()
        if len(rf) == 0: continue
        if 'ninki' not in rf.columns: rf['ninki'] = rf['odds'].rank(method='first')
        tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()
        pred = Predictor(use_f, CAT_FEATURES)
        pred.train(tr, ep=50, lr=0.003, seed=42)
        ps = pred.predict(rf)
        mc = qmc_sim(ps, rf, n=100000)
        act = rd.sort_values('kakutei_jyuni').head(3)
        a3_list = [(int(r['kakutei_jyuni']), int(r['umaban']), float(r['odds']), int(r['ninki'])) for _, r in act.iterrows()]
        odds_map = {int(r['umaban']): float(r['odds']) for _, r in rd.iterrows()}
        mc_cache[year] = (mc, rd, a3_list, odds_map)
        print(f'  {year} done')

    # 5頭固定の全パターン + V1従来
    configs = [
        ('V1従来(5)', 5, 0, 0),
        ('堅0+穴5(NN純粋)', 5, 0, 99),
    ]
    for cutoff in [3, 4, 5, 6]:
        for n_pop in range(1, min(6, cutoff+1)):
            n_disc = 5 - n_pop
            configs.append((f'堅{n_pop}+穴{n_disc}/<={cutoff}', 5, n_pop, cutoff))

    print('\n' + '=' * 120)
    print('  5頭固定 ROI分析: 全配分 × 全券種 (10年, 1点100円)')
    print('=' * 120)

    # ヘッダー
    print(f'\n  {"配分":<18s} | {"占有":>4s} {"的中":>4s} | '
          f'{"単勝ROI":>8s} {"複勝ROI":>8s} {"ワイドROI":>9s} {"馬連ROI":>8s} {"3連複ROI":>9s} | '
          f'{"単勝配当":>10s} {"ワイド配当":>10s} {"馬連配当":>10s} {"3連複配当":>10s}')
    print('-' * 120)

    for label, total, n_pop, cutoff in configs:
        total_cost = {t: 0 for t in ['単勝','複勝','ワイド','馬連','3連複']}
        total_pay = {t: 0 for t in ['単勝','複勝','ワイド','馬連','3連複']}
        total_hits = {t: 0 for t in ['単勝','複勝','ワイド','馬連','3連複']}

        for year, (mc, rd, a3_list, odds_map) in mc_cache.items():
            a_set = {a[1] for a in a3_list}
            a1, a2, a3 = a3_list[0][1], a3_list[1][1], a3_list[2][1]
            o1, o2, o3 = a3_list[0][2], a3_list[1][2], a3_list[2][2]

            if cutoff == 0:
                sel = list(mc.head(total)['umaban'].astype(int))
            elif cutoff == 99:
                sel = list(mc.head(total)['umaban'].astype(int))
            else:
                pop = mc[mc['ninki'] <= cutoff].head(n_pop)
                disc = mc[mc['ninki'] > cutoff].head(5 - n_pop)
                sel_df = pd.concat([pop, disc])
                sel = list(sel_df['umaban'].astype(int))
            sel_set = set(sel)

            n = len(sel)
            pairs = list(combinations(sel, 2))
            triples = list(combinations(sel, 3))

            # 単勝
            total_cost['単勝'] += n * 100
            if a1 in sel_set:
                total_pay['単勝'] += o1 * 100
                total_hits['単勝'] += 1

            # 複勝
            total_cost['複勝'] += n * 100
            for u in sel:
                if u in a_set:
                    total_pay['複勝'] += odds_map.get(u, 1) * 0.25 * 100
                    total_hits['複勝'] += 1

            # ワイド
            total_cost['ワイド'] += len(pairs) * 100
            in_top3 = sel_set & a_set
            for p in combinations(in_top3, 2):
                oa, ob = odds_map.get(p[0], 1), odds_map.get(p[1], 1)
                total_pay['ワイド'] += (oa * ob) ** 0.5 * 70
                total_hits['ワイド'] += 1

            # 馬連
            total_cost['馬連'] += len(pairs) * 100
            if a1 in sel_set and a2 in sel_set:
                total_pay['馬連'] += o1 * o2 * 8
                total_hits['馬連'] += 1

            # 3連複
            total_cost['3連複'] += len(triples) * 100
            if a_set.issubset(sel_set):
                total_pay['3連複'] += o1 * o2 * o3 * 1.5
                total_hits['3連複'] += 1

        roi = {}
        for t in ['単勝','複勝','ワイド','馬連','3連複']:
            roi[t] = total_pay[t] / total_cost[t] * 100 if total_cost[t] > 0 else 0

        best_mark = ''
        if roi['単勝'] >= 100 or roi['ワイド'] >= 100 or roi['馬連'] >= 100 or roi['3連複'] >= 100:
            best_mark = ' ★'

        ov_count = 0
        win_count = 0
        for year, (mc, rd, a3_list, odds_map) in mc_cache.items():
            a_set = {a[1] for a in a3_list}
            a1 = a3_list[0][1]
            if cutoff == 0 or cutoff == 99:
                sel = set(mc.head(total)['umaban'].astype(int))
            else:
                pop = mc[mc['ninki'] <= cutoff].head(n_pop)
                disc = mc[mc['ninki'] > cutoff].head(5 - n_pop)
                sel = set(pd.concat([pop, disc])['umaban'].astype(int))
            if len(sel & a_set) >= 2: ov_count += 1
            if a1 in sel: win_count += 1

        print(f'  {label:<18s} | {ov_count:>3d}/10 {win_count:>3d}/10 | '
              f'{roi["単勝"]:>7.1f}% {roi["複勝"]:>7.1f}% {roi["ワイド"]:>8.1f}% {roi["馬連"]:>7.1f}% {roi["3連複"]:>8.1f}% | '
              f'{total_pay["単勝"]:>9,.0f}円 {total_pay["ワイド"]:>9,.0f}円 {total_pay["馬連"]:>9,.0f}円 {total_pay["3連複"]:>9,.0f}円{best_mark}')


if __name__ == '__main__':
    run()
