"""堅2+穴3/<=5 vs V1 年別一覧"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import qmc as sqmc, norm
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

    for year in range(2016, 2026):
        if year not in sakura_ids:
            continue
        rid = sakura_ids[year]
        rd = df_h[df_h['race_id'] == rid]
        rdate = rd['date'].iloc[0]
        rf = feat[feat['race_id'] == rid].copy()
        if len(rf) == 0:
            continue
        if 'ninki' not in rf.columns:
            rf['ninki'] = rf['odds'].rank(method='first')

        tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()
        pred = Predictor(use_f, CAT_FEATURES)
        pred.train(tr, ep=50, lr=0.003, seed=42)
        ps = pred.predict(rf)
        mc = qmc_sim(ps, rf, n=100000)

        act = rd.sort_values('kakutei_jyuni').head(3)
        a_map = {}
        for _, r in act.iterrows():
            a_map[int(r['umaban'])] = (int(r['kakutei_jyuni']), r['horse_name'].strip(), int(r['ninki']), float(r['odds']))
        a_set = set(a_map.keys())

        # 堅2+穴3/<=5
        pop = mc[mc['ninki'] <= 5].head(2)
        disc = mc[mc['ninki'] > 5].head(3)
        sel = pd.concat([pop, disc]).sort_values('expected_rank')
        pop_uma = set(pop['umaban'].astype(int))

        # V1
        v1 = mc.head(5)

        ov_new = len(set(sel['umaban'].astype(int)) & a_set)
        ov_v1 = len(set(v1['umaban'].astype(int)) & a_set)

        # 結果行
        result_str = ''
        for _, r in act.iterrows():
            result_str += f"{int(r['kakutei_jyuni'])}着{r['horse_name'].strip()}({int(r['ninki'])}人気/{r['odds']:.1f}倍)  "

        print(f'')
        print(f'=== {year}年桜花賞 ===')
        print(f'  結果: {result_str}')

        # 堅2+穴3
        line = f'  堅2穴3 [{ov_new}/3]: '
        for _, r in sel.iterrows():
            u = int(r['umaban'])
            nk = int(r['ninki']) if pd.notna(r['ninki']) else 0
            nm = r['horse_name'].strip()
            is_pop = u in pop_uma
            tag = '\u2605' if is_pop else '\u2606'  # ★=本命 ☆=穴
            if u in a_map:
                fin = a_map[u][0]
                hit = f'-> {fin}着!'
            else:
                fin_row = rd[rd['umaban'] == u]
                fin = int(fin_row.iloc[0]['kakutei_jyuni']) if len(fin_row) > 0 else 18
                hit = f'-> {fin}着'
            line += f'{tag}{nm}({nk}人気/{r["odds"]:.1f}倍){hit}  '
        print(line)

        # V1
        line = f'  V1従来 [{ov_v1}/3]: '
        for _, r in v1.iterrows():
            u = int(r['umaban'])
            nk = int(r['ninki']) if pd.notna(r['ninki']) else 0
            nm = r['horse_name'].strip()
            if u in a_map:
                fin = a_map[u][0]
                hit = f'-> {fin}着!'
            else:
                fin_row = rd[rd['umaban'] == u]
                fin = int(fin_row.iloc[0]['kakutei_jyuni']) if len(fin_row) > 0 else 18
                hit = f'-> {fin}着'
            line += f'{nm}({nk}人気/{r["odds"]:.1f}倍){hit}  '
        print(line)

    # サマリー
    print('')
    print('=' * 60)
    print('  ★=本命(5番人気以内からQMC上位2頭)')
    print('  ☆=穴(6番人気以降からQMC上位3頭)')
    print('  単勝ROI: 堅2穴3=139.6%  V1=74.8%')
    print('  馬連ROI: 堅2穴3=56.9%   V1=5.8%')


if __name__ == '__main__':
    run()
