"""
ROI逆算: 各配分 × 各買い方でROIを計算
買い方: 単勝BOX, 複勝BOX, ワイドBOX, 馬連BOX, 3連複BOX
配当は実オッズから推定（単勝=正確、他=近似）
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


def estimate_payouts(sel_umabans, act_top3, odds_map, heads):
    """
    選抜馬番セット × 実際TOP3 → 各券種の配当推定
    odds_map: {umaban: odds}
    act_top3: [(着順, umaban, odds), ...] 着順昇順
    """
    sel = set(sel_umabans)
    a1, a2, a3 = act_top3[0][1], act_top3[1][1], act_top3[2][1]
    o1, o2, o3 = act_top3[0][2], act_top3[1][2], act_top3[2][2]
    a_set = {a1, a2, a3}

    results = {}

    # 単勝BOX: N点買い。1着が含まれていれば的中
    n = len(sel)
    cost_tansho = n * 100
    if a1 in sel:
        payout_tansho = o1 * 100
    else:
        payout_tansho = 0
    results['単勝'] = {'cost': cost_tansho, 'payout': payout_tansho}

    # 複勝BOX: N点買い。TOP3に含まれる各馬で的中
    cost_fukusho = n * 100
    payout_fukusho = 0
    for u in sel:
        if u in a_set:
            o = odds_map.get(u, 1)
            # 複勝配当 ≈ 単勝オッズの20-30%。18頭立てG1では概ね25%
            payout_fukusho += o * 0.25 * 100
    results['複勝'] = {'cost': cost_fukusho, 'payout': payout_fukusho}

    # ワイドBOX: C(N,2)点。TOP3のうち2頭が含まれていれば的中（複数的中あり）
    pairs = list(combinations(sel, 2))
    cost_wide = len(pairs) * 100
    payout_wide = 0
    # 的中ペア: sel内のTOP3馬の全2-組合せ
    in_top3 = sel & a_set
    for p in combinations(in_top3, 2):
        oa, ob = odds_map.get(p[0], 1), odds_map.get(p[1], 1)
        # ワイド配当 ≈ sqrt(odds_a * odds_b) * 70
        payout_wide += (oa * ob) ** 0.5 * 70
    results['ワイド'] = {'cost': cost_wide, 'payout': payout_wide}

    # 馬連BOX: C(N,2)点。1着-2着の組合せが含まれていれば的中
    cost_umaren = len(pairs) * 100
    payout_umaren = 0
    if a1 in sel and a2 in sel:
        # 馬連配当 ≈ odds_1 * odds_2 * 8
        payout_umaren = o1 * o2 * 8
    results['馬連'] = {'cost': cost_umaren, 'payout': payout_umaren}

    # 3連複BOX: C(N,3)点。TOP3の3頭が全て含まれていれば的中
    triples = list(combinations(sel, 3))
    cost_sanren = len(triples) * 100
    payout_sanren = 0
    if a_set.issubset(sel):
        # 3連複配当 ≈ odds_1 * odds_2 * odds_3 * 1.5
        payout_sanren = o1 * o2 * o3 * 1.5
    results['3連複'] = {'cost': cost_sanren, 'payout': payout_sanren}

    return results


def run():
    from src.binary_parser import load_hanshin_data
    print('Loading...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    feat = pd.read_pickle('data/features_v9b_cache.pkl')
    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}
    use_f = [f for f in FEATURES_V9 if f in feat.columns]

    # QMCキャッシュ
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
        a3_list = [(int(r['kakutei_jyuni']), int(r['umaban']), float(r['odds'])) for _, r in act.iterrows()]
        odds_map = {int(r['umaban']): float(r['odds']) for _, r in rd.iterrows()}
        mc_cache[year] = (mc, rd, a3_list, odds_map)
        print(f'  {year} done')

    # 実際のTOP3表示
    print('\n=== 実際の結果 ===')
    for year, (mc, rd, a3_list, odds_map) in mc_cache.items():
        top3_str = ', '.join(f'{a[1]}番({a[2]:.1f}倍)' for a in a3_list)
        print(f'  {year}: {top3_str}')

    # テスト対象の配分
    configs = [
        # (label, total, n_pop, cutoff)
        ('V1従来5頭', 5, 0, 0),  # special: QMC TOP5
        ('堅5+穴0/<=5', 5, 5, 5),
        ('堅6+穴0/<=6', 6, 6, 6),
        ('堅4+穴3/<=4', 7, 4, 4),
        ('堅5+穴2/<=5', 7, 5, 5),
        ('堅4+穴4/<=4', 8, 4, 4),
        ('堅5+穴3/<=5', 8, 5, 5),
        ('堅4+穴4/<=5', 8, 4, 5),
        ('堅3+穴7/<=4', 10, 3, 4),
    ]

    print('\n' + '=' * 110)
    print('  ROI分析: 配分 × 券種 (10年間合計, 1レース1点100円均等)')
    print('=' * 110)

    for label, total, n_pop, cutoff in configs:
        # 10年分の選抜と配当計算
        total_by_type = {}
        for btype in ['単勝', '複勝', 'ワイド', '馬連', '3連複']:
            total_by_type[btype] = {'cost': 0, 'payout': 0, 'hits': 0}

        year_details = []
        for year, (mc, rd, a3_list, odds_map) in mc_cache.items():
            a3_set = {a[1] for a in a3_list}

            if cutoff == 0:
                # V1従来
                sel_umabans = list(mc.head(total)['umaban'].astype(int))
            else:
                pop = mc[mc['ninki'] <= cutoff].head(n_pop)
                disc = mc[mc['ninki'] > cutoff].head(total - n_pop)
                sel = pd.concat([pop, disc])
                sel_umabans = list(sel['umaban'].astype(int))

            payouts = estimate_payouts(sel_umabans, a3_list, odds_map, len(rd))
            overlap = len(set(sel_umabans) & a3_set)

            for btype, p in payouts.items():
                total_by_type[btype]['cost'] += p['cost']
                total_by_type[btype]['payout'] += p['payout']
                if p['payout'] > 0:
                    total_by_type[btype]['hits'] += 1

            year_details.append({
                'year': year, 'sel': sel_umabans, 'overlap': overlap,
                'payouts': payouts,
            })

        # 表示
        print(f'\n  【{label}】 ({total}頭)')
        n_pts = {
            '単勝': total,
            '複勝': total,
            'ワイド': len(list(combinations(range(total), 2))),
            '馬連': len(list(combinations(range(total), 2))),
            '3連複': len(list(combinations(range(total), 3))),
        }
        print(f'    {"券種":>6s} {"点数/R":>6s} {"10R投資":>10s} {"10R配当":>10s} {"的中":>5s} {"ROI":>8s}')
        print(f'    {"-"*55}')
        for btype in ['単勝', '複勝', 'ワイド', '馬連', '3連複']:
            t = total_by_type[btype]
            roi = t['payout'] / t['cost'] * 100 if t['cost'] > 0 else 0
            marker = ' ★' if roi >= 100 else ''
            print(f'    {btype:>6s} {n_pts[btype]:>5d}点 {t["cost"]:>9,}円 {t["payout"]:>9,.0f}円 {t["hits"]:>4d}/10 {roi:>7.1f}%{marker}')

        # 年別詳細（3連複）
        print(f'    --- 3連複 年別 ---')
        for yd in year_details:
            p = yd['payouts']['3連複']
            hit = '○' if p['payout'] > 0 else '×'
            print(f'      {yd["year"]}: {hit} 占有{yd["overlap"]}/3 投資{p["cost"]:,}円 配当{p["payout"]:,.0f}円 選抜{yd["sel"]}')


if __name__ == '__main__':
    run()
