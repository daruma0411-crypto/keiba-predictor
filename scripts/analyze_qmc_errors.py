"""
QMCエラー分析: 全芝重賞/Lレース10年分で
「QMCが外した馬」「QMCが過大評価した馬」のパターンを大量収集
→ ディベートルールを自動発見
"""
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
            'finish': df['finish'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
            'ninki': df['ninki'].values if 'ninki' in df.columns else np.nan,
        })

def qmc_sim_simple(preds, rf=None, n=100000):
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
    res = preds[['horse_name','umaban','finish']].copy()
    res['expected_rank'] = rk.mean(axis=0)
    res['win_prob'] = (rk==1).mean(axis=0)
    res['top3_prob'] = sum((rk==i).mean(axis=0) for i in range(1,4))
    return res.sort_values('expected_rank')


def run():
    from src.binary_parser import load_all_data

    print('Loading data...')
    df_all = load_all_data(years=range(2015, 2026))
    df_all = df_all[df_all['kakutei_jyuni'] > 0]
    feat = pd.read_pickle('data/features_v9b_cache.pkl')
    use_f = [f for f in FEATURES_V9 if f in feat.columns]

    # 重賞/Lレースを特定: featキャッシュからis_turf=1 & prev_race_class>=5(Listed以上)のレース
    # grade_cdがfeatにない場合、df_allのgrade_cdでrace_idを特定
    if 'grade_cd' in df_all.columns:
        stakes = df_all[df_all['grade_cd'].isin(['A', 'B', 'C', 'L'])]
        stakes_rids_all = set(stakes['race_id'].unique())
    else:
        stakes_rids_all = set()

    # feat内の芝レースで重賞/L該当
    turf_feat = feat[feat['is_turf'] == 1]
    if stakes_rids_all:
        target_rids = [rid for rid in stakes_rids_all if rid in set(turf_feat['race_id'].unique())]
    else:
        # grade_cdがない場合: prev_race_class>=5のレースが多い race_id を重賞と推定
        race_max_class = turf_feat.groupby('race_id')['prev_race_class'].max()
        target_rids = list(race_max_class[race_max_class >= 5].index)

    print(f'Target stakes/L races: {len(target_rids)}')

    # 年別にwalk-forwardで処理
    # NN学習は年ごとに1回（効率化）
    all_overrated = []   # QMC TOP5に入ったが着外(4着以下)
    all_missed = []      # QMC TOP5に入らなかったが3着以内
    all_correct = []     # QMC TOP5に入って3着以内

    years = sorted(set(int(rid[:4]) for rid in target_rids))
    print(f'Years: {years}')

    for year in years:
        if year < 2017:  # 学習データが少なすぎる年をスキップ
            continue

        year_rids = [rid for rid in target_rids if rid.startswith(str(year))]
        if not year_rids:
            continue

        # 年の最初のレース日付でNN学習（年1回）
        first_date = feat[feat['race_id'].isin(year_rids)]['date'].min()
        tr = feat[
            (feat['date'] < first_date) &
            (feat['is_turf'] == 1) &
            (feat['past_count'] > 0) &
            (feat['finish'] > 0)
        ].sort_values('date').tail(20000).copy()

        if len(tr) < 5000:
            print(f'  {year}: skip (train={len(tr)})')
            continue

        print(f'  {year}: {len(year_rids)} races, train={len(tr):,}...', end='', flush=True)

        pred = Predictor(use_f, CAT_FEATURES)
        pred.train(tr, ep=50, lr=0.003, seed=42)

        race_count = 0
        for rid in year_rids:
            rf = feat[feat['race_id'] == rid].copy()
            if len(rf) < 8:  # 少頭数スキップ
                continue

            if 'ninki' not in rf.columns and 'odds' in rf.columns:
                rf['ninki'] = rf['odds'].rank(method='first')

            ps = pred.predict(rf)
            mc = qmc_sim_simple(ps, rf, n=50000)

            # 実際のTOP3
            actual_top3 = set(rf.sort_values('finish').head(3)['umaban'].astype(int))

            # QMC TOP5
            qmc_top5 = mc.head(5)
            qmc_top5_uma = set(qmc_top5['umaban'].astype(int))

            # 全馬の特徴量を収集
            for _, r in mc.iterrows():
                u = int(r['umaban'])
                fd = rf[rf['umaban'] == u]
                if len(fd) == 0:
                    continue
                fd = fd.iloc[0]
                qmc_rank = mc.index.get_loc(r.name) + 1 if r.name in mc.index else 99
                # mc is sorted by expected_rank, so enumerate
                actual_finish = int(fd['finish'])
                in_qmc_top5 = u in qmc_top5_uma
                in_actual_top3 = u in actual_top3

                row = {
                    'race_id': rid,
                    'year': year,
                    'umaban': u,
                    'horse_name': r['horse_name'],
                    'actual_finish': actual_finish,
                    'qmc_rank': list(mc['umaban'].astype(int)).index(u) + 1,
                    'in_qmc_top5': in_qmc_top5,
                    'in_actual_top3': in_actual_top3,
                    'mu': float(ps[ps['umaban']==u]['mu'].values[0]) if len(ps[ps['umaban']==u]) > 0 else np.nan,
                    'sigma': float(ps[ps['umaban']==u]['sigma'].values[0]) if len(ps[ps['umaban']==u]) > 0 else np.nan,
                    'expected_rank': float(r['expected_rank']),
                    'win_prob': float(r['win_prob']),
                    'top3_prob': float(r['top3_prob']),
                    # 特徴量
                    'ema_finish': fd.get('ema_finish', np.nan),
                    'ema_time_zscore': fd.get('ema_time_zscore', np.nan),
                    'ema_time_diff': fd.get('ema_time_diff', np.nan),
                    'prev_race_class': fd.get('prev_race_class', np.nan),
                    'log_prize_money': fd.get('log_prize_money', np.nan),
                    'avg_run_style': fd.get('avg_run_style', np.nan),
                    'interval_days': fd.get('interval_days', np.nan),
                    'same_dist_finish': fd.get('same_dist_finish', np.nan),
                    'same_surface_finish': fd.get('same_surface_finish', np.nan),
                    'prev_dist_diff': fd.get('prev_dist_diff', np.nan),
                    'ema_agari': fd.get('ema_agari', np.nan),
                    'long_stretch_avg': fd.get('long_stretch_avg', np.nan),
                    'weighted_ema_finish': fd.get('weighted_ema_finish', np.nan),
                    'odds': fd.get('odds', np.nan),
                    'ninki': fd.get('ninki', np.nan),
                    'win_rate': fd.get('win_rate', np.nan),
                    'top3_rate': fd.get('top3_rate', np.nan),
                    'jockey_win_rate': fd.get('jockey_win_rate', np.nan),
                    'trainer_win_rate': fd.get('trainer_win_rate', np.nan),
                    'bataijyu': fd.get('bataijyu', np.nan),
                    'zogen_sa': fd.get('zogen_sa', np.nan),
                    'past_count': fd.get('past_count', np.nan),
                    'heads': fd.get('heads', np.nan),
                }

                if in_qmc_top5 and not in_actual_top3:
                    all_overrated.append(row)
                elif not in_qmc_top5 and in_actual_top3:
                    all_missed.append(row)
                elif in_qmc_top5 and in_actual_top3:
                    all_correct.append(row)

            race_count += 1

        print(f' done ({race_count} processed)')

    # DataFrameに変換
    df_over = pd.DataFrame(all_overrated)
    df_miss = pd.DataFrame(all_missed)
    df_correct = pd.DataFrame(all_correct)

    print(f'\n{"="*70}')
    print(f'  QMCエラー分析結果')
    print(f'{"="*70}')
    print(f'  過大評価(TOP5入り→着外): {len(df_over):,}件')
    print(f'  見逃し(TOP5外→3着内):   {len(df_miss):,}件')
    print(f'  正解(TOP5入り→3着内):    {len(df_correct):,}件')

    # === パターン分析 ===
    print(f'\n{"="*70}')
    print(f'  過大評価パターン vs 正解パターン')
    print(f'{"="*70}')

    compare_cols = [
        'prev_race_class', 'ema_time_zscore', 'sigma', 'mu',
        'ema_finish', 'interval_days', 'same_dist_finish',
        'prev_dist_diff', 'avg_run_style', 'ema_agari',
        'log_prize_money', 'win_rate', 'top3_rate',
        'jockey_win_rate', 'trainer_win_rate',
        'past_count', 'zogen_sa', 'odds', 'ninki',
    ]

    print(f'\n  {"特徴量":<22s} {"過大評価(着外)":>14s} {"正解(3着内)":>14s} {"差":>10s} {"見逃し":>14s}')
    print(f'  {"-"*76}')
    for col in compare_cols:
        if col in df_over.columns and col in df_correct.columns:
            v_over = df_over[col].median()
            v_correct = df_correct[col].median()
            v_miss = df_miss[col].median() if col in df_miss.columns else np.nan
            diff = v_over - v_correct
            marker = ' ***' if abs(diff) > abs(v_correct * 0.2) and pd.notna(diff) else ''
            print(f'  {col:<22s} {v_over:>14.3f} {v_correct:>14.3f} {diff:>+10.3f} {v_miss:>14.3f}{marker}')

    # === 見逃しパターン vs 正解 ===
    print(f'\n{"="*70}')
    print(f'  見逃し馬の特徴（QMC TOP5外だが3着内）')
    print(f'{"="*70}')

    if len(df_miss) > 0:
        print(f'\n  人気分布:')
        if 'ninki' in df_miss.columns:
            bins = [(1,3), (4,6), (7,10), (11,99)]
            labels = ['1-3人気', '4-6人気', '7-10人気', '11人気~']
            for (lo, hi), lab in zip(bins, labels):
                cnt = ((df_miss['ninki'] >= lo) & (df_miss['ninki'] <= hi)).sum()
                pct = cnt / len(df_miss) * 100
                print(f'    {lab}: {cnt}件 ({pct:.1f}%)')

        print(f'\n  QMC順位分布:')
        for lo, hi, lab in [(6,8,'6-8位'), (9,12,'9-12位'), (13,99,'13位~')]:
            cnt = ((df_miss['qmc_rank'] >= lo) & (df_miss['qmc_rank'] <= hi)).sum()
            pct = cnt / len(df_miss) * 100
            print(f'    {lab}: {cnt}件 ({pct:.1f}%)')

    # 保存
    df_over.to_pickle('data/qmc_overrated.pkl')
    df_miss.to_pickle('data/qmc_missed.pkl')
    df_correct.to_pickle('data/qmc_correct.pkl')
    print(f'\n  Saved: data/qmc_overrated.pkl, qmc_missed.pkl, qmc_correct.pkl')


if __name__ == '__main__':
    run()
