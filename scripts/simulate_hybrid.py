"""
ハイブリッドバックテスト: V1バイナリ骨格 + CSV上がり3F/ペース
V1(80%)を超えることが目標。
- ベース: features_v9b_cache.pkl（バイナリ由来、4角通過順が正確）
- 追加: CSV由来の ema_agari_3f, ema_agari_rank, ema_agari_zscore,
        ema_pace_front, ema_pace_diff, pace_h_time_diff, pace_s_time_diff
- マージキー: ketto_num + date
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
np.random.seed(42)

# === V1特徴量 + CSV追加特徴量 ===
FEATURES_HYBRID = [
    # V1骨格（26特徴量）
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
    # CSV追加（上がり3F系）
    'ema_agari_3f',       # 上がり3F秒のEMA
    'ema_agari_rank',     # 上がり3F順位のEMA
    'ema_agari_zscore',   # 上がり3Fレース内偏差値のEMA
    # CSV追加（ペース系）
    'ema_pace_front',     # 前半ペースEMA
    'ema_pace_diff',      # 前後半差EMA
    'pace_h_time_diff',   # ハイペース時着差
    'pace_s_time_diff',   # スローペース時着差
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
                if fit: le = LabelEncoder(); le.fit(v); self.les[c] = le
                else:
                    le = self.les[c]; v = v.map(lambda x: x if x in le.classes_ else le.classes_[0])
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
            if c in self.les: vs = len(self.les[c].classes_); ed[c] = (vs, min(50, max(4,(vs+1)//2)))
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
        return pd.DataFrame({'mu':mu.squeeze().numpy(),'sigma':s.squeeze().numpy(),
            'horse_name':df['horse_name'].values,'ketto_num':df['ketto_num'].values,
            'umaban':df['umaban'].values,'odds':df['odds'].values if 'odds' in df.columns else np.nan})


def qmc_sim(preds, rf=None, race_features=None, n=100000):
    """V1と同じQMC"""
    from scipy.stats import qmc as sqmc, norm
    rf = rf or race_features
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
    res = preds[['horse_name','umaban','odds']].copy()
    for pos in range(1, min(nh+1, 19)): res[f'prob_{pos}'] = (rk==pos).mean(axis=0)
    res['expected_rank'] = rk.mean(axis=0); res['win_prob'] = res['prob_1']
    res['top3_prob'] = res[['prob_1','prob_2','prob_3']].sum(axis=1)
    if 'odds' in res.columns: res['ev_win'] = res['win_prob']*res['odds']
    return res.sort_values('expected_rank')


def build_csv_agari_features():
    """CSVから上がり3F/ペース系EMAだけを計算して返す"""
    cache_path = 'data/csv_agari_cache.pkl'
    if os.path.exists(cache_path):
        print('Loading cached CSV agari features...')
        return pd.read_pickle(cache_path)

    print('Building CSV agari features (one-time, ~20min)...')
    from src.csv_parser import load_csv_data
    from src.features_v2 import build_features_v2
    df_csv = load_csv_data('C:/TXT/10ne_deta', surface_filter='芝')
    v2 = build_features_v2(df_csv)

    # 必要なカラムだけ抽出
    keep = ['ketto_num', 'date', 'race_id',
            'ema_agari_3f', 'ema_agari_rank', 'ema_agari_zscore',
            'ema_pace_front', 'ema_pace_diff',
            'pace_h_time_diff', 'pace_s_time_diff']
    out = v2[[c for c in keep if c in v2.columns]].copy()
    out.to_pickle(cache_path)
    print(f'Saved: {cache_path} ({len(out):,} rows)')
    return out


def run():
    from src.binary_parser import load_all_data, load_hanshin_data
    print('='*60)
    print('  ハイブリッドバックテスト: V1骨格 + CSV上がり3F/ペース')
    print('='*60)

    # === 1. V1ベースデータ ===
    print('\n[1] Loading V1 base...')
    feat = pd.read_pickle('data/features_v9b_cache.pkl')
    print(f'  V1: {len(feat):,} rows')

    # === 2. CSV追加特徴量をマージ ===
    print('\n[2] Loading CSV agari/pace features...')
    csv_feat = build_csv_agari_features()
    print(f'  CSV features: {len(csv_feat):,} rows')

    # マージキー: ketto_num + date
    feat['_merge_key'] = feat['ketto_num'].astype(str) + '_' + feat['date'].astype(str)
    csv_feat['_merge_key'] = csv_feat['ketto_num'].astype(str) + '_' + csv_feat['date'].astype(str)

    # CSV特徴量カラム
    csv_cols = ['ema_agari_3f', 'ema_agari_rank', 'ema_agari_zscore',
                'ema_pace_front', 'ema_pace_diff',
                'pace_h_time_diff', 'pace_s_time_diff']
    csv_cols = [c for c in csv_cols if c in csv_feat.columns]

    # 重複除去（同一馬同一日に複数レースの場合最後を採用）
    csv_dedup = csv_feat.drop_duplicates(subset='_merge_key', keep='last')
    merge_map = csv_dedup.set_index('_merge_key')[csv_cols]

    for col in csv_cols:
        feat[col] = feat['_merge_key'].map(merge_map[col])

    matched = feat[csv_cols[0]].notna().sum()
    print(f'  Matched: {matched:,}/{len(feat):,} ({matched/len(feat)*100:.1f}%)')
    feat.drop(columns=['_merge_key'], inplace=True)

    # === 3. 使用特徴量 ===
    use_f = [f for f in FEATURES_HYBRID if f in feat.columns]
    missing = [f for f in FEATURES_HYBRID if f not in feat.columns]
    print(f'\n[3] Features: {len(use_f)} available, {len(missing)} missing')
    if missing: print(f'  Missing: {missing}')

    # === 4. 桜花賞特定 ===
    print('\n[4] Loading Hanshin data for race lookup...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}
    print(f'  Sakura: {sorted(sakura_ids.keys())}')

    # === 5. Walk-forward バックテスト ===
    results = []
    for year in range(2016, 2026):
        if year not in sakura_ids: continue
        rid = sakura_ids[year]
        rd = df_h[df_h['race_id'] == rid]; rdate = rd['date'].iloc[0]
        rf = feat[feat['race_id'] == rid].copy()
        if len(rf) == 0: continue

        tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)&(feat['ema_agari_3f'].notna())].sort_values('date').tail(20000).copy()

        pred = Predictor(use_f, CAT_FEATURES)
        pred.train(tr, ep=50, lr=0.003, seed=42)
        ps = pred.predict(rf)
        mc = qmc_sim(ps, race_features=rf, n=100000)

        act = rd.sort_values('kakutei_jyuni').head(3)
        a3 = set(act['umaban'].astype(int)); a1 = int(act.iloc[0]['umaban'])
        t5 = mc.head(5); t5u = set(t5['umaban'].astype(int))
        ov = len(t5u & a3); w = int(t5.iloc[0]['umaban']) == a1
        t5f = []
        print(f'\n{year}年桜花賞:')
        print(f'  学習: {len(tr):,} | 出走: {len(rf)} | CSV特徴量マッチ: {rf[csv_cols[0]].notna().sum()}/{len(rf)}')
        for rk, (_, r) in enumerate(t5.iterrows(), 1):
            u = int(r['umaban']); a = rd[rd['umaban']==u]
            af = int(a.iloc[0]['kakutei_jyuni']) if len(a) > 0 else 18; t5f.append(af)
            h = '*' if u in a3 else ' '
            print(f'    {h}{rk}位 [{u:2d}] {r["horse_name"]:16s} 勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} odds={r["odds"]:.1f} (実際{af}着)')
        print(f'  実際TOP3:')
        for _, r in act.iterrows():
            print(f'    {int(r["kakutei_jyuni"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]:16s} ({int(r["ninki"])}人気 odds:{r["odds"]})')

        avg = np.mean(t5f)
        ana = any(r['odds']>=10 and int(r['umaban']) in a3 for _, r in t5.iterrows() if pd.notna(r.get('odds')))
        results.append({'year':year,'win':w,'overlap':ov,'t5':a1 in t5u,'avg':avg,'ana':ana,'p1f':t5f[0]})
        sys.stdout.flush()

    # === 6. KPIレポート ===
    n = len(results)
    print(f'\n{"="*60}')
    print(f'  ハイブリッド KPIレポート')
    print(f'{"="*60}')
    wn = sum(r['win'] for r in results); wr = wn/n*100
    t5ok = sum(1 for r in results if r['avg']<=9)
    an = sum(r['ana'] for r in results)
    o2 = sum(1 for r in results if r['overlap']>=2); ovr = o2/n*100

    print(f'\n  ■ KPI 1: 1着的中率: {wn}/{n} ({wr:.0f}%)')
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5ok}/{n}')
    print(f'  ■ KPI 3: 穴馬検知: {an}/{n}年')
    print(f'  ■ KPI 4: 馬券内占有率: {o2}/{n} ({ovr:.0f}%)')

    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        a = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["p1f"]}着 3着内{r["overlap"]}/3 TOP5平均{r["avg"]:.1f}着 {a}')

    print(f'\n  === 比較 ===')
    print(f'  V1本番(v9b):     占有率80%')
    print(f'  V2(CSV単体):     占有率60%')
    print(f'  ハイブリッド:     占有率{ovr:.0f}%')


if __name__ == '__main__':
    run()
