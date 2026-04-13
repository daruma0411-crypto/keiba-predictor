"""
2026年桜花賞予測 - v9bウォークフォワード方式（最適化版）
1. 2026年データを含む特徴量を構築（v9 EMA計算を高速化）
2. 桜花賞18頭の推論用特徴量構築
3. 2026/4/12より前の芝データで学習 (seed=42, ep=50)
4. QMC 10万回シミュレーション
"""
import sys
import io
import os
import warnings
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import qmc, norm
import re
import time

np.random.seed(42)

# === v9b定義 ===
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
# Grade + Class combined mapping
# grade_cd: A=G1, B=G2, C=G3, L=Listed, E=OP
# class_cd 1桁目: 0=障害等, 1=2歳, 2=3歳, 3=古馬(4歳以上), 4=3歳以上
# class_cd 2桁目: 1=OP/重賞, 2=3勝, 3=未勝利/新馬, 4=1勝~2勝
GRADE_CLASS_SCORE = {
    'A': 8, 'B': 7, 'C': 6, 'L': 5, 'E': 4,
}
CLASS_CD_SCORE = {
    '01': 4, '11': 4, '21': 4, '31': 4, '41': 4,
    '02': 3, '12': 3, '22': 3, '32': 3, '42': 3,
    '04': 2, '14': 2, '24': 2, '34': 2, '44': 2,
    '03': 1, '13': 1, '23': 1, '33': 1, '43': 1,
}
LONG_STRETCH = {'05', '04'}


# === 最適化版 add_v9_to_all ===
def add_v9_to_all_fast(feat, df_all, um_data):
    """v9特徴量を一括追加（事前groupbyで高速化）"""
    t0 = time.time()
    print('  Adding v9 features (optimized)...')
    feat = feat.copy()

    feat['prev_race_class'] = 1.0
    feat['log_prize_money'] = 0.0
    feat['weighted_ema_finish'] = feat['ema_finish'].copy()
    feat['ema_agari'] = 0.0
    feat['long_stretch_avg'] = np.nan
    feat['prev_dist_diff'] = 0.0
    feat['sire_code'] = 'unknown'

    sire_map = um_data['sire_id'].to_dict()
    feat['sire_code'] = feat['ketto_num'].map(sire_map).fillna('unknown').astype(str)

    df_sorted = df_all.sort_values(['ketto_num', 'date']).reset_index(drop=True)

    print('  Computing prev-race features...')
    df_sorted['prev_grade'] = df_sorted.groupby('ketto_num')['grade_cd'].shift(1)
    df_sorted['prev_class_cd'] = df_sorted.groupby('ketto_num')['class_cd'].shift(1)
    df_sorted['prev_kyori'] = df_sorted.groupby('ketto_num')['kyori'].shift(1)
    df_sorted['cum_prize'] = df_sorted.groupby('ketto_num')['honsyokin'].transform(
        lambda x: x.fillna(0).cumsum().shift(1).fillna(0))

    df_sorted['agari_raw'] = df_sorted['jyuni_4c'].fillna(0) - df_sorted['kakutei_jyuni'].fillna(0)
    df_sorted.loc[df_sorted['jyuni_4c'] == 0, 'agari_raw'] = 0

    ls_mask = df_sorted['place_code'].isin(LONG_STRETCH)
    if 'track_cd_raw' in df_sorted.columns:
        outer_mask = (df_sorted['place_code'].isin({'09', '08'})) & (
            df_sorted['track_cd_raw'].astype(str).str.startswith('18'))
        ls_mask = ls_mask | outer_mask
    df_sorted['is_long_stretch'] = ls_mask.astype(int)

    df_sorted['_key'] = df_sorted['race_id'] + '_' + df_sorted['ketto_num']
    feat['_key'] = feat['race_id'] + '_' + feat['ketto_num']

    # prev_race_class: grade_cd優先、なければclass_cdにフォールバック
    def map_race_class(row):
        grade = row.get('prev_grade', '')
        if pd.notna(grade) and grade in GRADE_CLASS_SCORE:
            return GRADE_CLASS_SCORE[grade]
        class_cd = row.get('prev_class_cd', '')
        if pd.notna(class_cd) and str(class_cd).strip() in CLASS_CD_SCORE:
            return CLASS_CD_SCORE[str(class_cd).strip()]
        return 1  # unknown/new horse

    prev_grade_map = df_sorted.set_index('_key')['prev_grade'].to_dict()
    prev_class_cd_map = df_sorted.set_index('_key')['prev_class_cd'].to_dict()
    feat['prev_grade_tmp'] = feat['_key'].map(prev_grade_map)
    feat['prev_class_cd_tmp'] = feat['_key'].map(prev_class_cd_map)
    feat['prev_race_class'] = feat[['prev_grade_tmp', 'prev_class_cd_tmp']].rename(
        columns={'prev_grade_tmp': 'prev_grade', 'prev_class_cd_tmp': 'prev_class_cd'}
    ).apply(map_race_class, axis=1).astype(float)
    feat.drop(columns=['prev_grade_tmp', 'prev_class_cd_tmp'], inplace=True)

    cum_prize_map = df_sorted.set_index('_key')['cum_prize'].to_dict()
    feat['log_prize_money'] = feat['_key'].map(cum_prize_map).fillna(0).apply(np.log1p)

    prev_kyori_map = df_sorted.set_index('_key')['prev_kyori'].to_dict()
    feat['prev_dist_diff'] = feat['_key'].map(prev_kyori_map).fillna(1600)
    feat['prev_dist_diff'] = 1600 - feat['prev_dist_diff']

    feat['weighted_ema_finish'] = feat['ema_finish'] / (feat['prev_race_class'].clip(1) / 3.0)

    # === 高速化ポイント: 事前にgroupbyしてdict化 ===
    print('  Pre-grouping df_sorted by horse...')
    horse_groups = {}
    for ketto, group in df_sorted.groupby('ketto_num'):
        g = group.sort_values('date')
        horse_groups[ketto] = (
            g['agari_raw'].values,
            g['is_long_stretch'].values,
            g['kakutei_jyuni'].values.astype(float),
            g['date'].values,
        )

    print('  Computing agari EMA per horse...')
    alpha = 0.3
    n_past = 5
    processed = 0
    total = feat['ketto_num'].nunique()

    for ketto, group_idx in feat.groupby('ketto_num').groups.items():
        horse = feat.loc[group_idx].sort_values('date')
        n = len(horse)
        if n <= 1:
            processed += 1
            continue

        if ketto not in horse_groups:
            processed += 1
            continue

        agari_vals, ls_vals, finish_vals, dates_all = horse_groups[ketto]

        for i_feat, feat_idx in enumerate(horse.index):
            feat_date = feat.loc[feat_idx, 'date']
            if pd.isna(feat_date):
                continue

            mask_before = dates_all < feat_date
            past_agari = agari_vals[mask_before][-n_past:]
            past_ls = ls_vals[mask_before][-n_past:]
            past_finish = finish_vals[mask_before][-n_past:]

            if len(past_agari) > 0:
                w = np.array([(1 - alpha) ** j for j in range(len(past_agari) - 1, -1, -1)])
                feat.loc[feat_idx, 'ema_agari'] = np.average(past_agari, weights=w)

            ls_mask_p = past_ls == 1
            if ls_mask_p.sum() > 0:
                feat.loc[feat_idx, 'long_stretch_avg'] = past_finish[ls_mask_p].mean()

        processed += 1
        if processed % 10000 == 0:
            elapsed = time.time() - t0
            print(f'    {processed}/{total} horses... ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'  Done: {processed} horses in {elapsed:.0f}s')
    feat['has_long_stretch'] = feat['long_stretch_avg'].notna().astype(int)
    feat.drop(columns=['_key'], inplace=True)
    return feat


# === NNモデル（v9bそのまま） ===
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
        self.sc = StandardScaler(); self.les = {}; self.m = None
        self.d = torch.device('cpu'); self._med = None

    def _p(self, df, fit=False):
        X = df[self.nf].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        if fit:
            self._med = X.median(); X = X.fillna(self._med)
            Xs = self.sc.fit_transform(X)
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
                    v = v.map(lambda x, _le=le: x if x in _le.classes_ else _le.classes_[0])
                ct[c] = torch.LongTensor(le.transform(v)).to(self.d)
        return Xt, ct

    def train(self, df, ep=50, lr=0.003, bs=256, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float)
        h = df['heads'].values.astype(float); h[h == 0] = 16
        yt = torch.FloatTensor((y - 1) / (h - 1)).unsqueeze(1).to(self.d)
        Xt, ct = self._p(df, True)
        ed = {}
        for c in self.cf:
            if c in self.les:
                vs = len(self.les[c].classes_)
                ed[c] = (vs, min(50, max(4, (vs + 1) // 2)))
        self.m = HorseRaceModel(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr)
        self.m.train(); n = len(Xt)
        for e in range(ep):
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                bi = idx[i:i + bs]
                mu, s = self.m(Xt[bi], {c: t[bi] for c, t in ct.items()})
                l = torch.mean(0.5 * torch.log(s ** 2) + 0.5 * ((yt[bi] - mu) / s) ** 2)
                opt.zero_grad(); l.backward(); opt.step()

    def predict(self, df):
        self.m.eval(); Xt, ct = self._p(df, False)
        with torch.no_grad():
            mu, s = self.m(Xt, ct)
        return pd.DataFrame({
            'mu': mu.squeeze().numpy(), 'sigma': s.squeeze().numpy(),
            'horse_name': df['horse_name'].values,
            'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })


# === QMC（v9bそのまま） ===
def qmc_sim(preds, rf=None, n=100000):
    nh = len(preds); mu = preds['mu'].values.copy(); sig = preds['sigma'].values.copy()
    rs = np.full(nh, 2.5); wk = np.arange(1, nh + 1)
    if rf is not None and len(rf) == nh:
        if 'avg_run_style' in rf.columns:
            r = rf['avg_run_style'].values; rs = np.where(np.isnan(r), 2.5, r)
        if 'wakuban' in rf.columns:
            wk = rf['wakuban'].values.astype(float)
    nn_ = np.sum(rs <= 1.5); pb = (nn_ - 1.5) * 0.3
    nd = nh * 4 + 1; s = qmc.Sobol(d=nd, scramble=True, seed=42)
    np2 = 2 ** int(np.ceil(np.log2(n))); sb = s.random(np2)[:n]
    sn = norm.ppf(np.clip(sb, 0.001, 0.999))
    j = 0
    ab = mu[np.newaxis, :] + sig[np.newaxis, :] * sn[:, j:j + nh]; j += nh
    p = pb + 0.15 * sn[:, j]; j += 1
    lu = sb[:, j:j + nh]; j += nh
    bu = sb[:, j:j + nh]; j += nh
    pn = sn[:, j:j + nh]
    p2 = p[:, np.newaxis]
    sa = (rs >= 3.0).astype(float)
    ni = (rs <= 1.5).astype(float)
    se = ((rs > 1.5) & (rs <= 2.5)).astype(float)
    ab -= p2 * 0.04 * sa[np.newaxis, :]
    ab += p2 * 0.06 * ni[np.newaxis, :]
    ab += p2 * 0.02 * se[np.newaxis, :]
    ab -= 0.01 * ((wk <= 3) & (rs <= 2.5)).astype(float)[np.newaxis, :]
    ab += 0.01 * ((wk >= 6) & (rs >= 3.5)).astype(float)[np.newaxis, :]
    ab += (lu < 0.05).astype(float) * 0.15
    is_ = ((wk <= 3) & (rs >= 3.0)).astype(float)
    ab += (bu < 0.10).astype(float) * is_[np.newaxis, :] * 0.10
    ab += 0.02 * pn
    rk = np.argsort(np.argsort(ab, axis=1), axis=1) + 1
    res = preds[['horse_name', 'umaban', 'odds']].copy()
    for pos in range(1, min(nh + 1, 19)):
        res[f'prob_{pos}'] = (rk == pos).mean(axis=0)
    res['expected_rank'] = rk.mean(axis=0)
    res['win_prob'] = res['prob_1']
    res['top3_prob'] = res[['prob_1', 'prob_2', 'prob_3']].sum(axis=1)
    if 'odds' in res.columns:
        res['ev_win'] = res['win_prob'] * res['odds']
    return res.sort_values('expected_rank')


# ==========================================================
# メイン処理
# ==========================================================
def main():
    from src.binary_parser import load_all_data
    from src.features import build_all_features
    from src.um_parser import load_um_data

    t_start = time.time()

    # ==============================
    # Step 1: 特徴量構築（2026年込み）
    # ==============================
    cache_path = 'data/features_v9b_2026.pkl'
    if os.path.exists(cache_path):
        print('[Step1] Loading cached 2026 features...')
        feat = pd.read_pickle(cache_path)
        print(f'  Loaded: {len(feat):,} rows ({feat["date"].min()} ~ {feat["date"].max()})')

        print('  Loading raw data for reference...')
        df_all = load_all_data(years=range(2014, 2027))
        df_all = df_all[df_all['kakutei_jyuni'] > 0]
    else:
        print('[Step1] Building features with 2026 data...')

        print('  Loading all data 2014-2026...')
        df_all = load_all_data(years=range(2014, 2027))
        df_all = df_all[df_all['kakutei_jyuni'] > 0]
        print(f'  Total: {len(df_all):,} records ({df_all["date"].min()} ~ {df_all["date"].max()})')

        print('  Building v4 features...')
        feat = build_all_features(df_all)
        print(f'  v4: {len(feat):,} rows')

        print('  Loading UM data...')
        um = load_um_data(years=range(2010, 2027))
        print(f'  UM: {len(um)} horses')

        print('  Adding v9 features (optimized)...')
        feat = add_v9_to_all_fast(feat, df_all, um)

        feat.to_pickle(cache_path)
        print(f'  Saved: {cache_path} ({len(feat):,} rows)')

    t1 = time.time()
    print(f'  Step1 done in {t1 - t_start:.0f}s')

    # ==============================
    # Step 2: 桜花賞18頭の推論用特徴量
    # ==============================
    print(f'\n[Step2] Building Sakura-sho 2026 race features...')

    # HTML出馬表から枠番・馬番
    with open('C:/TXT/桜花賞５.html', 'r', encoding='cp932') as f:
        html = f.read()

    rows_html = html.split('<TR>')[1:]
    entry_list = []
    for row in rows_html:
        text = re.sub(r'<[^>]+>', '|', row)
        cells = [c.strip() for c in text.split('|') if c.strip()]
        if len(cells) >= 10 and cells[0].isdigit() and cells[1].isdigit():
            entry_list.append({
                'wakuban': int(cells[0]),
                'umaban': int(cells[1]),
                'horse_name_html': cells[2],
            })

    print(f'  HTML entries: {len(entry_list)}')

    # CSVから血統番号マッピング
    csv = pd.read_csv('C:/TXT/桜花賞.csv', encoding='cp932', header=None)
    name_to_ketto = {}
    name_to_latest = {}
    for horse_name in csv[13].str.strip().unique():
        h_data = csv[csv[13].str.strip() == horse_name]
        ketto = '20' + str(h_data.iloc[0][37])
        name_to_ketto[horse_name] = ketto
        h_data = h_data.copy()
        h_data['_date'] = h_data[0].astype(str).str.zfill(2) + h_data[1].astype(str).str.zfill(2) + h_data[2].astype(str).str.zfill(2)
        latest = h_data.sort_values('_date', ascending=False).iloc[0]
        name_to_latest[horse_name] = latest

    # 各馬の最新featレコードを取得し、出馬表情報で上書き
    race_date = pd.Timestamp('2026-04-12')
    sakura_rows = []

    for entry in entry_list:
        hname = entry['horse_name_html']
        ketto = name_to_ketto.get(hname)
        if ketto is None:
            print(f'  WARNING: {hname} not found in CSV')
            continue

        horse_feat = feat[(feat['ketto_num'] == ketto) & (feat['date'] < race_date)]
        if len(horse_feat) > 0:
            latest_row = horse_feat.sort_values('date').iloc[-1].copy()
        else:
            print(f'  WARNING: {hname} ({ketto}) has no feature records')
            continue

        latest_row['wakuban'] = entry['wakuban']
        latest_row['umaban'] = entry['umaban']
        latest_row['heads'] = 18
        latest_row['futan'] = 55.0
        latest_row['horse_name'] = hname

        csv_latest = name_to_latest.get(hname)
        if csv_latest is not None:
            prev_year = int(csv_latest[0])
            prev_month = int(csv_latest[1])
            prev_day = int(csv_latest[2])
            prev_date = pd.Timestamp(2000 + prev_year, prev_month, prev_day)
            latest_row['interval_days'] = (race_date - prev_date).days
            prev_dist = int(csv_latest[11])
            latest_row['prev_dist_diff'] = 1600 - prev_dist

        latest_row['odds'] = np.nan
        latest_row['race_id'] = '20260412_09_02_06_11'
        latest_row['date'] = race_date

        sakura_rows.append(latest_row)

    rf = pd.DataFrame(sakura_rows).reset_index(drop=True)
    print(f'  Sakura-sho 2026: {len(rf)} horses')

    print(f'\n  {"馬番":>4s} {"馬名":18s} {"EMA着順":>8s} {"末脚EMA":>8s} {"直線適性":>8s} {"脚質":>6s} {"前走距離差":>7s} {"間隔日":>6s}')
    print(f'  {"-"*75}')
    for _, r in rf.sort_values('umaban').iterrows():
        ls_str = f'{r["long_stretch_avg"]:8.1f}' if pd.notna(r['long_stretch_avg']) else '     N/A'
        print(f'  {int(r["umaban"]):4d} {r["horse_name"]:18s} '
              f'{r["ema_finish"]:8.2f} {r["ema_agari"]:8.2f} '
              f'{ls_str} {r["avg_run_style"]:6.1f} '
              f'{r["prev_dist_diff"]:7.0f} {r["interval_days"]:6.0f}')

    t2 = time.time()
    print(f'  Step2 done in {t2 - t1:.0f}s')

    # ==============================
    # Step 3: 学習
    # ==============================
    print(f'\n[Step3] Training model (seed=42, ep=50, lr=0.003)...')
    use_f = [f for f in FEATURES_V9 if f in feat.columns]
    print(f'  Features: {len(use_f)}')

    tr = feat[
        (feat['date'] < race_date) &
        (feat['is_turf'] == 1) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date').tail(20000).copy()
    print(f'  Training data: {len(tr):,} samples ({tr["date"].min()} ~ {tr["date"].max()})')

    pred = Predictor(use_f, CAT_FEATURES)
    pred.train(tr, ep=50, lr=0.003, seed=42)
    t3 = time.time()
    print(f'  Training done in {t3 - t2:.0f}s')

    # ==============================
    # Step 4: 推論 + QMC
    # ==============================
    print(f'\n[Step4] Predict + QMC (100,000 simulations)...')
    ps = pred.predict(rf)
    mc = qmc_sim(ps, rf=rf, n=100000)
    t4 = time.time()
    print(f'  QMC done in {t4 - t3:.0f}s')

    # ==============================
    # 結果出力
    # ==============================
    print(f'\n{"="*80}')
    print(f'  2026年 第86回桜花賞 v9b予測結果')
    print(f'  阪神11R 芝1600m外 18頭 2026/4/12')
    print(f'{"="*80}')
    print(f'\n  {"順位":>4s} {"枠":>2s} {"馬番":>4s} {"馬名":18s} {"勝率":>7s} {"複勝率":>7s} {"期待着順":>8s} {"mu":>7s} {"sigma":>7s}')
    print(f'  {"-"*76}')
    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        rf_row = rf[rf['umaban'] == u].iloc[0]
        ps_row = ps[ps['umaban'] == u].iloc[0]
        print(f'  {rank:4d} {int(rf_row["wakuban"]):2d} {u:4d} {r["horse_name"]:18s} '
              f'{r["win_prob"]:7.1%} {r["top3_prob"]:7.1%} {r["expected_rank"]:8.1f} '
              f'{ps_row["mu"]:7.4f} {ps_row["sigma"]:7.4f}')

    # 詳細データ
    print(f'\n  === 詳細特徴量（上位10頭） ===')
    print(f'  {"馬番":>4s} {"馬名":18s} {"脚質":>6s} {"末脚EMA":>8s} {"直線適性":>8s} {"EMA着順":>8s} {"前走距差":>7s} {"間隔日":>6s} {"前走class":>9s}')
    print(f'  {"-"*90}')
    for _, r in mc.head(10).iterrows():
        u = int(r['umaban'])
        rf_row = rf[rf['umaban'] == u].iloc[0]
        ls_str = f'{rf_row["long_stretch_avg"]:8.1f}' if pd.notna(rf_row['long_stretch_avg']) else '     N/A'
        print(f'  {u:4d} {r["horse_name"]:18s} '
              f'{rf_row["avg_run_style"]:6.1f} {rf_row["ema_agari"]:8.2f} '
              f'{ls_str} {rf_row["ema_finish"]:8.2f} '
              f'{rf_row["prev_dist_diff"]:7.0f} {rf_row["interval_days"]:6.0f} '
              f'{rf_row["prev_race_class"]:9.0f}')

    total_time = time.time() - t_start
    print(f'\n  Total time: {total_time:.0f}s')
    print('\nDone.')


if __name__ == '__main__':
    main()
