"""
桜花賞予測モデル v9b
v9の修正版: 学習データにもv9特徴量を正しく計算する
- 上がり3F代替(末脚指標): 4角順位-着順のEMA
- 直線長コース適性: 東京/阪神外/京都外/新潟での着順
- 前走距離差: 1600mとの差
- 全て事前計算済みv4データに一括追加
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

# Grade + Class combined mapping
# grade_cd: A=G1, B=G2, C=G3, L=Listed, E=OP
# class_cd 1桁目: 0=障害等, 1=2歳, 2=3歳, 3=古馬(4歳以上), 4=3歳以上
# class_cd 2桁目: 1=OP/重賞, 2=3勝, 3=未勝利/新馬, 4=1勝~2勝
# ※ grade_cdが空の場合にclass_cdで補完する
# 8段階スコア: クラス情報を最大限反映
# NNが高クラス(7,8)をレアとして扱う問題はLayer3ディベートでカバーする
GRADE_CLASS_SCORE = {
    'A': 8,   # G1
    'B': 7,   # G2
    'C': 6,   # G3
    'L': 5,   # Listed
    'E': 4,   # OP特別
}
CLASS_CD_SCORE = {
    '01': 4, '11': 4, '21': 4, '31': 4, '41': 4,  # OP/重賞級
    '02': 3, '12': 3, '22': 3, '32': 3, '42': 3,  # 3勝クラス
    '04': 2, '14': 2, '24': 2, '34': 2, '44': 2,  # 2勝クラス
    '03': 1, '13': 1, '23': 1, '33': 1, '43': 1,  # 1勝/未勝利/新馬
}
LONG_STRETCH = {'05', '04'}  # 東京、新潟


def add_v9_to_all(feat, df_all, um_data):
    """
    v4特徴量DataFrameにv9特徴量を一括追加する
    馬ごとのgroupby処理で高速化
    """
    print('  Adding v9 features to all data...')
    feat = feat.copy()

    # 初期化
    feat['prev_race_class'] = 1.0
    feat['log_prize_money'] = 0.0
    feat['weighted_ema_finish'] = feat['ema_finish'].copy()
    feat['ema_agari'] = 0.0
    feat['long_stretch_avg'] = np.nan
    feat['prev_dist_diff'] = 0.0
    feat['sire_code'] = 'unknown'

    # 種牡馬マッピング（一括）
    sire_map = um_data['sire_id'].to_dict()
    feat['sire_code'] = feat['ketto_num'].map(sire_map).fillna('unknown').astype(str)

    # df_allから必要な情報を馬ごとに前処理
    # 前走情報: 馬ごとにソートして前走を取得
    df_sorted = df_all.sort_values(['ketto_num', 'date']).reset_index(drop=True)

    # 前走のgrade_cd, kyori, 賞金を馬ごとに計算
    print('  Computing per-horse prev-race features...')

    # 馬ごとの前走情報をshift()で一括計算
    df_sorted['prev_grade'] = df_sorted.groupby('ketto_num')['grade_cd'].shift(1)
    df_sorted['prev_class_cd'] = df_sorted.groupby('ketto_num')['class_cd'].shift(1)
    df_sorted['prev_kyori'] = df_sorted.groupby('ketto_num')['kyori'].shift(1)
    df_sorted['cum_prize'] = df_sorted.groupby('ketto_num')['honsyokin'].transform(
        lambda x: x.fillna(0).cumsum().shift(1).fillna(0))

    # 末脚指標: 4角順位-着順（大きいほど末脚が切れる）
    df_sorted['agari_raw'] = df_sorted['jyuni_4c'].fillna(0) - df_sorted['kakutei_jyuni'].fillna(0)
    df_sorted.loc[df_sorted['jyuni_4c'] == 0, 'agari_raw'] = 0

    # 直線長コースフラグ
    df_sorted['is_long_stretch'] = 0
    ls_mask = df_sorted['place_code'].isin(LONG_STRETCH)
    # 阪神/京都の外回り(track_cd_raw startswith '18')
    if 'track_cd_raw' in df_sorted.columns:
        outer_mask = (df_sorted['place_code'].isin({'09', '08'})) & (
            df_sorted['track_cd_raw'].astype(str).str.startswith('18'))
        ls_mask = ls_mask | outer_mask
    df_sorted['is_long_stretch'] = ls_mask.astype(int)

    # race_idベースでマッピング
    # feat.race_id + feat.ketto_num → df_sortedの対応行を特定
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

    # log_prize_money
    cum_prize_map = df_sorted.set_index('_key')['cum_prize'].to_dict()
    feat['log_prize_money'] = feat['_key'].map(cum_prize_map).fillna(0).apply(np.log1p)

    # prev_dist_diff
    prev_kyori_map = df_sorted.set_index('_key')['prev_kyori'].to_dict()
    # prev_dist_diff: 1600固定に戻す（C修正はバックテストで干渉するため保留）
    feat['prev_dist_diff'] = feat['_key'].map(prev_kyori_map).fillna(1600)
    feat['prev_dist_diff'] = 1600 - feat['prev_dist_diff']

    # weighted_ema_finish: grade重み付きEMA
    # 簡易版: prev_race_class × ema_finish の相互作用で代替
    feat['weighted_ema_finish'] = feat['ema_finish'] / (feat['prev_race_class'].clip(1) / 3.0)

    # ema_agari: 馬ごとの末脚EMA
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

        # df_sortedからこの馬の末脚データを取得
        horse_all = df_sorted[df_sorted['ketto_num'] == ketto].sort_values('date')
        agari_vals = horse_all['agari_raw'].values
        ls_vals = horse_all['is_long_stretch'].values
        finish_vals = horse_all['kakutei_jyuni'].values.astype(float)
        dates_all = horse_all['date'].values

        # feat内の各行に対して、その日より前のデータでEMAを計算
        for i_feat, feat_idx in enumerate(horse.index):
            feat_date = feat.loc[feat_idx, 'date']
            if pd.isna(feat_date):
                continue

            # この日より前のデータ
            mask_before = dates_all < feat_date
            past_agari = agari_vals[mask_before]
            past_ls = ls_vals[mask_before]
            past_finish = finish_vals[mask_before]

            # 直近n_past走
            past_agari = past_agari[-n_past:]
            past_ls = past_ls[-n_past:]
            past_finish = past_finish[-n_past:]

            if len(past_agari) > 0:
                w = np.array([(1 - alpha) ** j for j in range(len(past_agari) - 1, -1, -1)])
                feat.loc[feat_idx, 'ema_agari'] = np.average(past_agari, weights=w)

            # 直線長コースでの着順平均
            ls_mask_p = past_ls == 1
            if ls_mask_p.sum() > 0:
                feat.loc[feat_idx, 'long_stretch_avg'] = past_finish[ls_mask_p].mean()

        processed += 1
        if processed % 10000 == 0:
            print(f'    {processed}/{total} horses...')

    print(f'  Done: {processed} horses')
    feat['has_long_stretch'] = feat['long_stretch_avg'].notna().astype(int)
    feat.drop(columns=['_key'], inplace=True)
    return feat


# NNモデル（v8と同じ）
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


# QMC v9: 逃げ先行ペナルティ強化
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


def run():
    from src.binary_parser import load_all_data, load_hanshin_data
    from src.um_parser import load_um_data

    print('Loading...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    df_all = load_all_data(years=range(2014, 2026))
    df_all = df_all[df_all['kakutei_jyuni'] > 0]
    um = load_um_data(years=range(2010, 2026))
    feat = pd.read_pickle('data/features_all_v4.pkl')

    # v9特徴量を全データに一括追加（キャッシュ利用）
    cache_path = 'data/features_v9b_cache.pkl'
    if os.path.exists(cache_path):
        print('Loading cached v9b features...')
        feat = pd.read_pickle(cache_path)
    else:
        feat = add_v9_to_all(feat, df_all, um)
        feat.to_pickle(cache_path)
        print(f'Saved cache: {cache_path}')

    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    use_f = [f for f in FEATURES_V9 if f in feat.columns]
    print(f'Features: {use_f}')

    results = []
    for year in range(2016, 2026):
        if year not in sakura_ids: continue
        rid = sakura_ids[year]
        rd = df_h[df_h['race_id'] == rid]; rdate = rd['date'].iloc[0]
        rf = feat[feat['race_id'] == rid].copy()
        if len(rf) == 0: continue

        tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()

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
        print(f'  予測TOP5:')
        for rk, (_, r) in enumerate(t5.iterrows(), 1):
            u = int(r['umaban']); a = rd[rd['umaban']==u]
            af = int(a.iloc[0]['kakutei_jyuni']) if len(a) > 0 else 18; t5f.append(af)
            h = '*' if u in a3 else ' '
            print(f'    {h}{rk}位 [{u:2d}] {r["horse_name"]:16s} 勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} odds={r["odds"]:.1f} (実際{af}着)')
        print(f'  実際:')
        for _, r in act.iterrows():
            print(f'    {int(r["kakutei_jyuni"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]:16s} ({int(r["ninki"])}人気 odds:{r["odds"]})')
        avg = np.mean(t5f)
        ana = any(r['odds']>=10 and int(r['umaban']) in a3 for _, r in t5.iterrows() if pd.notna(r.get('odds')))
        results.append({'year':year,'win':w,'overlap':ov,'t5':a1 in t5u,'avg':avg,'ana':ana,'p1f':t5f[0]})
        sys.stdout.flush()

    n = len(results)
    print(f'\n{"="*60}')
    print(f'  v9b KPIレポート')
    print(f'{"="*60}')
    wn = sum(r['win'] for r in results); wr = wn/n*100
    t5ok = sum(1 for r in results if r['avg']<=9)
    an = sum(r['ana'] for r in results)
    o2 = sum(1 for r in results if r['overlap']>=2); ovr = o2/n*100
    print(f'\n  ■ KPI 1: 1着的中率: {wn}/{n} ({wr:.0f}%) → {"✅" if 30<=wr<=40 else "❌"}')
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5ok}/{n} → {"✅" if t5ok==n else "❌"}')
    print(f'  ■ KPI 3: 穴馬検知: {an}/{n}年 → {"✅" if an>=1 else "❌"}')
    print(f'  ■ KPI 4: 馬券内占有率: {o2}/{n} ({ovr:.0f}%) → {"✅" if ovr>=80 else "❌"}')
    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        a = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["p1f"]}着 3着内{r["overlap"]}/3 TOP5平均{r["avg"]:.1f}着 {a}')

if __name__ == '__main__':
    run()
