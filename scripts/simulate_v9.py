"""
桜花賞予測モデル v9
v8からの改善:
- 上がり3F偏差値(ema_agari_zscore): 末脚の切れを評価
- 直線長コース適性(long_stretch_zscore): 阪神外回り適性
- 前走距離差(prev_distance_diff): 距離延長/短縮の影響
- MCのハイペース時逃げ先行ペナルティ強化
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

# 直線が長いコース（場コード）
# 05=東京, 09=阪神(外回り), 08=京都(外回り), 04=新潟
LONG_STRETCH_PLACES = {'05', '04'}  # 東京、新潟は確実に直線長い
# 阪神・京都は内外回りで異なるが、track_cd_rawで判別
# 1800=阪神外回り/京都外回り → 直線長い
# 2400=阪神内回り/京都内回り → 直線短い


FEATURES_V9 = [
    'wakuban', 'futan', 'bataijyu', 'zogen_sa', 'heads',
    'past_count', 'ema_time_zscore', 'ema_finish',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'avg_jyuni_3c', 'avg_jyuni_4c',
    'prev_unlucky',
    'prev_race_class', 'log_prize_money',
    'weighted_ema_finish', 'weighted_ema_time',
    # v9新規
    'ema_agari_zscore',
    'long_stretch_zscore',
    'prev_distance_diff',
]
CAT_FEATURES = ['kisyu_code', 'chokyosi_code', 'banusi_code', 'sire_code']

CLASS_MAP = {'A': 5, 'B': 4, 'C': 4, 'L': 3, 'E': 3}


def compute_race_class(grade_cd):
    if pd.isna(grade_cd) or grade_cd == '':
        return 1
    return CLASS_MAP.get(grade_cd, 1)


def is_long_stretch(place_code, track_cd_raw):
    """直線が長いコースかどうか"""
    if place_code in LONG_STRETCH_PLACES:
        return True
    # 阪神(09)/京都(08)の外回り = track_cd_raw が 1800系
    if place_code in {'09', '08'}:
        raw = str(track_cd_raw).strip()
        if raw.startswith('18'):
            return True
    return False


def add_v9_features(race_feat, df_all, um_data):
    """v9の全追加特徴量を計算"""
    feat = race_feat.copy()

    # v8の特徴量
    feat['prev_race_class'] = 1
    feat['log_prize_money'] = 0.0
    feat['weighted_ema_finish'] = feat.get('ema_finish', pd.Series(10, index=feat.index))
    feat['weighted_ema_time'] = feat.get('ema_time_zscore', pd.Series(0, index=feat.index))
    feat['prev_unlucky'] = 0
    feat['sire_code'] = 'unknown'

    # v9の特徴量
    feat['ema_agari_zscore'] = 0.0
    feat['long_stretch_zscore'] = 0.0
    feat['prev_distance_diff'] = 0.0

    for idx in feat.index:
        ketto = feat.loc[idx, 'ketto_num']
        race_date = feat.loc[idx, 'date']
        if pd.isna(race_date):
            continue

        history = df_all[
            (df_all['ketto_num'] == ketto) &
            (df_all['date'] < race_date) &
            (df_all['kakutei_jyuni'] > 0)
        ].sort_values('date')

        if len(history) == 0:
            continue

        last = history.iloc[-1]
        alpha = 0.3
        recent = history.tail(5)
        n_recent = len(recent)

        # --- v8: prev_race_class ---
        feat.loc[idx, 'prev_race_class'] = compute_race_class(last.get('grade_cd', ''))

        # --- v8: prev_unlucky ---
        last_finish = last.get('kakutei_jyuni', 99)
        if 4 <= last_finish <= 9:
            feat.loc[idx, 'prev_unlucky'] = 1

        # --- v8: log_prize_money ---
        feat.loc[idx, 'log_prize_money'] = np.log1p(history['honsyokin'].fillna(0).sum())

        # --- v8: weighted_ema ---
        if n_recent > 0:
            finishes = recent['kakutei_jyuni'].values.astype(float)
            grades = recent.get('grade_cd', pd.Series('', index=recent.index))
            class_w = np.array([compute_race_class(g) for g in grades])
            time_w = np.array([(1 - alpha) ** j for j in range(n_recent - 1, -1, -1)])
            comb_w = time_w * class_w
            valid = ~np.isnan(finishes)
            if valid.sum() > 0:
                feat.loc[idx, 'weighted_ema_finish'] = np.average(finishes[valid], weights=comb_w[valid])

        # --- v8: sire_code ---
        if ketto in um_data.index:
            sire_id = um_data.loc[ketto].get('sire_id', 'unknown')
            feat.loc[idx, 'sire_code'] = str(sire_id) if pd.notna(sire_id) else 'unknown'

        # --- v9: ema_agari_zscore (上がり3F偏差) ---
        # haron_time_l3が0の場合(バイナリ未取得)、timeのレース内偏差で代替
        # time_zscoreを「末脚指標」として流用
        # 通過順位が後方で着順が良い = 末脚が切れる
        if n_recent > 0:
            agari_scores = []
            for _, r in recent.iterrows():
                j3 = r.get('jyuni_3c', 0)
                j4 = r.get('jyuni_4c', 0)
                finish = r.get('kakutei_jyuni', 10)
                if pd.notna(j4) and j4 > 0 and pd.notna(finish) and finish > 0:
                    # 末脚指標: 4角順位 - 着順 (正=追い込んだ)
                    agari_scores.append(j4 - finish)
                else:
                    agari_scores.append(0)

            agari_arr = np.array(agari_scores, dtype=float)
            time_w_a = np.array([(1 - alpha) ** j for j in range(n_recent - 1, -1, -1)])
            feat.loc[idx, 'ema_agari_zscore'] = np.average(agari_arr, weights=time_w_a)

        # --- v9: long_stretch_zscore (直線長コース適性) ---
        if n_recent > 0:
            ls_finishes = []
            for _, r in recent.iterrows():
                pc = r.get('place_code', '')
                tcr = r.get('track_cd_raw', '')
                if is_long_stretch(pc, tcr):
                    f = r.get('kakutei_jyuni', np.nan)
                    if pd.notna(f) and f > 0:
                        ls_finishes.append(f)

            if len(ls_finishes) > 0:
                feat.loc[idx, 'long_stretch_zscore'] = -np.mean(ls_finishes)  # 小さい着順=良い→負にして大きい=良い
            else:
                feat.loc[idx, 'long_stretch_zscore'] = 0.0  # データなし

        # --- v9: prev_distance_diff ---
        prev_kyori = last.get('kyori', 1600)
        if pd.notna(prev_kyori):
            feat.loc[idx, 'prev_distance_diff'] = 1600 - prev_kyori  # 桜花賞1600mとの差
        else:
            feat.loc[idx, 'prev_distance_diff'] = 0

    return feat


# ============================================================
# NNモデル（v8と同じ）
# ============================================================
class HorseRaceModel(nn.Module):
    def __init__(self, num_features, embedding_dims=None):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        if embedding_dims:
            for name, (vocab_size, embed_dim) in embedding_dims.items():
                self.embeddings[name] = nn.Embedding(vocab_size, embed_dim)
                total_embed_dim += embed_dim
        input_dim = num_features + total_embed_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.mu_head = nn.Linear(32, 1)
        self.sigma_head = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x, cats=None):
        parts = [x]
        if cats:
            for n, i in cats.items():
                if n in self.embeddings:
                    parts.append(self.embeddings[n](i))
        h = self.network(torch.cat(parts, dim=1))
        return self.mu_head(h), self.sigma_head(h) + 0.1


class RacePredictor:
    def __init__(self, nf, cf=None):
        self.nf, self.cf = nf, cf or []
        self.scaler = StandardScaler()
        self.les = {}
        self.model = None
        self.dev = torch.device('cpu')
        self._med = None

    def _prep(self, df, fit=False):
        X = df[self.nf].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        if fit:
            self._med = X.median()
            X = X.fillna(self._med)
            Xs = self.scaler.fit_transform(X)
        else:
            X = X.fillna(self._med)
            Xs = self.scaler.transform(X)
        Xt = torch.FloatTensor(Xs).to(self.dev)
        ct = {}
        for c in self.cf:
            if c in df.columns:
                v = df[c].fillna('unknown').astype(str)
                if fit:
                    le = LabelEncoder(); le.fit(v); self.les[c] = le
                else:
                    le = self.les[c]
                    v = v.map(lambda x: x if x in le.classes_ else le.classes_[0])
                ct[c] = torch.LongTensor(le.transform(v)).to(self.dev)
        return Xt, ct

    def train(self, df, epochs=50, lr=0.003, bs=256, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float)
        h = df['heads'].values.astype(float); h[h == 0] = 16
        yt = torch.FloatTensor((y - 1) / (h - 1)).unsqueeze(1).to(self.dev)
        Xt, ct = self._prep(df, fit=True)
        ed = {}
        for c in self.cf:
            if c in self.les:
                vs = len(self.les[c].classes_)
                ed[c] = (vs, min(50, max(4, (vs + 1) // 2)))
        self.model = HorseRaceModel(len(self.nf), ed).to(self.dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        n = len(Xt)
        for ep in range(epochs):
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                bi = idx[i:i + bs]
                mu, sig = self.model(Xt[bi], {c: t[bi] for c, t in ct.items()})
                loss = torch.mean(0.5 * torch.log(sig**2) + 0.5 * ((yt[bi] - mu) / sig)**2)
                opt.zero_grad(); loss.backward(); opt.step()

    def predict(self, df):
        self.model.eval()
        Xt, ct = self._prep(df, fit=False)
        with torch.no_grad():
            mu, sig = self.model(Xt, ct)
        return pd.DataFrame({
            'mu': mu.squeeze().numpy(), 'sigma': sig.squeeze().numpy(),
            'horse_name': df['horse_name'].values, 'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })


# ============================================================
# QMC v9: ハイペース時の逃げ先行ペナルティ強化
# ============================================================
def qmc_simulation(predictions, race_features=None, n_simulations=100000):
    n_horses = len(predictions)
    mu = predictions['mu'].values.copy()
    sigma = predictions['sigma'].values.copy()
    run_styles = np.full(n_horses, 2.5)
    wakubans = np.arange(1, n_horses + 1)
    if race_features is not None and len(race_features) == n_horses:
        if 'avg_run_style' in race_features.columns:
            rs = race_features['avg_run_style'].values
            run_styles = np.where(np.isnan(rs), 2.5, rs)
        if 'wakuban' in race_features.columns:
            wakubans = race_features['wakuban'].values.astype(float)

    n_nige = np.sum(run_styles <= 1.5)
    pace_base = (n_nige - 1.5) * 0.3

    n_dims = n_horses * 4 + 1
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=42)
    n_pow2 = 2 ** int(np.ceil(np.log2(n_simulations)))
    sobol = sampler.random(n_pow2)[:n_simulations]
    sobol_norm = norm.ppf(np.clip(sobol, 0.001, 0.999))

    i = 0
    ability = mu[np.newaxis, :] + sigma[np.newaxis, :] * sobol_norm[:, i:i + n_horses]; i += n_horses
    pace = pace_base + 0.15 * sobol_norm[:, i]; i += 1
    late_u = sobol[:, i:i + n_horses]; i += n_horses
    blocked_u = sobol[:, i:i + n_horses]; i += n_horses
    pos_noise = sobol_norm[:, i:i + n_horses]

    pace_2d = pace[:, np.newaxis]
    sashi = (run_styles >= 3.0).astype(float)
    nige = (run_styles <= 1.5).astype(float)
    senkou = ((run_styles > 1.5) & (run_styles <= 2.5)).astype(float)

    # v9: ハイペース時の補正を強化
    # 差し追込: ハイペースで有利（v8: 0.03 → v9: 0.04）
    ability -= pace_2d * 0.04 * sashi[np.newaxis, :]
    # 逃げ: ハイペースで大幅不利（v8: 0.04 → v9: 0.06）
    ability += pace_2d * 0.06 * nige[np.newaxis, :]
    # 先行: ハイペースでやや不利（v8: なし → v9: 0.02）
    ability += pace_2d * 0.02 * senkou[np.newaxis, :]

    # 枠順
    ability -= 0.01 * ((wakubans <= 3) & (run_styles <= 2.5)).astype(float)[np.newaxis, :]
    ability += 0.01 * ((wakubans >= 6) & (run_styles >= 3.5)).astype(float)[np.newaxis, :]

    # 出遅れ・進路塞がり
    ability += (late_u < 0.05).astype(float) * 0.15
    inner_sashi = ((wakubans <= 3) & (run_styles >= 3.0)).astype(float)
    ability += (blocked_u < 0.10).astype(float) * inner_sashi[np.newaxis, :] * 0.10
    ability += 0.02 * pos_noise

    rankings = np.argsort(np.argsort(ability, axis=1), axis=1) + 1
    results = predictions[['horse_name', 'umaban', 'odds']].copy()
    for pos in range(1, min(n_horses + 1, 19)):
        results[f'prob_{pos}'] = (rankings == pos).mean(axis=0)
    results['expected_rank'] = rankings.mean(axis=0)
    results['win_prob'] = results['prob_1']
    results['top3_prob'] = results[['prob_1', 'prob_2', 'prob_3']].sum(axis=1)
    if 'odds' in results.columns:
        results['ev_win'] = results['win_prob'] * results['odds']
    return results.sort_values('expected_rank')


# ============================================================
# バックテスト
# ============================================================
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

    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    def add_train_features(t, um_ref):
        t = t.copy()
        t['prev_race_class'] = 2
        t['log_prize_money'] = np.log1p(t.get('honsyokin', pd.Series(0, index=t.index)).fillna(0))
        t['weighted_ema_finish'] = t.get('ema_finish', pd.Series(5, index=t.index))
        t['weighted_ema_time'] = t.get('ema_time_zscore', pd.Series(0, index=t.index))
        t['prev_unlucky'] = 0
        t['ema_agari_zscore'] = 0.0
        t['long_stretch_zscore'] = 0.0
        t['prev_distance_diff'] = 0.0
        sire_map = um_ref['sire_id'].to_dict()
        t['sire_code'] = t['ketto_num'].map(sire_map).fillna('unknown').astype(str)
        return t

    results = []
    for year in range(2016, 2026):
        if year not in sakura_ids:
            continue
        rid = sakura_ids[year]
        race_data = df_h[df_h['race_id'] == rid]
        race_date = race_data['date'].iloc[0]
        race_feat = feat[feat['race_id'] == rid].copy()
        if len(race_feat) == 0:
            continue

        race_feat = add_v9_features(race_feat, df_all, um)

        train = feat[
            (feat['date'] < race_date) & (feat['is_turf'] == 1) &
            (feat['past_count'] > 0) & (feat['finish'] > 0)
        ].sort_values('date').tail(20000).copy()
        train = add_train_features(train, um)

        use_f = [f for f in FEATURES_V9 if f in train.columns and f in race_feat.columns]

        pred = RacePredictor(use_f, CAT_FEATURES)
        pred.train(train, epochs=50, lr=0.003, seed=42)
        preds = pred.predict(race_feat)
        mc = qmc_simulation(preds, race_features=race_feat, n_simulations=100000)

        actual = race_data.sort_values('kakutei_jyuni').head(3)
        actual_top3 = set(actual['umaban'].astype(int))
        actual_1st = int(actual.iloc[0]['umaban'])
        top5 = mc.head(5)
        top5_umas = set(top5['umaban'].astype(int))
        overlap = len(top5_umas & actual_top3)
        win = int(top5.iloc[0]['umaban']) == actual_1st

        top5_finishes = []
        print(f'\n{year}年桜花賞:')
        print(f'  予測TOP5:')
        for rank, (_, r) in enumerate(top5.iterrows(), 1):
            uma = int(r['umaban'])
            act = race_data[race_data['umaban'] == uma]
            af = int(act.iloc[0]['kakutei_jyuni']) if len(act) > 0 else 18
            top5_finishes.append(af)
            hit = '*' if uma in actual_top3 else ' '
            print(f'    {hit}{rank}位 [{uma:2d}] {r["horse_name"]:16s} '
                  f'勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} '
                  f'odds={r["odds"]:.1f} (実際{af}着)')
        print(f'  実際:')
        for _, r in actual.iterrows():
            print(f'    {int(r["kakutei_jyuni"])}着 [{int(r["umaban"]):2d}] '
                  f'{r["horse_name"]:16s} ({int(r["ninki"])}人気 odds:{r["odds"]})')

        avg_f = np.mean(top5_finishes)
        ana = any(r['odds'] >= 10 and int(r['umaban']) in actual_top3
                  for _, r in top5.iterrows() if pd.notna(r.get('odds')))
        results.append({
            'year': year, 'win': win, 'overlap': overlap,
            't5': actual_1st in top5_umas, 'avg': avg_f,
            'ana': ana, 'p1f': top5_finishes[0],
        })
        sys.stdout.flush()

    n = len(results)
    print(f'\n{"="*60}')
    print(f'  v9 KPIレポート')
    print(f'{"="*60}')
    w = sum(r['win'] for r in results); wr = w/n*100
    t5ok = sum(1 for r in results if r['avg'] <= 9)
    ana_n = sum(r['ana'] for r in results)
    ov2 = sum(1 for r in results if r['overlap'] >= 2); ovr = ov2/n*100

    print(f'\n  ■ KPI 1: 1着的中率: {w}/{n} ({wr:.0f}%) → {"✅" if 30<=wr<=40 else "❌"}')
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5ok}/{n} → {"✅" if t5ok==n else "❌"}')
    print(f'  ■ KPI 3: 穴馬検知: {ana_n}/{n}年 → {"✅" if ana_n>=1 else "❌"}')
    print(f'  ■ KPI 4: 馬券内占有率: {ov2}/{n} ({ovr:.0f}%) → {"✅" if ovr>=80 else "❌"}')

    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        a = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["p1f"]}着 3着内{r["overlap"]}/3 TOP5平均{r["avg"]:.1f}着 {a}')

if __name__ == '__main__':
    run()
