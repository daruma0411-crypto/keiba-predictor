"""
桜花賞予測モデル v8
- 種牡馬Embedding追加
- 前走レース格(prev_race_class)追加
- 獲得賞金Log(log_prize_money)追加
- 重み付きEMA(weighted_ema_time, weighted_ema_finish)追加
- 主観スコアなし、QMC+展開要素
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


# ============================================================
# 1. 特徴量定義
# ============================================================
FEATURES_V8 = [
    'wakuban', 'futan', 'bataijyu', 'zogen_sa', 'heads',
    'past_count', 'ema_time_zscore', 'ema_finish',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'avg_jyuni_3c', 'avg_jyuni_4c',
    'prev_unlucky',
    # v8新規
    'prev_race_class',
    'log_prize_money',
    'weighted_ema_finish',
    'weighted_ema_time',
]
CAT_FEATURES = ['kisyu_code', 'chokyosi_code', 'banusi_code', 'sire_code']


# ============================================================
# 2. 特徴量計算
# ============================================================
CLASS_MAP = {
    'A': 5,  # G1
    'B': 4,  # G2
    'C': 4,  # G3
    'L': 3,  # Listed
    'E': 3,  # OP
}

def compute_race_class(grade_cd):
    """グレードコードからレース格を数値化"""
    if pd.isna(grade_cd) or grade_cd == '':
        return 1  # 条件戦
    return CLASS_MAP.get(grade_cd, 1)


def add_v8_features(race_feat, df_all, um_data):
    """v8の追加特徴量を計算"""
    feat = race_feat.copy()
    feat['prev_race_class'] = 1
    feat['log_prize_money'] = 0.0
    feat['weighted_ema_finish'] = feat.get('ema_finish', pd.Series(10, index=feat.index))
    feat['weighted_ema_time'] = feat.get('ema_time_zscore', pd.Series(0, index=feat.index))
    feat['prev_unlucky'] = 0
    feat['sire_code'] = 'unknown'

    for idx in feat.index:
        ketto = feat.loc[idx, 'ketto_num']
        race_date = feat.loc[idx, 'date']
        if pd.isna(race_date):
            continue

        # 過去走取得
        history = df_all[
            (df_all['ketto_num'] == ketto) &
            (df_all['date'] < race_date) &
            (df_all['kakutei_jyuni'] > 0)
        ].sort_values('date')

        if len(history) == 0:
            continue

        # --- 前走レース格 ---
        last = history.iloc[-1]
        grade = last.get('grade_cd', '')
        feat.loc[idx, 'prev_race_class'] = compute_race_class(grade)

        # --- 前走不運フラグ ---
        last_finish = last.get('kakutei_jyuni', 99)
        if 4 <= last_finish <= 9:
            feat.loc[idx, 'prev_unlucky'] = 1

        # --- 累積賞金(log) ---
        total_prize = history['honsyokin'].fillna(0).sum()
        feat.loc[idx, 'log_prize_money'] = np.log1p(total_prize)

        # --- 重み付きEMA (レース格で重み付け) ---
        recent = history.tail(5)
        if len(recent) > 0:
            finishes = recent['kakutei_jyuni'].values.astype(float)
            grades = recent.get('grade_cd', pd.Series('', index=recent.index))
            class_weights = np.array([compute_race_class(g) for g in grades])

            # EMAの時間重み
            alpha = 0.3
            time_weights = np.array([(1 - alpha) ** j for j in range(len(recent) - 1, -1, -1)])

            # 合成重み = 時間重み × クラス重み
            combined_weights = time_weights * class_weights
            valid = ~np.isnan(finishes)

            if valid.sum() > 0:
                feat.loc[idx, 'weighted_ema_finish'] = np.average(
                    finishes[valid], weights=combined_weights[valid])

            # タイムZ-score（あれば）
            if 'time_zscore' in recent.columns:
                tz = recent['time_zscore'].values
                valid_t = ~np.isnan(tz)
                if valid_t.sum() > 0:
                    feat.loc[idx, 'weighted_ema_time'] = np.average(
                        tz[valid_t], weights=combined_weights[valid_t])

        # --- 種牡馬コード ---
        if ketto in um_data.index:
            sire_id = um_data.loc[ketto].get('sire_id', 'unknown')
            feat.loc[idx, 'sire_code'] = str(sire_id) if pd.notna(sire_id) else 'unknown'

    return feat


# ============================================================
# 3. NNモデル
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

    def forward(self, numeric_features, categorical_features=None):
        parts = [numeric_features]
        if categorical_features:
            for name, indices in categorical_features.items():
                if name in self.embeddings:
                    parts.append(self.embeddings[name](indices))
        x = torch.cat(parts, dim=1)
        h = self.network(x)
        return self.mu_head(h), self.sigma_head(h) + 0.1


class RacePredictor:
    def __init__(self, numeric_features, categorical_features=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features or []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.device = torch.device('cpu')

    def _prepare_data(self, df, fit=False):
        X_numeric = df[self.numeric_features].copy()
        for col in X_numeric.columns:
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
        if fit:
            medians = X_numeric.median()
            X_numeric = X_numeric.fillna(medians)
            self._medians = medians
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_numeric = X_numeric.fillna(self._medians)
            X_scaled = self.scaler.transform(X_numeric)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        cat_tensors = {}
        for cat in self.categorical_features:
            if cat in df.columns:
                values = df[cat].fillna('unknown').astype(str)
                if fit:
                    le = LabelEncoder()
                    le.fit(values)
                    self.label_encoders[cat] = le
                else:
                    le = self.label_encoders[cat]
                    values = values.map(lambda x: x if x in le.classes_ else le.classes_[0])
                cat_tensors[cat] = torch.LongTensor(le.transform(values)).to(self.device)
        return X_tensor, cat_tensors

    def train(self, df, epochs=50, lr=0.003, batch_size=256, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        df = df[df['finish'] > 0].copy()
        y = df['finish'].values.astype(float)
        heads = df['heads'].values.astype(float)
        heads[heads == 0] = 16
        y_normalized = (y - 1) / (heads - 1)
        y_tensor = torch.FloatTensor(y_normalized).unsqueeze(1).to(self.device)
        X_tensor, cat_tensors = self._prepare_data(df, fit=True)

        embedding_dims = {}
        for cat in self.categorical_features:
            if cat in self.label_encoders:
                vs = len(self.label_encoders[cat].classes_)
                embedding_dims[cat] = (vs, min(50, max(4, (vs + 1) // 2)))
        self.model = HorseRaceModel(len(self.numeric_features), embedding_dims).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        n = len(X_tensor)
        for epoch in range(epochs):
            indices = torch.randperm(n)
            total_loss = 0
            for i in range(0, n, batch_size):
                bi = indices[i:i + batch_size]
                mu, sigma = self.model(X_tensor[bi], {c: t[bi] for c, t in cat_tensors.items()})
                loss = torch.mean(0.5 * torch.log(sigma**2) + 0.5 * ((y_tensor[bi] - mu) / sigma)**2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bi)

    def predict(self, df):
        self.model.eval()
        X_tensor, cat_tensors = self._prepare_data(df, fit=False)
        with torch.no_grad():
            mu, sigma = self.model(X_tensor, cat_tensors)
        return pd.DataFrame({
            'mu': mu.squeeze().numpy(), 'sigma': sigma.squeeze().numpy(),
            'horse_name': df['horse_name'].values, 'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })


# ============================================================
# 4. QMC
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

    idx = 0
    ability = mu[np.newaxis, :] + sigma[np.newaxis, :] * sobol_norm[:, idx:idx + n_horses]; idx += n_horses
    pace = pace_base + 0.15 * sobol_norm[:, idx]; idx += 1
    late_u = sobol[:, idx:idx + n_horses]; idx += n_horses
    blocked_u = sobol[:, idx:idx + n_horses]; idx += n_horses
    pos_noise = sobol_norm[:, idx:idx + n_horses]

    pace_2d = pace[:, np.newaxis]
    sashi = (run_styles >= 3.0).astype(float)
    nige = (run_styles <= 1.5).astype(float)
    ability -= pace_2d * 0.03 * sashi[np.newaxis, :]
    ability += pace_2d * 0.04 * nige[np.newaxis, :]
    ability -= 0.01 * ((wakubans <= 3) & (run_styles <= 2.5)).astype(float)[np.newaxis, :]
    ability += 0.01 * ((wakubans >= 6) & (run_styles >= 3.5)).astype(float)[np.newaxis, :]
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
# 5. バックテスト
# ============================================================
def run_backtest():
    from src.binary_parser import load_all_data, load_hanshin_data
    from src.um_parser import load_um_data

    print('Loading data...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    df_all = load_all_data(years=range(2014, 2026))
    df_all = df_all[df_all['kakutei_jyuni'] > 0]
    um = load_um_data(years=range(2010, 2026))
    feat = pd.read_pickle('data/features_all_v4.pkl')

    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    # 学習データにもv8特徴量を追加する必要がある
    # 高速化のため、学習データには簡易版を適用
    def add_train_v8(train_df, df_all_ref, um_ref):
        """学習データにv8の追加カラムを設定（簡易版）"""
        t = train_df.copy()
        # prev_race_class: grade_cdがあればそれを使う
        if 'grade_cd' in df_all_ref.columns:
            # 学習データの各レースのgrade_cdを取得
            race_grades = df_all_ref.drop_duplicates('race_id')[['race_id', 'grade_cd']].set_index('race_id')
            # 前走のgrade_cdを取得するのは重いので、簡易的に自分のレースの格を使用
            t['prev_race_class'] = 2  # デフォルト
        else:
            t['prev_race_class'] = 2

        t['log_prize_money'] = np.log1p(t.get('honsyokin', pd.Series(0, index=t.index)).fillna(0))
        t['weighted_ema_finish'] = t.get('ema_finish', pd.Series(5, index=t.index))
        t['weighted_ema_time'] = t.get('ema_time_zscore', pd.Series(0, index=t.index))
        t['prev_unlucky'] = 0
        t['sire_code'] = 'unknown'

        # 種牡馬を紐付け
        if 'ketto_num' in t.columns:
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

        # 予測対象にv8特徴量追加
        race_feat = add_v8_features(race_feat, df_all, um)

        # 学習データ
        train = feat[
            (feat['date'] < race_date) & (feat['is_turf'] == 1) &
            (feat['past_count'] > 0) & (feat['finish'] > 0)
        ].sort_values('date').tail(20000).copy()
        train = add_train_v8(train, df_all, um)

        # 使用特徴量
        use_features = [f for f in FEATURES_V8 if f in train.columns and f in race_feat.columns]

        predictor = RacePredictor(use_features, CAT_FEATURES)
        predictor.train(train, epochs=50, lr=0.003, seed=42)
        preds = predictor.predict(race_feat)

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
        ana_hit = any(
            r['odds'] >= 10 and int(r['umaban']) in actual_top3
            for _, r in top5.iterrows() if pd.notna(r.get('odds'))
        )
        results.append({
            'year': year, 'win': win, 'overlap': overlap,
            't5': actual_1st in top5_umas, 'avg_top5': avg_f,
            'ana': ana_hit, 'pred1_finish': top5_finishes[0],
        })
        sys.stdout.flush()

    # KPIレポート
    n = len(results)
    print(f'\n{"="*60}')
    print(f'  v8 KPIレポート')
    print(f'{"="*60}')
    w = sum(r['win'] for r in results)
    wr = w / n * 100
    print(f'\n  ■ KPI 1: 1着的中率: {w}/{n} ({wr:.0f}%) → {"✅" if 30 <= wr <= 40 else "❌"}')

    t5_ok = sum(1 for r in results if r['avg_top5'] <= 9)
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5_ok}/{n} → {"✅" if t5_ok == n else "❌"}')

    ana = sum(r['ana'] for r in results)
    print(f'  ■ KPI 3: 穴馬検知: {ana}/{n}年 → {"✅" if ana >= 1 else "❌"}')

    ov2 = sum(1 for r in results if r['overlap'] >= 2)
    ovr = ov2 / n * 100
    print(f'  ■ KPI 4: 馬券内占有率: {ov2}/{n} ({ovr:.0f}%) → {"✅" if ovr >= 80 else "❌"}')

    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        ana_s = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["pred1_finish"]}着 3着内{r["overlap"]}/3 TOP5平均{r["avg_top5"]:.1f}着 {ana_s}')


if __name__ == '__main__':
    run_backtest()
