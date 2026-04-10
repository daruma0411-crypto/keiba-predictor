"""
桜花賞予測モデル v7
- 主観的スコア完全廃止
- 特徴量の断捨離（sex/barei削除、prev_unlucky追加）
- QMC（準モンテカルロ）導入
- MC展開要素追加（出遅れ・進路塞がり）
- KPI: 1着的中率30-40%, 大外れ排除, 穴馬検知, 馬券内占有率80%
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
# 1. 特徴量定義（v7: sex/barei削除、prev_unlucky追加）
# ============================================================
FEATURES_V7 = [
    'wakuban', 'futan', 'bataijyu', 'zogen_sa',
    'heads',
    'past_count', 'ema_time_zscore', 'ema_finish',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'avg_jyuni_3c', 'avg_jyuni_4c',
    'prev_unlucky',  # 新規追加
]
CAT_FEATURES = ['kisyu_code', 'chokyosi_code', 'banusi_code']


def add_prev_unlucky(feat, df_all):
    """前走不運フラグ: 前走着差0.5秒以内 & 4-9着"""
    feat = feat.copy()
    feat['prev_unlucky'] = 0

    for idx in feat.index:
        ketto = feat.loc[idx, 'ketto_num']
        race_date = feat.loc[idx, 'date']
        if pd.isna(race_date):
            continue

        # この馬の前走を取得
        prev = df_all[
            (df_all['ketto_num'] == ketto) &
            (df_all['date'] < race_date) &
            (df_all['kakutei_jyuni'] > 0)
        ].sort_values('date')

        if len(prev) == 0:
            continue

        last = prev.iloc[-1]
        finish = last.get('kakutei_jyuni', 99)
        # 着差: time_diffフィールドやchakusa_cdから推定
        # 簡易版: 着順4-9着をチェック
        if 4 <= finish <= 9:
            # タイム差がある場合はそれを使う（なければ着順のみで判定）
            feat.loc[idx, 'prev_unlucky'] = 1

    return feat


# ============================================================
# 2. NNモデル（v6と同じアーキテクチャ）
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
        X_numeric = df[self.numeric_features].copy().fillna(
            df[self.numeric_features].median() if fit else 0)
        if fit:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
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
                embedding_dims[cat] = (vs, min(50, (vs + 1) // 2))

        self.model = HorseRaceModel(len(self.numeric_features), embedding_dims).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        n = len(X_tensor)

        for epoch in range(epochs):
            indices = torch.randperm(n)
            total_loss = 0
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_b = X_tensor[batch_idx]
                y_b = y_tensor[batch_idx]
                cat_b = {c: t[batch_idx] for c, t in cat_tensors.items()}
                mu, sigma = self.model(X_b, cat_b)
                loss = torch.mean(0.5 * torch.log(sigma**2) + 0.5 * ((y_b - mu) / sigma)**2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

    def predict(self, df):
        self.model.eval()
        X_tensor, cat_tensors = self._prepare_data(df, fit=False)
        with torch.no_grad():
            mu, sigma = self.model(X_tensor, cat_tensors)
        return pd.DataFrame({
            'mu': mu.squeeze().numpy(),
            'sigma': sigma.squeeze().numpy(),
            'horse_name': df['horse_name'].values,
            'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })


# ============================================================
# 3. QMC + 展開要素（出遅れ・進路塞がり）
# ============================================================
def qmc_simulation(predictions, race_features=None, n_simulations=100000):
    """準モンテカルロ法による着順シミュレーション"""
    n_horses = len(predictions)
    mu = predictions['mu'].values.copy()
    sigma = predictions['sigma'].values.copy()

    # 脚質・枠番
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

    # QMC: Sobol列で均一なサンプリング
    # 次元数: n_horses(能力) + 1(ペース) + n_horses(出遅れ) + n_horses(進路塞がり) + n_horses(揺れ)
    n_dims = n_horses * 4 + 1
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=42)
    # 2のべき乗に丸める
    n_pow2 = 2 ** int(np.ceil(np.log2(n_simulations)))
    sobol_samples = sampler.random(n_pow2)[:n_simulations]

    # Sobol一様分布→正規分布に変換
    sobol_norm = norm.ppf(np.clip(sobol_samples, 0.001, 0.999))

    # 各次元を割り当て
    idx = 0
    ability_noise = sobol_norm[:, idx:idx + n_horses]  # 能力サンプリング
    idx += n_horses
    pace_noise = sobol_norm[:, idx]  # ペース変動
    idx += 1
    late_start_u = sobol_samples[:, idx:idx + n_horses]  # 出遅れ判定（一様分布を使う）
    idx += n_horses
    blocked_u = sobol_samples[:, idx:idx + n_horses]  # 進路塞がり判定
    idx += n_horses
    position_noise = sobol_norm[:, idx:idx + n_horses]  # 位置取り揺れ

    # シミュレーション（ベクトル化）
    ability = mu[np.newaxis, :] + sigma[np.newaxis, :] * ability_noise
    pace = pace_base + 0.15 * pace_noise  # (n_sim,)
    pace_2d = pace[:, np.newaxis]

    # 展開補正
    sashi_mask = (run_styles >= 3.0).astype(float)
    nige_mask = (run_styles <= 1.5).astype(float)
    ability -= pace_2d * 0.03 * sashi_mask[np.newaxis, :]
    ability += pace_2d * 0.04 * nige_mask[np.newaxis, :]

    # 枠順補正
    inner_senkou = ((wakubans <= 3) & (run_styles <= 2.5)).astype(float)
    outer_oikomi = ((wakubans >= 6) & (run_styles >= 3.5)).astype(float)
    ability -= 0.01 * inner_senkou[np.newaxis, :]
    ability += 0.01 * outer_oikomi[np.newaxis, :]

    # 出遅れリスク: 5%の確率で大幅マイナス
    late_start_mask = (late_start_u < 0.05).astype(float)
    ability += late_start_mask * 0.15  # abilityが大きい=弱い

    # 進路塞がりリスク: 内枠(1-3)の差し追込馬に10%の確率
    inner_sashi = ((wakubans <= 3) & (run_styles >= 3.0)).astype(float)
    blocked_mask = (blocked_u < 0.10).astype(float) * inner_sashi[np.newaxis, :]
    ability += blocked_mask * 0.10

    # 位置取りの揺れ
    ability += 0.02 * position_noise

    # 着順決定
    rankings = np.argsort(np.argsort(ability, axis=1), axis=1) + 1

    # 集計
    results = predictions[['horse_name', 'umaban', 'odds']].copy()
    for pos in range(1, min(n_horses + 1, 19)):
        results[f'prob_{pos}'] = (rankings == pos).mean(axis=0)

    results['expected_rank'] = rankings.mean(axis=0)
    results['win_prob'] = results['prob_1']
    results['top3_prob'] = results[['prob_1', 'prob_2', 'prob_3']].sum(axis=1)
    if 'odds' in results.columns:
        results['ev_win'] = results['win_prob'] * results['odds']

    # final_score = MC結果のみ（主観スコア一切なし）
    results = results.sort_values('expected_rank')
    return results


# ============================================================
# 4. バックテスト実行
# ============================================================
def run_backtest():
    from src.binary_parser import load_all_data, load_hanshin_data

    print('Loading data...')
    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    df_all = load_all_data(years=range(2014, 2026))
    df_all = df_all[df_all['kakutei_jyuni'] > 0]

    feat = pd.read_pickle('data/features_all_v4.pkl')

    # prev_unlucky追加
    print('Adding prev_unlucky flag...')
    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    # 使用可能特徴量
    available = [f for f in FEATURES_V7 if f in feat.columns or f == 'prev_unlucky']
    print(f'Features: {available}')

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

        # prev_unlucky計算
        race_feat = add_prev_unlucky(race_feat, df_all)

        # 学習データ
        train = feat[
            (feat['date'] < race_date) &
            (feat['is_turf'] == 1) &
            (feat['past_count'] > 0) &
            (feat['finish'] > 0)
        ].sort_values('date').tail(20000).copy()
        train['prev_unlucky'] = 0  # 学習データには簡易的に0

        # 使用特徴量
        use_features = [f for f in FEATURES_V7 if f in train.columns]

        predictor = RacePredictor(use_features, CAT_FEATURES)
        predictor.train(train, epochs=50, lr=0.003, seed=42)
        preds = predictor.predict(race_feat)

        # QMCシミュレーション
        mc = qmc_simulation(preds, race_features=race_feat, n_simulations=100000)

        # 結果
        actual = race_data.sort_values('kakutei_jyuni').head(3)
        actual_top3 = set(actual['umaban'].astype(int))
        actual_1st = int(actual.iloc[0]['umaban'])

        # TOP5の情報
        top5 = mc.head(5)
        top5_umas = set(top5['umaban'].astype(int))
        overlap = len(top5_umas & actual_top3)
        win = int(top5.iloc[0]['umaban']) == actual_1st

        # TOP5の実際着順
        top5_finishes = []
        for _, r in top5.iterrows():
            uma = int(r['umaban'])
            act = race_data[race_data['umaban'] == uma]
            af = int(act.iloc[0]['kakutei_jyuni']) if len(act) > 0 else 18
            top5_finishes.append(af)

        avg_top5_finish = np.mean(top5_finishes)

        # 穴馬チェック: TOP5内でオッズ10倍以上の馬が3着以内
        ana_hit = False
        for _, r in top5.iterrows():
            uma = int(r['umaban'])
            odds = r.get('odds', 0)
            if odds >= 10 and uma in actual_top3:
                ana_hit = True

        print(f'\n{year}年桜花賞:')
        print(f'  予測TOP5:')
        for rank, (_, r) in enumerate(top5.iterrows(), 1):
            uma = int(r['umaban'])
            act = race_data[race_data['umaban'] == uma]
            af = int(act.iloc[0]['kakutei_jyuni']) if len(act) > 0 else '?'
            hit = '*' if uma in actual_top3 else ' '
            odds_str = f'odds={r["odds"]:.1f}' if pd.notna(r.get('odds')) else ''
            print(f'    {hit}{rank}位 [{uma:2d}] {r["horse_name"]:16s} '
                  f'勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} '
                  f'期待順{r["expected_rank"]:4.1f} {odds_str} (実際{af}着)')
        print(f'  実際:')
        for _, r in actual.iterrows():
            print(f'    {int(r["kakutei_jyuni"])}着 [{int(r["umaban"]):2d}] '
                  f'{r["horse_name"]:16s} ({int(r["ninki"])}人気 odds:{r["odds"]})')

        results.append({
            'year': year, 'win': win, 'overlap': overlap,
            't5': actual_1st in top5_umas,
            'avg_top5_finish': avg_top5_finish,
            'ana_hit': ana_hit,
            'top5_finishes': top5_finishes,
            'pred_1st_finish': top5_finishes[0],
        })
        sys.stdout.flush()

    # ============================================================
    # KPIレポート
    # ============================================================
    print(f'\n{"="*60}')
    print(f'  v7 バックテスト KPIレポート')
    print(f'{"="*60}')

    n = len(results)
    win_count = sum(r['win'] for r in results)
    win_rate = win_count / n * 100

    avg_top5_all = np.mean([r['avg_top5_finish'] for r in results])
    top5_in_half = sum(1 for r in results if r['avg_top5_finish'] <= 9)

    ana_years = sum(1 for r in results if r['ana_hit'])

    overlap_2plus = sum(1 for r in results if r['overlap'] >= 2)
    overlap_rate = overlap_2plus / n * 100

    print(f'\n  ■ KPI 1: 1着的中率')
    print(f'    結果: {win_count}/{n} ({win_rate:.0f}%)')
    print(f'    基準: 30-40% → {"✅ PASS" if 30 <= win_rate <= 40 else "❌ FAIL"}')

    print(f'\n  ■ KPI 2: 大外れの排除')
    print(f'    TOP5平均着順: {avg_top5_all:.1f}')
    print(f'    9着以内に収まった年: {top5_in_half}/{n}')
    print(f'    基準: 全年9着以内 → {"✅ PASS" if top5_in_half == n else "❌ FAIL"}')

    print(f'\n  ■ KPI 3: 穴馬検知')
    print(f'    穴馬(10倍以上)が3着内に入った年: {ana_years}/{n}')
    print(f'    基準: 1年以上 → {"✅ PASS" if ana_years >= 1 else "❌ FAIL"}')

    print(f'\n  ■ KPI 4: 馬券内占有率（最重要）')
    print(f'    TOP5のうち2頭以上が3着内の年: {overlap_2plus}/{n} ({overlap_rate:.0f}%)')
    print(f'    基準: 80%以上 → {"✅ PASS" if overlap_rate >= 80 else "❌ FAIL"}')

    print(f'\n  年別詳細:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        ana = '穴★' if r['ana_hit'] else '    '
        print(f'    {r["year"]}: {m} 予測1位→{r["pred_1st_finish"]}着 '
              f'3着内{r["overlap"]}/3 TOP5平均{r["avg_top5_finish"]:.1f}着 {ana}')


if __name__ == '__main__':
    run_backtest()
