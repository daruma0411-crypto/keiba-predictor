"""
競馬予測 ニューラルネットワークモデル
各馬の能力値（μ, σ）を推定する
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class HorseRaceModel(nn.Module):
    """
    各馬の能力値（平均μと不確実性σ）を出力するNN
    MCシミュレーションの入力として使用
    """

    def __init__(self, num_features, embedding_dims=None):
        """
        Parameters
        ----------
        num_features : int
            数値特徴量の数
        embedding_dims : dict
            {カテゴリ名: (vocab_size, embed_dim)} の辞書
        """
        super().__init__()

        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        if embedding_dims:
            for name, (vocab_size, embed_dim) in embedding_dims.items():
                self.embeddings[name] = nn.Embedding(vocab_size, embed_dim)
                total_embed_dim += embed_dim

        input_dim = num_features + total_embed_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # μ (能力値の平均) とσ (不確実性) を出力
        self.mu_head = nn.Linear(32, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus(),  # σは正の値
        )

    def forward(self, numeric_features, categorical_features=None):
        """
        Parameters
        ----------
        numeric_features : Tensor (batch, num_features)
        categorical_features : dict {name: Tensor (batch,)}

        Returns
        -------
        mu : Tensor (batch, 1) - 能力値の平均（小さいほど強い）
        sigma : Tensor (batch, 1) - 不確実性
        """
        parts = [numeric_features]

        if categorical_features:
            for name, indices in categorical_features.items():
                if name in self.embeddings:
                    parts.append(self.embeddings[name](indices))

        x = torch.cat(parts, dim=1)
        h = self.network(x)

        mu = self.mu_head(h)
        sigma = self.sigma_head(h) + 0.1  # 最低限の不確実性

        return mu, sigma


class RacePredictor:
    """
    学習・予測を行うラッパークラス
    """

    def __init__(self, numeric_features, categorical_features=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features or []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.device = torch.device('cpu')

    def _prepare_data(self, df, fit=False):
        """DataFrameからTensorを準備"""
        # 数値特徴量
        X_numeric = df[self.numeric_features].copy()
        X_numeric = X_numeric.fillna(X_numeric.median())

        if fit:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # カテゴリ特徴量
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
                    # 未知のカテゴリを処理
                    values = values.map(
                        lambda x: x if x in le.classes_ else le.classes_[0])

                encoded = le.transform(values)
                cat_tensors[cat] = torch.LongTensor(encoded).to(self.device)

        return X_tensor, cat_tensors

    def train(self, df, epochs=100, lr=0.001, batch_size=256, seed=42):
        """モデルを学習する（Gaussian NLL）"""
        # 再現性のためシード固定
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 着順0（取消/除外/競走中止）を除外
        df = df[df['finish'] > 0].copy()

        # ラベル: 着順を能力値として使う（正規化）
        y = df['finish'].values.astype(float)
        heads = df['heads'].values.astype(float)
        heads[heads == 0] = 16
        y_normalized = (y - 1) / (heads - 1)
        y_tensor = torch.FloatTensor(y_normalized).unsqueeze(1).to(self.device)

        X_tensor, cat_tensors = self._prepare_data(df, fit=True)

        # モデル初期化
        embedding_dims = {}
        for cat in self.categorical_features:
            if cat in self.label_encoders:
                vocab_size = len(self.label_encoders[cat].classes_)
                embed_dim = min(50, (vocab_size + 1) // 2)
                embedding_dims[cat] = (vocab_size, embed_dim)

        self.model = HorseRaceModel(
            num_features=len(self.numeric_features),
            embedding_dims=embedding_dims,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 学習ループ
        self.model.train()
        n = len(X_tensor)

        for epoch in range(epochs):
            indices = torch.randperm(n)
            total_loss = 0

            for i in range(0, n, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                cat_batch = {}
                for cat, tensor in cat_tensors.items():
                    cat_batch[cat] = tensor[batch_idx]

                mu, sigma = self.model(X_batch, cat_batch)

                loss = torch.mean(
                    0.5 * torch.log(sigma ** 2) +
                    0.5 * ((y_batch - mu) / sigma) ** 2
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_idx)

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, df):
        """
        各馬の能力値(μ, σ)を予測する

        Returns
        -------
        pd.DataFrame with columns: mu, sigma, horse_name, ketto_num, odds
        """
        self.model.eval()
        X_tensor, cat_tensors = self._prepare_data(df, fit=False)

        with torch.no_grad():
            mu, sigma = self.model(X_tensor, cat_tensors)

        result = pd.DataFrame({
            'mu': mu.squeeze().numpy(),
            'sigma': sigma.squeeze().numpy(),
            'horse_name': df['horse_name'].values,
            'ketto_num': df['ketto_num'].values,
            'umaban': df['umaban'].values,
            'odds': df['odds'].values if 'odds' in df.columns else np.nan,
        })

        return result


def monte_carlo_simulation(predictions, race_features=None, n_simulations=100000):
    """
    展開要素を含むモンテカルロシミュレーション

    展開ロジック:
    - 逃げ馬が多い → ハイペース → 差し追込有利
    - 逃げ馬が少ない → スローペース → 逃げ先行有利
    - 内枠の先行馬は位置取り有利
    - 外枠の追込馬は不利

    Parameters
    ----------
    predictions : pd.DataFrame
        predict()の出力 (mu, sigma, horse_name, umaban, odds)
    race_features : pd.DataFrame or None
        出走馬の特徴量（avg_run_style, wakuban等）
    n_simulations : int
        シミュレーション回数
    """
    n_horses = len(predictions)
    mu = predictions['mu'].values.copy()
    sigma = predictions['sigma'].values.copy()

    # 展開要素の準備
    run_styles = np.full(n_horses, 2.5)  # デフォルト=先行
    wakubans = np.arange(1, n_horses + 1)  # デフォルト
    umabans = predictions['umaban'].values.astype(float)

    if race_features is not None and len(race_features) == n_horses:
        if 'avg_run_style' in race_features.columns:
            rs = race_features['avg_run_style'].values
            rs = np.where(np.isnan(rs), 2.5, rs)
            run_styles = rs
        if 'wakuban' in race_features.columns:
            wakubans = race_features['wakuban'].values.astype(float)

    # 逃げ馬数を計算（脚質1.0〜1.5が逃げ）
    n_nige = np.sum(run_styles <= 1.5)
    n_senkou = np.sum((run_styles > 1.5) & (run_styles <= 2.5))

    # ペース判定: 逃げ馬が多い→ハイペース
    # pace_factor: 正=ハイペース(差し有利), 負=スロー(逃げ有利)
    pace_base = (n_nige - 1.5) * 0.3  # 逃げ1頭=スロー、3頭以上=ハイ

    # MC実行（ベクトル化）
    # 1) 基本能力をサンプリング (n_simulations x n_horses)
    ability = np.random.normal(
        mu[np.newaxis, :], sigma[np.newaxis, :],
        size=(n_simulations, n_horses)
    )

    # 2) ペース変動 (n_simulations,)
    pace = pace_base + np.random.normal(0, 0.15, size=n_simulations)

    # 3) 展開補正（ベクトル化）
    # 差し追込マスク（style >= 3.0）
    sashi_mask = (run_styles >= 3.0).astype(float)  # (n_horses,)
    # 逃げマスク（style <= 1.5）
    nige_mask = (run_styles <= 1.5).astype(float)
    # 先行マスク（style <= 2.5）
    senkou_mask = (run_styles <= 2.5).astype(float)
    # 追込マスク（style >= 3.5）
    oikomi_mask = (run_styles >= 3.5).astype(float)

    # ハイペース→差し有利、逃げ不利
    pace_2d = pace[:, np.newaxis]  # (n_simulations, 1)
    ability -= pace_2d * 0.03 * sashi_mask[np.newaxis, :]  # 差し有利
    ability += pace_2d * 0.04 * nige_mask[np.newaxis, :]   # 逃げ不利

    # 内枠(1-3)の先行馬は有利
    inner_senkou = ((wakubans <= 3) & (run_styles <= 2.5)).astype(float)
    ability -= 0.01 * inner_senkou[np.newaxis, :]

    # 外枠(6-8)の追込馬は不利
    outer_oikomi = ((wakubans >= 6) & (run_styles >= 3.5)).astype(float)
    ability += 0.01 * outer_oikomi[np.newaxis, :]

    # 位置取りのランダム性
    ability += np.random.normal(0, 0.02, size=(n_simulations, n_horses))

    # 4) 着順決定
    rankings_all = np.argsort(np.argsort(ability, axis=1), axis=1) + 1

    # 集計
    results = predictions[['horse_name', 'umaban', 'odds']].copy()

    for pos in range(1, min(n_horses + 1, 19)):
        results[f'prob_{pos}'] = (rankings_all == pos).mean(axis=0)

    results['expected_rank'] = rankings_all.mean(axis=0)
    results['win_prob'] = results['prob_1']
    results['top3_prob'] = results[['prob_1', 'prob_2', 'prob_3']].sum(axis=1)

    if 'odds' in results.columns:
        results['ev_win'] = results['win_prob'] * results['odds']

    results = results.sort_values('expected_rank')

    return results


def explain_prediction(predictions, mc_results, feature_df, numeric_features):
    """
    予測根拠を生成する

    Parameters
    ----------
    predictions : pd.DataFrame
        predict()の出力
    mc_results : pd.DataFrame
        monte_carlo_simulation()の出力
    feature_df : pd.DataFrame
        特徴量データ
    numeric_features : list
        数値特徴量名

    Returns
    -------
    dict
        各馬の予測根拠
    """
    explanations = {}

    # 特徴量の平均値（比較用）
    feat_means = feature_df[numeric_features].mean()
    feat_stds = feature_df[numeric_features].std()

    for _, row in mc_results.head(5).iterrows():
        horse = row['horse_name']
        umaban = row['umaban']

        # 該当馬の特徴量
        horse_feat = feature_df[feature_df['horse_name'] == horse]
        if len(horse_feat) == 0:
            continue
        horse_feat = horse_feat.iloc[-1]

        strengths = []
        weaknesses = []

        for feat in numeric_features:
            val = horse_feat.get(feat)
            if pd.isna(val):
                continue

            mean = feat_means[feat]
            std = feat_stds[feat]
            if std == 0:
                continue

            z = (val - mean) / std

            # 特徴量ごとに「良い方向」を判定
            if feat in ['ema_time_zscore', 'ema_finish', 'same_dist_finish',
                        'same_surface_finish']:
                # 小さい方が良い
                if z < -0.5:
                    strengths.append(f"{feat}: {val:.2f} (平均より優秀)")
                elif z > 0.5:
                    weaknesses.append(f"{feat}: {val:.2f} (平均より劣る)")
            elif feat in ['win_rate', 'top3_rate']:
                # 大きい方が良い
                if z > 0.5:
                    strengths.append(f"{feat}: {val:.1%} (高い)")
                elif z < -0.5:
                    weaknesses.append(f"{feat}: {val:.1%} (低い)")

        explanations[horse] = {
            'umaban': umaban,
            'win_prob': f"{row['win_prob']:.1%}",
            'top3_prob': f"{row['top3_prob']:.1%}",
            'expected_rank': f"{row['expected_rank']:.1f}",
            'strengths': strengths,
            'weaknesses': weaknesses,
        }

    return explanations
