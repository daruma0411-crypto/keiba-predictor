"""
競馬予測 特徴量エンジニアリング
過去走データから予測に使う特徴量を構築する
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_race_zscore(group, col):
    """レース内でのZ-score（標準化）を計算"""
    mean = group[col].mean()
    std = group[col].std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=group.index)
    return (group[col] - mean) / std


def estimate_running_style(row):
    """通過順位から脚質を推定（1=逃げ, 2=先行, 3=差し, 4=追込）"""
    corners = []
    for c in ['jyuni_3c', 'jyuni_4c']:
        val = row.get(c)
        if pd.notna(val) and val > 0:
            corners.append(val)
    if not corners:
        return 0  # 不明

    avg_pos = np.mean(corners)
    heads = row.get('heads', 16)
    if pd.isna(heads) or heads == 0:
        heads = 16

    ratio = avg_pos / heads
    if ratio <= 0.15:
        return 1  # 逃げ
    elif ratio <= 0.35:
        return 2  # 先行
    elif ratio <= 0.65:
        return 3  # 差し
    else:
        return 4  # 追込


def compute_ema(values, alpha=0.3):
    """指数移動平均を計算（直近ほど重い）"""
    if len(values) == 0:
        return np.nan
    weights = [(1 - alpha) ** i for i in range(len(values))]
    weights.reverse()
    weights = np.array(weights)
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return np.nan
    return np.average(values[valid], weights=weights[valid])


def build_past_race_features(df, n_past=5, ema_alpha=0.3):
    """
    各出走馬の過去走から特徴量を構築する

    Parameters
    ----------
    df : pd.DataFrame
        binary_parser.load_hanshin_data() で取得したデータ（全場・全クラス推奨）
    n_past : int
        参照する過去走数
    ema_alpha : float
        EMAの減衰率（大きいほど直近重視）

    Returns
    -------
    pd.DataFrame
        特徴量付きデータ
    """
    df = df.copy()
    df = df.sort_values(['ketto_num', 'date', 'race_num']).reset_index(drop=True)

    # レース内Z-scoreを先に計算
    # レース内Z-score
    time_zs = []
    l3f_zs = []
    for _, g in df.groupby('race_id'):
        time_zs.append(compute_race_zscore(g, 'time'))
        l3f_zs.append(compute_race_zscore(g, 'haron_time_l3'))
    df['time_zscore'] = pd.concat(time_zs).reindex(df.index)
    # NOTE: l3f_zscore is effectively dead — haron_time_l3 is always zero in the binary files.
    # Kept for schema compatibility but values will be uniformly zero.
    df['l3f_zscore'] = pd.concat(l3f_zs).reindex(df.index)

    # 脚質推定
    df['run_style'] = df.apply(estimate_running_style, axis=1)

    # 馬ごとに過去走の特徴量を計算（ベクトル化版）
    # まず馬ごとにソートされた状態で rolling/shift ベースで計算

    # EMA用のrolling関数
    def rolling_ema(series, alpha, n):
        """過去n走のEMAを各行で計算（自分自身は含まない）"""
        result = np.full(len(series), np.nan)
        vals = series.values
        for i in range(1, len(vals)):
            window = vals[max(0, i - n):i]
            valid = window[~np.isnan(window)]
            if len(valid) == 0:
                continue
            weights = np.array([(1 - alpha) ** j for j in range(len(valid) - 1, -1, -1)])
            result[i] = np.average(valid, weights=weights)
        return pd.Series(result, index=series.index)

    # === 騎手・調教師の累積成績を計算 ===
    print("  Computing jockey/trainer stats...")

    # 騎手の勝率・複勝率（日付順で累積、当該レースは含まない）
    df_sorted = df.sort_values('date')
    for role, code_col in [('jockey', 'kisyu_code'), ('trainer', 'chokyosi_code')]:
        if code_col not in df.columns:
            df[f'{role}_win_rate'] = np.nan
            df[f'{role}_top3_rate'] = np.nan
            continue

        role_stats = {}
        win_rates = np.full(len(df_sorted), np.nan)
        top3_rates = np.full(len(df_sorted), np.nan)

        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            code = row.get(code_col)
            if pd.isna(code) or code == '' or code == '0':
                continue

            # このレース前までの成績を取得
            if code in role_stats:
                total, wins, top3s = role_stats[code]
                if total > 0:
                    pos = df_sorted.index.get_loc(idx)
                    win_rates[pos] = wins / total
                    top3_rates[pos] = top3s / total

            # この結果を累積に追加
            finish = row.get('kakutei_jyuni')
            if pd.notna(finish) and finish > 0:
                if code not in role_stats:
                    role_stats[code] = [0, 0, 0]
                role_stats[code][0] += 1
                if finish == 1:
                    role_stats[code][1] += 1
                if finish <= 3:
                    role_stats[code][2] += 1

        df[f'{role}_win_rate'] = pd.Series(win_rates, index=df_sorted.index).reindex(df.index)
        df[f'{role}_top3_rate'] = pd.Series(top3_rates, index=df_sorted.index).reindex(df.index)

    # === 通過順位の詳細 ===
    for corner in ['jyuni_1c', 'jyuni_2c', 'jyuni_3c', 'jyuni_4c']:
        if corner in df.columns:
            # 過去走の平均通過順位（shift+expanding）
            col_name = f'avg_{corner}'
            df[col_name] = np.nan

    # 各馬のデータを処理
    print(f"  Processing {df['ketto_num'].nunique()} horses...")

    # 初期化
    df['past_count'] = 0
    df['ema_time_zscore'] = np.nan
    # NOTE: ema_l3f_zscore is dead — source data (haron_time_l3) is always zero in binaries.
    # Kept in schema for backward compatibility but will remain NaN.
    df['ema_l3f_zscore'] = np.nan
    df['ema_finish'] = np.nan
    df['win_rate'] = np.nan
    df['top3_rate'] = np.nan
    df['avg_run_style'] = np.nan
    df['same_dist_finish'] = np.nan
    df['same_surface_finish'] = np.nan
    df['interval_days'] = np.nan

    processed = 0
    total_horses = df['ketto_num'].nunique()

    for ketto_num, idx in df.groupby('ketto_num').groups.items():
        horse = df.loc[idx].sort_values('date')
        n = len(horse)

        if n <= 1:
            processed += 1
            continue

        # past_count: 自分より前の走数
        df.loc[horse.index, 'past_count'] = list(range(n))

        # EMA特徴量
        df.loc[horse.index, 'ema_time_zscore'] = rolling_ema(
            horse['time_zscore'], ema_alpha, n_past).values
        df.loc[horse.index, 'ema_finish'] = rolling_ema(
            horse['kakutei_jyuni'].astype(float), ema_alpha, n_past).values

        # 勝率・複勝率（累積、自分を含まない）
        wins = (horse['kakutei_jyuni'] == 1).astype(float)
        top3s = (horse['kakutei_jyuni'] <= 3).astype(float)
        cum_wins = wins.shift(1).expanding().mean()
        cum_top3 = top3s.shift(1).expanding().mean()
        df.loc[horse.index, 'win_rate'] = cum_wins.values
        df.loc[horse.index, 'top3_rate'] = cum_top3.values

        # 平均脚質
        rs = horse['run_style'].replace(0, np.nan).astype(float)
        cum_rs = rs.shift(1).expanding().mean()
        df.loc[horse.index, 'avg_run_style'] = cum_rs.values

        # 休養日数
        df.loc[horse.index, 'interval_days'] = horse['date'].diff().dt.days.values

        # 平均通過順位（各コーナー）
        for corner in ['jyuni_1c', 'jyuni_2c', 'jyuni_3c', 'jyuni_4c']:
            if corner in horse.columns:
                vals = horse[corner].replace(0, np.nan).astype(float)
                cum_avg = vals.shift(1).expanding().mean()
                df.loc[horse.index, f'avg_{corner}'] = cum_avg.values

        # 同距離・同馬場の成績（シンプル版：直近n_past走から）
        if 'kyori' in horse.columns and 'surface' in horse.columns:
            for i_local in range(1, n):
                row_idx = horse.index[i_local]
                past_rows = horse.iloc[max(0, i_local - n_past):i_local]
                cur_kyori = horse.iloc[i_local].get('kyori')
                cur_surface = horse.iloc[i_local].get('surface')

                if pd.notna(cur_kyori):
                    same_d = past_rows[past_rows['kyori'] == cur_kyori]
                    if len(same_d) > 0:
                        df.loc[row_idx, 'same_dist_finish'] = same_d['kakutei_jyuni'].mean()

                if pd.notna(cur_surface) and cur_surface != '':
                    same_s = past_rows[past_rows['surface'] == cur_surface]
                    if len(same_s) > 0:
                        df.loc[row_idx, 'same_surface_finish'] = same_s['kakutei_jyuni'].mean()

        processed += 1
        if processed % 10000 == 0:
            print(f"  {processed}/{total_horses} horses processed...")

    print(f"  Done: {processed} horses processed")

    # past_featとして返す（build_all_featuresとの互換性のため）
    feat_cols = ['past_count', 'ema_time_zscore', 'ema_l3f_zscore', 'ema_finish',
                 'win_rate', 'top3_rate', 'avg_run_style',
                 'same_dist_finish', 'same_surface_finish', 'interval_days',
                 'jockey_win_rate', 'jockey_top3_rate',
                 'trainer_win_rate', 'trainer_top3_rate',
                 'avg_jyuni_1c', 'avg_jyuni_2c',
                 'avg_jyuni_3c', 'avg_jyuni_4c']
    feat_df = df[feat_cols].copy()

    return feat_df


def build_race_context_features(df):
    """
    レース単位の文脈特徴量（レース前に判明するもの）

    Parameters
    ----------
    df : pd.DataFrame
        特徴量構築対象のデータ

    Returns
    -------
    pd.DataFrame
        レース文脈特徴量
    """
    context = pd.DataFrame(index=df.index)

    # 枠番・馬番
    context['wakuban'] = df['wakuban']
    context['umaban'] = df['umaban']

    # 斤量
    context['futan'] = df['futan']

    # 馬体重・増減
    context['bataijyu'] = df['bataijyu']
    context['zogen_sa'] = df['zogen_sa']

    # 性別（数値化）
    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    context['sex'] = df['sex'].map(sex_map).fillna(0)

    # 年齢
    context['barei'] = df['barei']

    # 距離
    if 'kyori' in df.columns:
        context['kyori'] = df['kyori']

    # 芝/ダート（数値化）
    if 'surface' in df.columns:
        context['is_turf'] = (df['surface'] == '芝').astype(int)

    # 頭数（レースごとの出走馬数から計算）
    race_counts = df.groupby('race_id')['ketto_num'].transform('count')
    context['heads'] = race_counts

    # Embedding用ID（カテゴリ特徴量）
    context['kisyu_code'] = df['kisyu_code']
    context['chokyosi_code'] = df['chokyosi_code']
    context['banusi_code'] = df['banusi_code']

    return context


def build_all_features(df, n_past=5, ema_alpha=0.3):
    """
    全特徴量を構築してレース予測用DataFrameを返す

    Parameters
    ----------
    df : pd.DataFrame
        load_hanshin_data()の出力
    n_past : int
        過去走参照数
    ema_alpha : float
        EMA減衰率

    Returns
    -------
    pd.DataFrame
        予測に使える全特徴量 + ラベル(kakutei_jyuni)
    """
    df = df.reset_index(drop=True)
    original_index = df.index.copy()

    print("Building past race features...")
    past_feat = build_past_race_features(df, n_past, ema_alpha)
    # past_featはdf内部でsort→reset_indexされるので、元のdfと行が対応しない
    # past_featのインデックスはsort後のもの → 元のdfのインデックスに戻す必要がある

    print("Building race context features...")
    # ctx_featは元のdfと同じ順序
    ctx_feat = build_race_context_features(df)

    # past_featを元のdfの順序に合わせるため、dfを再度同じソートで処理
    df_sorted = df.sort_values(['ketto_num', 'date', 'race_num']).reset_index(drop=True)

    # past_featはdf_sortedと同じ順序なので、df_sortedのメタ情報を使う
    meta = pd.DataFrame({
        'finish': df_sorted['kakutei_jyuni'].values,
        'is_win': (df_sorted['kakutei_jyuni'] == 1).astype(int).values,
        'is_top3': (df_sorted['kakutei_jyuni'] <= 3).astype(int).values,
        'race_id': df_sorted['race_id'].values,
        'date': df_sorted['date'].values,
        'horse_name': df_sorted['horse_name'].values,
        'ketto_num': df_sorted['ketto_num'].values,
        'odds': df_sorted['odds'].values,
        'wakuban': df_sorted['wakuban'].values,
        'umaban': df_sorted['umaban'].values,
        'futan': df_sorted['futan'].values,
        'bataijyu': df_sorted['bataijyu'].values,
        'zogen_sa': df_sorted['zogen_sa'].values,
        'sex': df_sorted['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0).values,
        'barei': df_sorted['barei'].values,
        'kyori': df_sorted.get('kyori', pd.Series(np.nan, index=df_sorted.index)).values,
        'is_turf': (df_sorted.get('surface', pd.Series('', index=df_sorted.index)) == '芝').astype(int).values,
        'kisyu_code': df_sorted['kisyu_code'].values,
        'kisyu_name': df_sorted['kisyu_name'].values if 'kisyu_name' in df_sorted.columns else '',
        'chokyosi_code': df_sorted['chokyosi_code'].values,
        'banusi_code': df_sorted['banusi_code'].values,
        'grade_cd': df_sorted['grade_cd'].values if 'grade_cd' in df_sorted.columns else '',
        'class_cd': df_sorted['class_cd'].values if 'class_cd' in df_sorted.columns else '',
    })

    # 頭数
    meta_df = pd.DataFrame(meta)
    race_counts = meta_df.groupby('race_id')['ketto_num'].transform('count')
    meta_df['heads'] = race_counts

    # past_featとmetaを結合（同じ順序）
    past_feat = past_feat.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)
    result = pd.concat([meta_df, past_feat], axis=1)

    # 重複カラムを除去
    result = result.loc[:, ~result.columns.duplicated()]

    # Binary indicators for NaN-heavy features
    result['has_same_dist'] = result['same_dist_finish'].notna().astype(int)
    if 'long_stretch_avg' in result.columns:
        result['has_long_stretch'] = result['long_stretch_avg'].notna().astype(int)
    else:
        result['has_long_stretch'] = 0

    # 初出走を除外（過去走がない馬は予測できない）
    result = result[result['past_count'] > 0].copy()

    print(f"Features built: {len(result)} samples, {result['race_id'].nunique()} races")

    return result


# 数値特徴量カラム（モデル入力用）
NUMERIC_FEATURES = [
    'wakuban', 'umaban', 'futan', 'bataijyu', 'zogen_sa',
    'sex', 'barei', 'kyori', 'is_turf', 'heads',
    'past_count', 'ema_time_zscore', 'ema_finish',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'avg_jyuni_1c', 'avg_jyuni_2c',
    'avg_jyuni_3c', 'avg_jyuni_4c',
    'has_same_dist',
]

# カテゴリ特徴量（Embedding用）
CATEGORICAL_FEATURES = [
    'kisyu_code', 'chokyosi_code', 'banusi_code',
]


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.binary_parser import load_all_data, filter_open_class

    # 全場のデータで過去走を参照し、阪神OP以上をフィルタ
    print("Loading ALL venue data (2015-2025)...")
    df_all = load_all_data(years=range(2015, 2026))
    print(f"All data: {len(df_all)} records, {df_all['race_id'].nunique()} races")

    print("\nBuilding features on ALL data...")
    features = build_all_features(df_all)

    # 阪神OP以上のみフィルタ
    if 'race_id' in features.columns:
        hanshin_open = filter_open_class(df_all[df_all['place_code'] == '09'])
        target_race_ids = set(hanshin_open['race_id'].unique())
        features_target = features[features['race_id'].isin(target_race_ids)].copy()
        print(f"\nHanshin Open+: {len(features_target)} samples, {features_target['race_id'].nunique()} races")
    else:
        features_target = features

    print(f"\nFeature stats:")
    for col in NUMERIC_FEATURES:
        if col in features_target.columns:
            v = features_target[col]
            print(f"  {col:25s}: mean={v.mean():8.2f}  std={v.std():8.2f}  nan%={v.isna().mean()*100:5.1f}")
