"""
特徴量エンジン v2 — TARGET CSVベース
csv_parser.pyの出力DataFrameから予測用特徴量を構築する。
上がり3F/ペース/馬場状態/補正タイム等、バイナリでは取れなかったデータを活用。
"""

import pandas as pd
import numpy as np
import time


def rolling_ema(series, alpha=0.3, n_past=10):
    """shift(1)してからEMAを計算（未来情報リーク防止）"""
    shifted = series.shift(1)
    result = shifted.copy()
    for i in range(1, len(shifted)):
        window = shifted.iloc[max(0, i - n_past + 1):i + 1].dropna()
        if len(window) > 0:
            w = np.array([(1 - alpha) ** j for j in range(len(window) - 1, -1, -1)])
            result.iloc[i] = np.average(window, weights=w)
    return result


def get_dist_band(kyori):
    """距離帯分類"""
    if pd.isna(kyori) or kyori <= 0:
        return 'unknown'
    if kyori <= 1400:
        return 'sprint'
    elif kyori <= 1800:
        return 'mile'
    elif kyori <= 2200:
        return 'inter'
    else:
        return 'long'


def classify_pace(pace_front, pace_front_std):
    """ペースをH/M/Sに分類"""
    if pd.isna(pace_front) or pd.isna(pace_front_std) or pace_front_std == 0:
        return 'M'
    diff = pace_front - pace_front_std
    if diff < -0.5:
        return 'H'  # ハイペース（基準より速い）
    elif diff > 0.5:
        return 'S'  # スローペース
    else:
        return 'M'


def build_features_v2(df, alpha=0.3, n_past=10):
    """
    メイン特徴量構築関数

    Parameters
    ----------
    df : DataFrame
        csv_parser.load_csv_data() の出力
    alpha : float
        EMA平滑化係数
    n_past : int
        EMA計算の最大過去走数

    Returns
    -------
    DataFrame
        特徴量付きDataFrame（1行=1馬1レース）
    """
    t0 = time.time()
    print('[features_v2] Building features...')

    df = df.copy()
    df_sorted = df.sort_values(['ketto_num', 'date']).reset_index(drop=True)

    # === 距離帯 ===
    df_sorted['dist_band'] = df_sorted['kyori'].apply(get_dist_band)

    # === ペース分類 ===
    df_sorted['pace_class'] = df_sorted.apply(
        lambda r: classify_pace(r['pace_front'], r['pace_front_std']), axis=1)

    # === レース内Z-score ===
    print('  Computing race-level z-scores...')
    for col in ['time_sec', 'agari_3f']:
        grouped = df_sorted.groupby('race_id')[col]
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        df_sorted[f'{col}_zscore'] = np.where(std > 0, (df_sorted[col] - mean) / std, 0)

    # === 馬ごとの特徴量 ===
    print('  Computing per-horse features...')

    # 初期化
    feat_cols_init = {
        'past_count': 0,
        'ema_finish': np.nan,
        'ema_time_zscore': np.nan,
        'ema_time_diff': np.nan,
        'ema_agari_3f': np.nan,         # 上がり3F EMA（新）
        'ema_agari_rank': np.nan,        # 上がり3F順位 EMA（新）
        'ema_agari_zscore': np.nan,      # 上がり3Fレース内Z EMA（新）
        'ema_pace_front': np.nan,        # 前半ペースEMA（新）
        'ema_pace_diff': np.nan,         # 前半-後半差 EMA（新）
        'win_rate': np.nan,
        'top3_rate': np.nan,
        'avg_run_style': np.nan,
        'same_dist_finish': np.nan,
        'same_surface_finish': np.nan,
        'same_baba_finish': np.nan,      # 同馬場状態着順（新）
        'interval_days': np.nan,
        'pace_h_time_diff': np.nan,      # ハイペース時着差EMA（新）
        'pace_s_time_diff': np.nan,      # スロー時着差EMA（新）
    }
    for col, default in feat_cols_init.items():
        df_sorted[col] = default

    # 通過順平均
    for corner in ['jyuni_1c', 'jyuni_2c', 'jyuni_3c']:
        df_sorted[f'avg_{corner}'] = np.nan

    # === 騎手・調教師 成績 ===
    print('  Computing jockey/trainer stats...')
    for role, code_col in [('jockey', 'kisyu_code'), ('trainer', 'chokyosi_code')]:
        # 全体成績
        role_stats = {}
        # 距離帯別成績
        role_dist_stats = {}

        wr = np.full(len(df_sorted), np.nan)
        tr = np.full(len(df_sorted), np.nan)
        dwr = np.full(len(df_sorted), np.nan)
        dtr = np.full(len(df_sorted), np.nan)

        for i in range(len(df_sorted)):
            row = df_sorted.iloc[i]
            code = row[code_col]
            band = row['dist_band']
            if pd.isna(code) or code == '' or code == '0':
                continue

            # 全体
            if code in role_stats:
                t, w, t3 = role_stats[code]
                if t > 0:
                    wr[i] = w / t
                    tr[i] = t3 / t

            # 距離帯別
            dkey = (code, band)
            if dkey in role_dist_stats:
                t, w, t3 = role_dist_stats[dkey]
                if t > 0:
                    dwr[i] = w / t
                    dtr[i] = t3 / t

            # 累積更新
            finish = row['finish']
            if pd.notna(finish) and finish > 0:
                for stats, key in [(role_stats, code), (role_dist_stats, dkey)]:
                    if key not in stats:
                        stats[key] = [0, 0, 0]
                    stats[key][0] += 1
                    if finish == 1:
                        stats[key][1] += 1
                    if finish <= 3:
                        stats[key][2] += 1

        df_sorted[f'{role}_win_rate'] = wr
        df_sorted[f'{role}_top3_rate'] = tr
        df_sorted[f'{role}_dist_win_rate'] = dwr
        df_sorted[f'{role}_dist_top3_rate'] = dtr

    # === 馬ごとのEMA特徴量 ===
    print('  Computing per-horse EMA features...')
    processed = 0
    total = df_sorted['ketto_num'].nunique()

    for ketto, idx in df_sorted.groupby('ketto_num').groups.items():
        horse = df_sorted.loc[idx].sort_values('date')
        n = len(horse)

        if n <= 1:
            processed += 1
            continue

        # past_count
        df_sorted.loc[horse.index, 'past_count'] = list(range(n))

        # EMA: 着順
        df_sorted.loc[horse.index, 'ema_finish'] = rolling_ema(
            horse['finish'].astype(float), alpha, n_past).values

        # EMA: タイムZ
        df_sorted.loc[horse.index, 'ema_time_zscore'] = rolling_ema(
            horse['time_sec_zscore'], alpha, n_past).values

        # EMA: 着差タイム
        df_sorted.loc[horse.index, 'ema_time_diff'] = rolling_ema(
            horse['time_diff'].astype(float), alpha, n_past).values

        # EMA: 上がり3F
        df_sorted.loc[horse.index, 'ema_agari_3f'] = rolling_ema(
            horse['agari_3f'], alpha, n_past).values

        # EMA: 上がり3F順位
        df_sorted.loc[horse.index, 'ema_agari_rank'] = rolling_ema(
            horse['agari_rank'].astype(float), alpha, n_past).values

        # EMA: 上がり3F Z-score
        df_sorted.loc[horse.index, 'ema_agari_zscore'] = rolling_ema(
            horse['agari_3f_zscore'], alpha, n_past).values

        # EMA: ペース（前半タイム）
        df_sorted.loc[horse.index, 'ema_pace_front'] = rolling_ema(
            horse['pace_front'], alpha, n_past).values

        # EMA: ペース差（前半-後半）→正=前傾、負=後傾
        pace_diff = horse['pace_front'] - horse['pace_rear']
        df_sorted.loc[horse.index, 'ema_pace_diff'] = rolling_ema(
            pace_diff, alpha, n_past).values

        # 勝率・複勝率
        wins = (horse['finish'] == 1).astype(float)
        top3s = (horse['finish'] <= 3).astype(float)
        df_sorted.loc[horse.index, 'win_rate'] = wins.shift(1).expanding().mean().values
        df_sorted.loc[horse.index, 'top3_rate'] = top3s.shift(1).expanding().mean().values

        # 脚質
        rs = horse['running_style'].replace(0, np.nan).astype(float)
        df_sorted.loc[horse.index, 'avg_run_style'] = rs.shift(1).expanding().mean().values

        # 休養日数
        df_sorted.loc[horse.index, 'interval_days'] = horse['date'].diff().dt.days.values

        # 通過順平均
        for corner in ['jyuni_1c', 'jyuni_2c', 'jyuni_3c']:
            vals = horse[corner].replace(0, np.nan).astype(float)
            df_sorted.loc[horse.index, f'avg_{corner}'] = vals.shift(1).expanding().mean().values

        # 同距離・同馬場・同馬場状態の成績
        for i_local in range(1, n):
            row_idx = horse.index[i_local]
            past = horse.iloc[max(0, i_local - n_past):i_local]
            cur = horse.iloc[i_local]

            # 同距離
            if pd.notna(cur.get('kyori')):
                same = past[past['kyori'] == cur['kyori']]
                if len(same) > 0:
                    df_sorted.loc[row_idx, 'same_dist_finish'] = same['finish'].mean()

            # 同芝ダ
            if pd.notna(cur.get('surface')) and cur['surface'] != '':
                same = past[past['surface'] == cur['surface']]
                if len(same) > 0:
                    df_sorted.loc[row_idx, 'same_surface_finish'] = same['finish'].mean()

            # 同馬場状態（良/稍/重/不良）
            if pd.notna(cur.get('baba')) and cur['baba'] != '':
                same = past[past['baba'] == cur['baba']]
                if len(same) > 0:
                    df_sorted.loc[row_idx, 'same_baba_finish'] = same['finish'].mean()

        # ペース別着差EMA（ハイペース/スロー）
        pace_classes = horse['pace_class'].values
        time_diffs = horse['time_diff'].astype(float).values
        h_td = np.full(n, np.nan)
        s_td = np.full(n, np.nan)
        for i_local in range(1, n):
            past_pc = pace_classes[:i_local]
            past_td = time_diffs[:i_local]
            h_mask = past_pc == 'H'
            s_mask = past_pc == 'S'
            if h_mask.sum() > 0:
                h_vals = past_td[h_mask][-n_past:]
                h_vals = h_vals[~np.isnan(h_vals)]
                if len(h_vals) > 0:
                    w = np.array([(1 - alpha) ** j for j in range(len(h_vals) - 1, -1, -1)])
                    h_td[i_local] = np.average(h_vals, weights=w)
            if s_mask.sum() > 0:
                s_vals = past_td[s_mask][-n_past:]
                s_vals = s_vals[~np.isnan(s_vals)]
                if len(s_vals) > 0:
                    w = np.array([(1 - alpha) ** j for j in range(len(s_vals) - 1, -1, -1)])
                    s_td[i_local] = np.average(s_vals, weights=w)

        df_sorted.loc[horse.index, 'pace_h_time_diff'] = h_td
        df_sorted.loc[horse.index, 'pace_s_time_diff'] = s_td

        processed += 1
        if processed % 10000 == 0:
            elapsed = time.time() - t0
            print(f'    {processed}/{total} horses... ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'  Done: {processed} horses in {elapsed:.0f}s')

    # === バイナリ互換カラム ===
    df_sorted['has_same_dist'] = df_sorted['same_dist_finish'].notna().astype(int)
    df_sorted['has_same_baba'] = df_sorted['same_baba_finish'].notna().astype(int)

    # === 前走クラス ===
    GRADE_CLASS_SCORE = {'A': 8, 'B': 7, 'C': 6, 'L': 5, 'E': 4}
    # grade_cdが数値の場合もある（7=未勝利等）
    def map_class(row):
        g = str(row.get('grade_cd', '')).strip()
        if g in GRADE_CLASS_SCORE:
            return GRADE_CLASS_SCORE[g]
        # クラス名から推定
        cn = str(row.get('class_name', '')).strip()
        if 'G1' in cn or 'GI' in cn:
            return 8
        if 'G2' in cn or 'GII' in cn:
            return 7
        if 'G3' in cn or 'GIII' in cn:
            return 6
        if '(L)' in cn or 'リステッド' in cn:
            return 5
        if 'OP' in cn or 'オープン' in cn:
            return 4
        if '3勝' in cn:
            return 3
        if '2勝' in cn:
            return 2
        return 1  # 未勝利/1勝/新馬

    df_sorted['race_class'] = df_sorted.apply(map_class, axis=1)
    df_sorted['prev_race_class'] = df_sorted.groupby('ketto_num')['race_class'].shift(1).fillna(1)

    # === prev_dist_diff ===
    df_sorted['prev_kyori'] = df_sorted.groupby('ketto_num')['kyori'].shift(1)
    df_sorted['prev_dist_diff'] = df_sorted['kyori'] - df_sorted['prev_kyori'].fillna(df_sorted['kyori'])

    # === 賞金log ===
    df_sorted['cum_prize'] = df_sorted.groupby('ketto_num')['prize'].transform(
        lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
    df_sorted['log_prize_money'] = np.log1p(df_sorted['cum_prize'])

    # === past_count > 0 フィルタ ===
    result = df_sorted[df_sorted['past_count'] > 0].copy()

    elapsed = time.time() - t0
    print(f'  Total: {len(result):,} rows in {elapsed:.0f}s')

    return result


# === 特徴量定義（predictor用） ===
FEATURES_V2 = [
    # 基本
    'wakuban', 'futan', 'bataijyu', 'zogen_sa', 'heads', 'barei',
    'past_count', 'interval_days',

    # 能力EMA
    'ema_finish', 'ema_time_zscore', 'ema_time_diff',

    # 上がり3F（新）
    'ema_agari_3f', 'ema_agari_rank', 'ema_agari_zscore',

    # ペース（新）
    'ema_pace_front', 'ema_pace_diff',
    'pace_h_time_diff', 'pace_s_time_diff',

    # 脚質・通過順
    'avg_run_style', 'avg_jyuni_1c', 'avg_jyuni_2c', 'avg_jyuni_3c',

    # 適性
    'same_dist_finish', 'same_surface_finish', 'same_baba_finish',
    'has_same_dist', 'has_same_baba',
    'prev_dist_diff',

    # 騎手・調教師
    'jockey_win_rate', 'jockey_top3_rate',
    'jockey_dist_win_rate', 'jockey_dist_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'trainer_dist_win_rate', 'trainer_dist_top3_rate',

    # クラス・賞金
    'prev_race_class', 'log_prize_money',

    # 成績
    'win_rate', 'top3_rate',
]

CAT_FEATURES_V2 = ['kisyu_code', 'chokyosi_code', 'sire_type']


if __name__ == '__main__':
    from csv_parser import load_csv_data
    df = load_csv_data()
    feat = build_features_v2(df)
    print(f'\nFeatures: {len(FEATURES_V2)} numeric + {len(CAT_FEATURES_V2)} categorical')
    print(f'Available in data: {sum(1 for f in FEATURES_V2 if f in feat.columns)}/{len(FEATURES_V2)}')

    # Save cache
    feat.to_pickle('data/features_v2_cache.pkl')
    print(f'Saved: data/features_v2_cache.pkl')
