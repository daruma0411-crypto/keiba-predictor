"""
学習データ期間の最適化バックテスト

比較する3軸:
  1. 期間: 2年 / 3年 / 5年
  2. 方式: tail(N件) / 固定年数
  3. 重み: 均等 / 直近重み(時間減衰サンプリング)

評価指標:
  - top3_overlap: 予測TOP5に実際の3着以内が何頭含まれるか
  - rank_corr: 予測順位と実着順のSpearman相関
  - winner_in_top5: 1着馬が予測TOP5に入っているか
"""

import sys
import os
import warnings
import time

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from itertools import product

from src.predictor_v2 import PredictorV2
from src.features_v2 import FEATURES_V2, CAT_FEATURES_V2


def get_training_data(feat, race_date, is_turf, method, period_years, n_samples, use_weight):
    """学習データを取得する"""
    base = feat[
        (feat['date'] < race_date) &
        (feat['is_turf'] == is_turf) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date')

    if method == 'fixed_years':
        cutoff = race_date - pd.DateOffset(years=period_years)
        tr = base[base['date'] >= cutoff].copy()
    else:  # tail_n
        tr = base.tail(n_samples).copy()

    if len(tr) == 0:
        return None, None

    if use_weight:
        # 時間減衰: 直近ほど高確率でサンプリング
        days_ago = (race_date - tr['date']).dt.days.values.astype(float)
        max_days = days_ago.max() if days_ago.max() > 0 else 1
        # 指数減衰: 半減期 = max_days/3
        weights = np.exp(-0.693 * days_ago / (max_days / 3))
        weights /= weights.sum()
        # 重み付きサンプリング (元のサイズと同じ件数)
        n = len(tr)
        idx = np.random.choice(len(tr), size=n, replace=True, p=weights)
        tr = tr.iloc[idx].copy()

    return tr, len(tr)


def evaluate_race(feat, race_rows, use_f, use_cf, method, period_years, n_samples, use_weight, ep=80):
    """1レースを評価する"""
    race_date = race_rows['date'].iloc[0]
    is_turf = 1

    tr, tr_size = get_training_data(feat, race_date, is_turf, method, period_years, n_samples, use_weight)
    if tr is None or len(tr) < 1000:
        return None

    pred = PredictorV2(use_f, use_cf)
    pred.train(tr, ep=ep, lr=0.003, seed=42)
    nn_preds = pred.predict(race_rows)

    # 予測順位 (mu小さい方が上位)
    nn_preds_sorted = nn_preds.sort_values('mu')
    pred_top5 = set(nn_preds_sorted['umaban'].head(5).values)

    # 実着順
    actual = race_rows[['umaban', 'finish']].sort_values('finish')
    actual_top3 = set(actual['umaban'].head(3).values)
    actual_winner = actual['umaban'].iloc[0]

    # 指標
    top3_overlap = len(pred_top5 & actual_top3)
    winner_in_top5 = 1 if actual_winner in pred_top5 else 0

    # Spearman相関
    merged = nn_preds[['umaban', 'mu']].merge(race_rows[['umaban', 'finish']], on='umaban')
    if len(merged) >= 3:
        corr, _ = spearmanr(merged['mu'], merged['finish'])
    else:
        corr = 0

    return {
        'top3_overlap': top3_overlap,
        'winner_in_top5': winner_in_top5,
        'rank_corr': corr,
        'tr_size': tr_size,
    }


def main():
    t0 = time.time()
    np.random.seed(42)

    print('Loading feature cache...')
    feat = pd.read_pickle('data/features_v2_cache.pkl')
    use_f = [f for f in FEATURES_V2 if f in feat.columns]
    use_cf = [f for f in CAT_FEATURES_V2 if f in feat.columns]

    # テスト対象: 2025年後半〜2026年の重賞・リステッド級レース
    # 芝、8頭以上、クラス上位のレースを抽出
    test_races = feat[
        (feat['date'] >= '2025-10-01') &
        (feat['is_turf'] == 1) &
        (feat['heads'] >= 10) &
        (feat['prev_race_class'] >= 4)  # OP以上
    ]['race_id'].unique()

    # レース数が多すぎる場合はサンプリング
    if len(test_races) > 30:
        np.random.shuffle(test_races)
        test_races = test_races[:30]

    print(f'Test races: {len(test_races)}')

    # === 比較する設定 ===
    configs = []

    # 1. tail(N) 方式
    for n in [20000, 30000, 40000, 50000]:
        for w in [False, True]:
            label = f'tail_{n//1000}k' + ('_weighted' if w else '')
            configs.append({
                'label': label,
                'method': 'tail_n',
                'period_years': None,
                'n_samples': n,
                'use_weight': w,
            })

    # 2. 固定年数方式
    for y in [2, 3, 5]:
        for w in [False, True]:
            label = f'fixed_{y}y' + ('_weighted' if w else '')
            configs.append({
                'label': label,
                'method': 'fixed_years',
                'period_years': y,
                'n_samples': None,
                'use_weight': w,
            })

    print(f'Configs: {len(configs)}')
    print()

    # === バックテスト実行 ===
    results = {c['label']: [] for c in configs}

    for ri, race_id in enumerate(test_races):
        race_rows = feat[feat['race_id'] == race_id].copy()
        if len(race_rows) < 8:
            continue

        race_date = race_rows['date'].iloc[0]
        print(f'[{ri+1}/{len(test_races)}] {race_id} ({race_date.date()}, {len(race_rows)}頭)...')

        for cfg in configs:
            try:
                res = evaluate_race(
                    feat, race_rows, use_f, use_cf,
                    cfg['method'], cfg['period_years'], cfg['n_samples'], cfg['use_weight']
                )
                if res:
                    results[cfg['label']].append(res)
            except Exception as e:
                print(f'  ERROR {cfg["label"]}: {e}')

    # === 集計 ===
    print(f'\n{"="*80}')
    print(f'結果集計 ({len(test_races)} races)')
    print(f'{"="*80}')
    print(f'\n{"設定":<25s} {"races":>5s} {"top3重複":>8s} {"1着的中":>8s} {"順位相関":>8s} {"学習件数":>8s}')
    print(f'{"-"*63}')

    summary = []
    for cfg in configs:
        r = results[cfg['label']]
        if not r:
            continue
        n = len(r)
        avg_overlap = np.mean([x['top3_overlap'] for x in r])
        avg_winner = np.mean([x['winner_in_top5'] for x in r])
        avg_corr = np.mean([x['rank_corr'] for x in r])
        avg_tr = np.mean([x['tr_size'] for x in r])

        print(f'{cfg["label"]:<25s} {n:5d} {avg_overlap:8.2f} {avg_winner:8.1%} {avg_corr:8.3f} {avg_tr:8.0f}')
        summary.append({
            'label': cfg['label'],
            'races': n,
            'top3_overlap': avg_overlap,
            'winner_in_top5': avg_winner,
            'rank_corr': avg_corr,
            'tr_size': avg_tr,
        })

    # ベスト表示
    if summary:
        best_overlap = max(summary, key=lambda x: x['top3_overlap'])
        best_winner = max(summary, key=lambda x: x['winner_in_top5'])
        best_corr = max(summary, key=lambda x: x['rank_corr'])

        print(f'\n--- ベスト ---')
        print(f'top3重複ベスト: {best_overlap["label"]} ({best_overlap["top3_overlap"]:.2f})')
        print(f'1着的中ベスト:  {best_winner["label"]} ({best_winner["winner_in_top5"]:.1%})')
        print(f'順位相関ベスト: {best_corr["label"]} ({best_corr["rank_corr"]:.3f})')

    elapsed = time.time() - t0
    print(f'\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)')


if __name__ == '__main__':
    main()
