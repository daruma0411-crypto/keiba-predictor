"""
券種別ROIバックテスト
NNモデルのμ上位5頭を選抜し、各券種の的中率・回収率を過去レースで検証する。

対象券種: 単勝, 馬連, 馬単, ワイド, 三連複
払戻金はJRA実績ベースの推定式を使用（実オッズからの近似）。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from itertools import combinations, permutations
from src.predictor import Predictor, CAT_FEATURES

# キャッシュに存在する特徴量のみ使用
FEATURES_V9_COMPAT = [
    'wakuban', 'futan', 'bataijyu', 'zogen_sa', 'heads',
    'past_count', 'ema_time_zscore', 'ema_finish', 'ema_time_diff',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'jockey_dist_win_rate', 'jockey_dist_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'trainer_dist_win_rate', 'trainer_dist_top3_rate',
    'avg_jyuni_3c', 'avg_jyuni_4c',
    'prev_race_class', 'log_prize_money',
    'ema_agari', 'long_stretch_avg', 'prev_dist_diff',
]

# ============================================================
# 設定
# ============================================================
TOP_N = 5          # 選抜頭数
TEST_START = '2024-01-01'
TEST_END   = '2026-04-05'
MIN_HEADS  = 8     # 最低出走頭数
TURF_ONLY  = True  # 芝のみ

# ============================================================
# データ読み込み
# ============================================================
print("Loading data...")
df = pd.read_pickle('data/features_v9b_2026.pkl')
df['date'] = pd.to_datetime(df['date'])

if TURF_ONLY:
    df = df[df['is_turf'] == 1]

# 学習/テスト分割
train_df = df[df['date'] < TEST_START].copy()
test_df  = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

# 出走頭数フィルタ
race_heads = test_df.groupby('race_id')['umaban'].count()
valid_races = race_heads[race_heads >= MIN_HEADS].index
test_df = test_df[test_df['race_id'].isin(valid_races)]

print(f"Train: {len(train_df):,} rows")
print(f"Test : {len(test_df):,} rows, {test_df['race_id'].nunique():,} races")
print(f"Period: {TEST_START} ~ {TEST_END}")

# ============================================================
# モデル学習（直近20,000件）
# ============================================================
print("\nTraining model...")
train_recent = train_df.sort_values('date').tail(20000)
pred = Predictor(numeric_features=FEATURES_V9_COMPAT)
pred.train(train_recent, ep=50, lr=0.003, seed=42)

# ============================================================
# レースごとに予測 → TOP5選抜 → 各券種判定
# ============================================================
print("\nRunning backtest...")

results = {
    'tansho':     {'bets': 0, 'hits': 0, 'payout': 0.0},  # 単勝: TOP5各100円 = 500円/R
    'umaren':     {'bets': 0, 'hits': 0, 'payout': 0.0},  # 馬連BOX: C(5,2)=10点
    'umatan':     {'bets': 0, 'hits': 0, 'payout': 0.0},  # 馬単BOX: P(5,2)=20点
    'wide':       {'bets': 0, 'hits': 0, 'payout': 0.0},  # ワイドBOX: C(5,2)=10点
    'sanrenpuku': {'bets': 0, 'hits': 0, 'payout': 0.0},  # 三連複BOX: C(5,3)=10点
}

race_ids = test_df['race_id'].unique()
n_races = len(race_ids)

for i, race_id in enumerate(race_ids):
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{n_races} races processed...")

    race = test_df[test_df['race_id'] == race_id].copy()
    if len(race) < MIN_HEADS:
        continue

    # NN予測
    nn_preds = pred.predict(race)
    race = race.copy()
    race['mu'] = nn_preds['mu'].values

    # μ上位5頭選抜（μが小さい＝着順が良い予測）
    top5 = race.nsmallest(TOP_N, 'mu')
    top5_umaban = set(top5['umaban'].values)

    # 実着順
    race_sorted = race.sort_values('finish')
    win_umaban   = race_sorted.iloc[0]['umaban']  # 1着
    place2_umaban = race_sorted.iloc[1]['umaban']  # 2着
    place3_umaban = race_sorted.iloc[2]['umaban']  # 3着

    win_odds = race_sorted.iloc[0]['odds']  # 1着馬のオッズ（≒単勝オッズ）

    top3_umaban = {win_umaban, place2_umaban, place3_umaban}

    # --- 単勝 ---
    # TOP5の各馬に100円ずつ（5点買い）
    results['tansho']['bets'] += TOP_N
    for _, h in top5.iterrows():
        if h['umaban'] == win_umaban:
            results['tansho']['hits'] += 1
            results['tansho']['payout'] += win_odds * 100

    # --- 馬連 BOX ---
    # 1着-2着の馬番組合せ（順不同）
    umaren_hit = {win_umaban, place2_umaban}
    combos = list(combinations(sorted(top5_umaban), 2))
    results['umaren']['bets'] += len(combos)
    for c in combos:
        if set(c) == umaren_hit:
            # 馬連配当推定: 単勝オッズの積の平方根 × 補正
            odds1 = race[race['umaban'] == win_umaban]['odds'].values[0]
            odds2 = race[race['umaban'] == place2_umaban]['odds'].values[0]
            umaren_payout = np.sqrt(odds1 * odds2) * 1.8 * 100
            results['umaren']['hits'] += 1
            results['umaren']['payout'] += umaren_payout

    # --- 馬単 BOX ---
    # 1着→2着の順番も一致
    perms = list(permutations(sorted(top5_umaban), 2))
    results['umatan']['bets'] += len(perms)
    for p in perms:
        if p[0] == win_umaban and p[1] == place2_umaban:
            odds1 = race[race['umaban'] == win_umaban]['odds'].values[0]
            odds2 = race[race['umaban'] == place2_umaban]['odds'].values[0]
            umatan_payout = np.sqrt(odds1 * odds2) * 3.0 * 100
            results['umatan']['hits'] += 1
            results['umatan']['payout'] += umatan_payout

    # --- ワイド BOX ---
    # 3着内の2頭の組合せ（複数的中あり）
    results['wide']['bets'] += len(combos)
    for c in combos:
        if set(c).issubset(top3_umaban):
            odds_a = race[race['umaban'] == c[0]]['odds'].values[0]
            odds_b = race[race['umaban'] == c[1]]['odds'].values[0]
            wide_payout = np.sqrt(odds_a * odds_b) * 0.7 * 100
            results['wide']['hits'] += 1
            results['wide']['payout'] += wide_payout

    # --- 三連複 BOX ---
    # 3着内3頭の組合せ（順不同）
    combos3 = list(combinations(sorted(top5_umaban), 3))
    results['sanrenpuku']['bets'] += len(combos3)
    for c in combos3:
        if set(c) == top3_umaban:
            odds1 = race[race['umaban'] == win_umaban]['odds'].values[0]
            odds2 = race[race['umaban'] == place2_umaban]['odds'].values[0]
            odds3 = race[race['umaban'] == place3_umaban]['odds'].values[0]
            sanren_payout = (odds1 * odds2 * odds3) ** (1/3) * 5.0 * 100
            results['sanrenpuku']['hits'] += 1
            results['sanrenpuku']['payout'] += sanren_payout


# ============================================================
# 結果表示
# ============================================================
print("\n" + "=" * 70)
print(f"NN TOP{TOP_N} 券種別ROIバックテスト")
print(f"期間: {TEST_START} ~ {TEST_END} (芝{'のみ' if TURF_ONLY else '+ダ'})")
print(f"対象: {n_races:,} レース (出走{MIN_HEADS}頭以上)")
print("=" * 70)
print(f"{'券種':<10} {'買い点数':>10} {'的中数':>8} {'的中率':>8} {'投資額':>12} {'回収額':>12} {'ROI':>8}")
print("-" * 70)

names = {
    'tansho': '単勝',
    'umaren': '馬連',
    'umatan': '馬単',
    'wide': 'ワイド',
    'sanrenpuku': '三連複',
}

for key in ['tansho', 'wide', 'umaren', 'umatan', 'sanrenpuku']:
    r = results[key]
    invest = r['bets'] * 100
    hit_rate = r['hits'] / r['bets'] * 100 if r['bets'] > 0 else 0
    roi = r['payout'] / invest * 100 if invest > 0 else 0
    print(f"{names[key]:<10} {r['bets']:>10,} {r['hits']:>8,} {hit_rate:>7.1f}% {invest:>11,}円 {r['payout']:>11,.0f}円 {roi:>7.1f}%")

print("-" * 70)
print("\n※配当は実オッズからの推定値（実際の払戻金とは異なります）")
print("  単勝: 単勝オッズ × 100")
print("  馬連: √(odds1×odds2) × 1.8 × 100")
print("  馬単: √(odds1×odds2) × 3.0 × 100")
print("  ワイド: √(odds_a×odds_b) × 0.7 × 100")
print("  三連複: ∛(odds1×odds2×odds3) × 5.0 × 100")
