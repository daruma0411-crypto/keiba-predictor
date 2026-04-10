"""
桜花賞ウォークフォワードシミュレーション（高速版）
事前計算済み特徴量を使用
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.model import RacePredictor, monte_carlo_simulation, explain_prediction
from src.binary_parser import load_all_data


def run():
    print("=" * 60)
    print("桜花賞 ウォークフォワードシミュレーション")
    print("=" * 60)

    # 事前計算済み特徴量を読み込み
    print("\nLoading pre-computed features...")
    features_all = pd.read_pickle('data/features_all.pkl')
    print(f"  Total features: {len(features_all):,} samples")

    # 元データも読み込み（桜花賞の特定とactual結果用）
    print("Loading raw data...")
    df_all = load_all_data(years=range(2015, 2026))

    # 桜花賞のrace_idを特定
    sakura_all = df_all[
        (df_all['race_name'].str.contains('桜花賞', na=False)) &
        (df_all['grade_cd'] == 'A')
    ]
    sakura_races = {}
    for race_id in sakura_all['race_id'].unique():
        year = int(race_id[:4])
        sakura_races[year] = race_id

    print(f"  桜花賞: {sorted(sakura_races.keys())}")

    results_summary = []

    for year in range(2016, 2026):
        if year not in sakura_races:
            continue

        race_id = sakura_races[year]
        race_data = df_all[df_all['race_id'] == race_id]
        race_date = race_data['date'].iloc[0]

        print(f"\n{'='*60}")
        print(f"  {year}年 桜花賞 ({race_date.strftime('%Y/%m/%d')})")
        print(f"  出走: {len(race_data)}頭")
        print(f"{'='*60}")

        # 学習データ: この桜花賞より前の芝データ
        train_mask = (
            (features_all['date'] < race_date) &
            (features_all['is_turf'] == 1) &
            (features_all['past_count'] > 0)
        )
        train_feat = features_all[train_mask].copy()

        # 予測対象: この桜花賞の出走馬
        race_feat = features_all[features_all['race_id'] == race_id].copy()

        if len(race_feat) == 0:
            print("  ※ 特徴量なし（スキップ）")
            continue

        print(f"  学習: {len(train_feat):,} samples / 予測: {len(race_feat)} horses")

        # モデル学習
        predictor = RacePredictor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        predictor.train(train_feat, epochs=80, lr=0.002)

        # 予測 + MC
        predictions = predictor.predict(race_feat)
        mc = monte_carlo_simulation(predictions, n_simulations=100000)

        # 予測結果
        print(f"\n  ★ 予測 ★")
        print(f"  {'番':>3s} {'馬名':18s} {'勝率':>6s} {'複勝':>6s} {'期待順':>6s} {'EV':>6s}")
        print(f"  {'-'*52}")
        for _, row in mc.head(5).iterrows():
            ev = f"{row['ev_win']:.2f}" if pd.notna(row.get('ev_win')) else "-"
            print(f"  {int(row['umaban']):3d} {row['horse_name']:18s} "
                  f"{row['win_prob']:6.1%} {row['top3_prob']:6.1%} "
                  f"{row['expected_rank']:6.1f} {ev:>6s}")

        # 予測根拠 (上位2頭)
        explanations = explain_prediction(
            predictions, mc, race_feat, NUMERIC_FEATURES)
        for horse, exp in list(explanations.items())[:2]:
            print(f"\n  [{exp['umaban']}] {horse}: ", end="")
            if exp['strengths']:
                print(f"強み={exp['strengths'][0]}", end="")
            if exp['weaknesses']:
                print(f" 弱み={exp['weaknesses'][0]}", end="")
            print()

        # 実際の結果
        actual = race_data.sort_values('kakutei_jyuni').head(3)
        print(f"\n  ★ 実際 ★")
        for _, row in actual.iterrows():
            print(f"  {int(row['kakutei_jyuni'])}着 [{int(row['umaban'])}] "
                  f"{row['horse_name']} (人気:{int(row['ninki'])} odds:{row['odds']})")

        # 的中判定
        pred_top1_uma = int(mc.iloc[0]['umaban'])
        pred_top3_uma = set(mc.head(3)['umaban'].astype(int))
        actual_top1_uma = int(actual.iloc[0]['umaban'])
        actual_top3_uma = set(actual['umaban'].astype(int))

        win_hit = pred_top1_uma == actual_top1_uma
        top3_overlap = len(pred_top3_uma & actual_top3_uma)

        # 予測上位5頭に1着馬が含まれているか
        pred_top5_uma = set(mc.head(5)['umaban'].astype(int))
        top5_has_winner = actual_top1_uma in pred_top5_uma

        results_summary.append({
            'year': year,
            'win_hit': win_hit,
            'top3_overlap': top3_overlap,
            'top5_has_winner': top5_has_winner,
            'pred_1st': mc.iloc[0]['horse_name'],
            'actual_1st': actual.iloc[0]['horse_name'],
        })

        mark = "◎" if win_hit else ("○" if top5_has_winner else "×")
        print(f"\n  判定: {mark} 単勝{'的中!' if win_hit else '×'} "
              f"| 上位3頭{top3_overlap}/3一致 "
              f"| 上位5頭に1着{'含む' if top5_has_winner else '含まず'}")

    # サマリー
    print(f"\n{'='*60}")
    print(f"  サマリー")
    print(f"{'='*60}")
    n = len(results_summary)
    win_total = sum(r['win_hit'] for r in results_summary)
    top5_total = sum(r['top5_has_winner'] for r in results_summary)
    avg_overlap = np.mean([r['top3_overlap'] for r in results_summary])

    print(f"  単勝的中:      {win_total}/{n} ({win_total/n*100:.0f}%)")
    print(f"  上位5に1着含む: {top5_total}/{n} ({top5_total/n*100:.0f}%)")
    print(f"  3着以内一致:    平均 {avg_overlap:.1f}/3")

    print()
    for r in results_summary:
        mark = "◎" if r['win_hit'] else ("○" if r['top5_has_winner'] else "×")
        print(f"  {r['year']}: {mark}  予測1位:{r['pred_1st']:15s} 実際1着:{r['actual_1st']}")


if __name__ == '__main__':
    run()
