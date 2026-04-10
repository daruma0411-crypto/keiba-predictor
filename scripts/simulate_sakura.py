"""
桜花賞10年ウォークフォワードシミュレーション
各年の桜花賞を、その年より前のデータのみで学習して予測する
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
from src.binary_parser import load_all_data, filter_open_class
from src.features import build_all_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.model import RacePredictor, monte_carlo_simulation, explain_prediction


def find_sakura_sho(df_all):
    """桜花賞のrace_idを年ごとに取得"""
    sakura = df_all[df_all['race_name'].str.contains('桜花賞', na=False)]
    sakura = sakura[sakura['grade_cd'] == 'A']  # G1のみ
    sakura_by_year = {}
    for race_id in sakura['race_id'].unique():
        year = int(race_id[:4])
        sakura_by_year[year] = race_id
    return sakura_by_year


def run_simulation():
    print("=" * 60)
    print("桜花賞 10年ウォークフォワードシミュレーション")
    print("=" * 60)

    # データ読み込み
    print("\n[1/3] Loading all venue data (2015-2025)...")
    df_all = load_all_data(years=range(2014, 2026))
    print(f"  Total: {len(df_all):,} records")

    # 桜花賞のレースIDを特定
    sakura_by_year = find_sakura_sho(df_all)
    print(f"\n  桜花賞が見つかった年: {sorted(sakura_by_year.keys())}")

    # 結果格納
    all_results = []

    for year in sorted(sakura_by_year.keys()):
        if year < 2016:
            continue  # 2015以前は学習データが少なすぎる

        race_id = sakura_by_year[year]

        print(f"\n{'='*60}")
        print(f"  {year}年 桜花賞 (race_id: {race_id})")
        print(f"{'='*60}")

        # このレースの出走馬情報
        race_horses = df_all[df_all['race_id'] == race_id]
        print(f"  出走頭数: {len(race_horses)}")

        # 学習データ: この年の桜花賞より前の全データ
        race_date = race_horses['date'].iloc[0]
        train_data = df_all[df_all['date'] < race_date].copy()
        print(f"  学習データ: {len(train_data):,} records (~ {race_date.strftime('%Y/%m/%d')})")

        # 予測対象: この桜花賞の出走馬の特徴量を計算
        # 出走馬の過去走を学習データから取得
        horse_ids = set(race_horses['ketto_num'].unique())
        horse_history = train_data[train_data['ketto_num'].isin(horse_ids)].copy()

        # この桜花賞のレコードも追加（特徴量計算のために）
        predict_data = pd.concat([horse_history, race_horses], ignore_index=True)

        # 特徴量構築
        print("  Building features...")
        features = build_all_features(predict_data)

        # 桜花賞のレコードのみ抽出
        race_features = features[features['race_id'] == race_id].copy()
        if len(race_features) == 0:
            print("  WARNING: No features for this race (all horses are first-time?)")
            continue

        print(f"  特徴量あり: {len(race_features)} horses")

        # 学習データの特徴量（芝OP以上で学習）
        train_open = filter_open_class(train_data)
        train_turf = train_open[train_open['surface'] == '芝'].copy()

        # 学習データが少なすぎる場合は全クラスも使う
        if len(train_turf) < 5000:
            train_for_model = train_data[train_data['surface'] == '芝'].copy()
        else:
            train_for_model = train_turf

        train_features = build_all_features(train_for_model)
        print(f"  学習用特徴量: {len(train_features):,} samples")

        # モデル学習
        print("  Training model...")
        predictor = RacePredictor(
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        predictor.train(train_features, epochs=80, lr=0.002)

        # 予測
        print("  Predicting...")
        predictions = predictor.predict(race_features)

        # MCシミュレーション (10万回)
        print("  Running Monte Carlo (100,000 simulations)...")
        mc_results = monte_carlo_simulation(predictions, n_simulations=100000)

        # 結果表示
        print(f"\n  ★ {year}年 桜花賞 予測 ★")
        print(f"  {'馬番':>4s} {'馬名':20s} {'勝率':>6s} {'複勝率':>6s} {'期待着順':>8s} {'EV':>6s}")
        print(f"  {'-'*56}")
        for _, row in mc_results.head(5).iterrows():
            ev = f"{row['ev_win']:.2f}" if pd.notna(row.get('ev_win')) else "N/A"
            print(f"  {int(row['umaban']):4d} {row['horse_name']:20s} "
                  f"{row['win_prob']:6.1%} {row['top3_prob']:6.1%} "
                  f"{row['expected_rank']:8.1f} {ev:>6s}")

        # 予測根拠
        explanations = explain_prediction(
            predictions, mc_results, race_features, NUMERIC_FEATURES)
        print(f"\n  予測根拠 (上位3頭):")
        for horse, exp in list(explanations.items())[:3]:
            print(f"  [{exp['umaban']}] {horse}")
            print(f"    勝率:{exp['win_prob']} 複勝率:{exp['top3_prob']}")
            if exp['strengths']:
                print(f"    強み: {'; '.join(exp['strengths'][:3])}")
            if exp['weaknesses']:
                print(f"    弱み: {'; '.join(exp['weaknesses'][:2])}")

        # 実際の結果
        actual = race_horses.sort_values('kakutei_jyuni').head(3)
        print(f"\n  ★ 実際の結果 ★")
        for _, row in actual.iterrows():
            print(f"  {int(row['kakutei_jyuni'])}着: [{int(row['umaban'])}] {row['horse_name']} "
                  f"(人気:{int(row['ninki'])} オッズ:{row['odds']})")

        # 的中判定
        pred_top1 = mc_results.iloc[0]['umaban']
        pred_top3 = set(mc_results.head(3)['umaban'])
        actual_top1 = actual.iloc[0]['umaban']
        actual_top3 = set(actual['umaban'])

        win_hit = pred_top1 == actual_top1
        top3_hit = len(pred_top3 & actual_top3)

        result = {
            'year': year,
            'win_hit': win_hit,
            'top3_overlap': top3_hit,
            'pred_top3': pred_top3,
            'actual_top3': actual_top3,
        }
        all_results.append(result)

        print(f"\n  判定: 単勝{'◎的中' if win_hit else '×'} "
              f"| 3着以内 {top3_hit}/3頭一致")

    # 全年サマリー
    print(f"\n{'='*60}")
    print(f"  10年シミュレーション結果サマリー")
    print(f"{'='*60}")
    win_total = sum(r['win_hit'] for r in all_results)
    print(f"  単勝的中: {win_total}/{len(all_results)} ({win_total/len(all_results)*100:.0f}%)")
    avg_overlap = np.mean([r['top3_overlap'] for r in all_results])
    print(f"  3着以内一致: 平均 {avg_overlap:.1f}/3頭")

    for r in all_results:
        mark = "◎" if r['win_hit'] else "×"
        print(f"  {r['year']}: {mark} (3着以内 {r['top3_overlap']}/3一致)")


if __name__ == '__main__':
    run_simulation()
