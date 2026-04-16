"""
桜花賞バックテスト V2 (CSVベース)
V2予測エンジン: CSV特徴量(上がり3F/ペース/馬場状態) + 4層NN + ListNet損失
- データ: TARGET CSV (csv_parser.load_csv_data)
- 特徴量: features_v2.build_features_v2 (FEATURES_V2 + CAT_FEATURES_V2)
- モデル: predictor_v2.PredictorV2 (4層NN, Gaussian NLL + ListNet)
- QMC: qmc_courses.qmc_sim (コース別プロファイル)
"""
import sys
import io
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np

np.random.seed(42)


def run(use_cache=False):
    from src.csv_parser import load_csv_data
    from src.features_v2 import build_features_v2, FEATURES_V2, CAT_FEATURES_V2
    from src.predictor_v2 import PredictorV2
    from src.qmc_courses import qmc_sim

    # === 1. データ読み込み ===
    print('='*60)
    print('  V2 桜花賞バックテスト (CSVベース)')
    print('='*60)

    cache_path = 'data/features_v2_cache.pkl'

    if use_cache and os.path.exists(cache_path):
        print(f'Loading cached V2 features: {cache_path}')
        feat = pd.read_pickle(cache_path)
    else:
        print('Loading CSV data...')
        df = load_csv_data('C:/TXT/10ne_deta', surface_filter='芝')

        print('Building V2 features...')
        feat = build_features_v2(df)

        if use_cache:
            feat.to_pickle(cache_path)
            print(f'Saved cache: {cache_path}')

    print(f'Total features: {len(feat):,} rows')
    print(f'Period: {feat["date"].min()} ~ {feat["date"].max()}')

    # === 2. 桜花賞レースの特定 (2016-2025) ===
    sakura_mask = feat['class_name'].str.contains('桜花賞', na=False)
    # grade_cd == 'A' (G1) またはclass_nameにG1を含む
    g1_mask = (feat['grade_cd'] == 'A') | feat['class_name'].str.contains('G1', na=False)
    sakura_all = feat[sakura_mask & g1_mask]

    # 年ごとのrace_idマッピング
    sakura_ids = {}
    for rid in sakura_all['race_id'].unique():
        row = sakura_all[sakura_all['race_id'] == rid].iloc[0]
        year = row['date'].year
        sakura_ids[year] = rid

    print(f'\nSakura Sho races found: {sorted(sakura_ids.keys())}')

    # 使用する特徴量（データに存在するもののみ）
    use_f = [f for f in FEATURES_V2 if f in feat.columns]
    use_cat = [c for c in CAT_FEATURES_V2 if c in feat.columns]
    print(f'Numeric features: {len(use_f)}/{len(FEATURES_V2)}')
    print(f'Categorical features: {len(use_cat)}/{len(CAT_FEATURES_V2)}')

    # === 3. Walk-forward バックテスト ===
    results = []
    for year in range(2016, 2026):
        if year not in sakura_ids:
            print(f'\n{year}年: 桜花賞データなし — skip')
            continue

        rid = sakura_ids[year]
        rf = feat[feat['race_id'] == rid].copy()
        if len(rf) == 0:
            print(f'\n{year}年: 特徴量なし — skip')
            continue

        rdate = rf['date'].iloc[0]

        # 学習データ: そのレースより前の芝データ (tail 40,000)
        tr = feat[
            (feat['date'] < rdate) &
            (feat['is_turf'] == 1) &
            (feat['past_count'] > 0) &
            (feat['finish'] > 0)
        ].sort_values('date').tail(40000).copy()

        print(f'\n{year}年桜花賞 ({rdate.strftime("%Y-%m-%d")}):')
        print(f'  学習データ: {len(tr):,} rows')
        print(f'  出走頭数: {len(rf)} heads')

        # --- Layer 1: PredictorV2 ---
        pred = PredictorV2(numeric_features=use_f, cat_features=use_cat)
        pred.train(tr, ep=80, lr=0.003, seed=42, alpha_rank=0.3)
        ps = pred.predict(rf)

        # --- Layer 2: QMC (コース別) ---
        mc = qmc_sim(ps, race_features=rf, course='hanshin_turf_1600_outer', n=100000)

        # --- 結果評価 ---
        # 実際の着順 (CSVでは'finish'カラム)
        act = rf.sort_values('finish').head(3)
        a3 = set(act['umaban'].astype(int))
        a1 = int(act.iloc[0]['umaban'])

        t5 = mc.head(5)
        t5u = set(t5['umaban'].astype(int))
        ov = len(t5u & a3)
        w = int(t5.iloc[0]['umaban']) == a1

        t5f = []
        print(f'  予測TOP5:')
        for rk, (_, r) in enumerate(t5.iterrows(), 1):
            u = int(r['umaban'])
            a = rf[rf['umaban'] == u]
            af = int(a.iloc[0]['finish']) if len(a) > 0 else 18
            t5f.append(af)
            h = '*' if u in a3 else ' '
            odds_val = r['odds'] if pd.notna(r.get('odds')) else 0.0
            print(f'    {h}{rk}位 [{u:2d}] {r["horse_name"]:16s} '
                  f'勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} '
                  f'odds={odds_val:.1f} (実際{af}着)')

        print(f'  実際TOP3:')
        for _, r in act.iterrows():
            ninki_val = int(r['ninki']) if pd.notna(r.get('ninki')) else 0
            odds_val = r['odds'] if pd.notna(r.get('odds')) else 0.0
            print(f'    {int(r["finish"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]:16s} '
                  f'({ninki_val}人気 odds:{odds_val})')

        avg = np.mean(t5f)
        ana = any(
            r['odds'] >= 10 and int(r['umaban']) in a3
            for _, r in t5.iterrows()
            if pd.notna(r.get('odds'))
        )
        results.append({
            'year': year,
            'win': w,
            'overlap': ov,
            't5': a1 in t5u,
            'avg': avg,
            'ana': ana,
            'p1f': t5f[0],
        })
        sys.stdout.flush()

    # === 4. KPIレポート ===
    if not results:
        print('\nNo results to report.')
        return

    n = len(results)
    print(f'\n{"="*60}')
    print(f'  V2 KPIレポート (CSVベース)')
    print(f'{"="*60}')

    wn = sum(r['win'] for r in results)
    wr = wn / n * 100
    t5ok = sum(1 for r in results if r['avg'] <= 9)
    an = sum(r['ana'] for r in results)
    o2 = sum(1 for r in results if r['overlap'] >= 2)
    ovr = o2 / n * 100

    target_win = 30 <= wr <= 40
    target_avg = t5ok == n
    target_ana = an >= 1
    target_ov = ovr >= 80

    print(f'\n  ■ KPI 1: 1着的中率: {wn}/{n} ({wr:.0f}%)'
          f' → {"OK" if target_win else "NG"} (target 30-40%)')
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5ok}/{n}'
          f' → {"OK" if target_avg else "NG"}')
    print(f'  ■ KPI 3: 穴馬検知: {an}/{n}年'
          f' → {"OK" if target_ana else "NG"}')
    print(f'  ■ KPI 4: 馬券内占有率: {o2}/{n} ({ovr:.0f}%)'
          f' → {"OK" if target_ov else "NG"} (target >=80%)')

    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        a = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["p1f"]}着 '
              f'3着内{r["overlap"]}/3 TOP5平均{r["avg"]:.1f}着 {a}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 桜花賞バックテスト (CSVベース)')
    parser.add_argument('--cache', action='store_true',
                        help='Save/load features_v2 cache as pkl')
    args = parser.parse_args()
    run(use_cache=args.cache)
