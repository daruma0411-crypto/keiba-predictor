"""
汎用レース予測スクリプト v2 (Layer 1→2→3 一気通貫)

使い方:
    py -3.13 scripts/predict.py C:/TXT/04112.csv
    py -3.13 scripts/predict.py C:/TXT/04112.csv --course fukushima_turf_1200
    py -3.13 scripts/predict.py C:/TXT/04112.csv --no-debate

v2エンジン: PredictorV2(4層NN+ListNet) + 40特徴量 + 40k学習データ + 80epoch
コースプロファイルが未定義なら自動でデフォルト係数を使用。
"""
import sys
import os
import warnings
import argparse
import time

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch

np.random.seed(42)

from src.predictor_v2 import PredictorV2
from src.features_v2 import FEATURES_V2, CAT_FEATURES_V2
from src.qmc_courses import qmc_sim, COURSE_PROFILES, list_courses
from src.prompts import build_prompt
from src.entry_parser import parse_entry_csv, build_race_features


def select_top5(mc, n_pop=5, cutoff=5):
    """分割選抜: cutoff番人気以内からQMC上位n_pop頭 + 残りをQMC順で補充して5頭"""
    if 'ninki' not in mc.columns:
        mc = mc.copy()
        mc['ninki'] = mc['odds'].rank(method='first')
    pop = mc[mc['ninki'] <= cutoff].head(n_pop)
    n_disc = 5 - len(pop)
    if n_disc > 0:
        disc = mc[~mc.index.isin(pop.index)].head(n_disc)
        selected = pd.concat([pop, disc])
    else:
        selected = pop
    return selected.sort_values('expected_rank').head(5)


def find_course(race_info, course_override=None):
    """コースプロファイルを決定する"""
    if course_override:
        if course_override in COURSE_PROFILES:
            return course_override
        print(f'  WARNING: {course_override} not in COURSE_PROFILES, using default')

    # 自動マッチ
    key = race_info['course_key']
    if key in COURSE_PROFILES:
        return key

    # 部分一致
    for k in COURSE_PROFILES:
        if race_info['venue_key'] in k and str(race_info['distance']) in k:
            return k

    return None


def main():
    parser = argparse.ArgumentParser(description='汎用レース予測 v2 (PredictorV2 + QMC + 議長ディベート)')
    parser.add_argument('csv_path', help='出馬表CSVパス')
    parser.add_argument('--course', default=None, help='コースプロファイルキー (省略時は自動検出)')
    parser.add_argument('--cache', default=None, help='特徴量キャッシュパス (省略時は最新を自動選択)')
    parser.add_argument('--sims', type=int, default=100000, help='QMCシミュレーション回数')
    parser.add_argument('--no-debate', action='store_true', help='Layer3ディベートプロンプトをスキップ')
    args = parser.parse_args()

    t_start = time.time()

    # ==================================================
    # Step 1: 出馬表CSV解析
    # ==================================================
    print(f'[Step1] Parsing entry CSV: {args.csv_path}')
    race_info, entries = parse_entry_csv(args.csv_path)
    print(f'  {race_info["date_str"]} {race_info["venue"]}{race_info["race_num"]}R '
          f'{race_info["surface"]}{race_info["distance"]}m '
          f'{race_info["class_name"]} {race_info["heads"]}頭')

    # ==================================================
    # Step 2: 特徴量キャッシュロード + レース用DF構築
    # ==================================================
    print(f'\n[Step2] Loading features & building race data...')
    if args.cache:
        cache_path = args.cache
    else:
        # 最新キャッシュを自動選択 (v2優先)
        candidates = [
            'data/features_v2_cache.pkl',
            'data/features_v9b_2026.pkl',
            'data/features_v9b_cache.pkl',
        ]
        cache_path = next((p for p in candidates if os.path.exists(p)), None)
        if cache_path is None:
            print('  ERROR: No feature cache found. Run feature build first.')
            sys.exit(1)

    feat = pd.read_pickle(cache_path)
    print(f'  Cache: {cache_path} ({len(feat):,} rows, {feat["date"].min()} ~ {feat["date"].max()})')

    race_info, rf, missing = build_race_features(args.csv_path, feat)
    print(f'  Race features: {len(rf)} horses')
    if missing:
        print(f'  ⚠ Missing from cache: {missing}')

    t1 = time.time()
    print(f'  Done in {t1 - t_start:.1f}s')

    # ==================================================
    # Step 3: NN学習 (Layer 1) — v2エンジン D構成
    # ==================================================
    print(f'\n[Step3/Layer1] Training PredictorV2 (seed=42, ep=80, lr=0.003, 40k)...')
    use_f = [f for f in FEATURES_V2 if f in feat.columns]
    use_cf = [f for f in CAT_FEATURES_V2 if f in feat.columns]

    is_turf = 1 if race_info['surface_key'] == 'turf' else 0
    tr = feat[
        (feat['date'] < race_info['date']) &
        (feat['is_turf'] == is_turf) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date').tail(40000).copy()
    print(f'  Training: {len(tr):,} samples ({tr["date"].min()} ~ {tr["date"].max()})')

    pred = PredictorV2(use_f, use_cf)
    pred.train(tr, ep=80, lr=0.003, seed=42)
    t2 = time.time()
    print(f'  Done in {t2 - t1:.1f}s')

    # ==================================================
    # Step 4: 推論 + QMC (Layer 2)
    # ==================================================
    course = find_course(race_info, args.course)
    if course:
        course_name = COURSE_PROFILES[course]['name']
        straight = COURSE_PROFILES[course]['straight']
        print(f'\n[Step4/Layer2] QMC ({course}, 直線{straight}m, {args.sims:,} sims)...')
        ps = pred.predict(rf)
        mc = qmc_sim(ps, race_features=rf, course=course, n=args.sims)
    else:
        print(f'\n[Step4/Layer2] QMC (default profile, {args.sims:,} sims)...')
        print(f'  ⚠ Course profile not found for {race_info["course_key"]}')
        print(f'  Available profiles:')
        list_courses()
        # デフォルトプロファイルで実行 (桜花賞ベース)
        ps = pred.predict(rf)
        mc = qmc_sim(ps, race_features=rf, course='hanshin_turf_1600_outer', n=args.sims)
        course_name = f'{race_info["venue"]}{race_info["surface"]}{race_info["distance"]}m (default profile)'
        straight = '?'

    t3 = time.time()
    print(f'  Done in {t3 - t2:.1f}s')

    # ==================================================
    # 結果出力
    # ==================================================
    print(f'\n{"="*80}')
    print(f'  {race_info["date_str"]} {race_info["venue"]}{race_info["race_num"]}R '
          f'{race_info["surface"]}{race_info["distance"]}m {race_info["class_name"]}')
    print(f'  v2(D) + QMC予測結果 | コース: {course_name}')
    print(f'{"="*80}')

    # 分割選抜: 5番人気以内からQMC上位5頭
    top5 = select_top5(mc, n_pop=5, cutoff=5)
    top5_umabans = set(top5['umaban'].astype(int))

    print(f'\n  {"順位":>4s} {"枠":>2s} {"番":>3s} {"馬名":18s} '
          f'{"勝率":>7s} {"複勝率":>7s} {"期待着順":>8s} {"μ":>7s} {"σ":>7s}  選抜')
    print(f'  {"-"*82}')
    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        rf_row = rf[rf['umaban'] == u]
        if len(rf_row) == 0:
            continue
        rf_row = rf_row.iloc[0]
        ps_row = ps[ps['umaban'] == u].iloc[0]
        tag = ' ★' if u in top5_umabans else ''
        print(f'  {rank:4d} {int(rf_row["wakuban"]):2d} {u:3d} {r["horse_name"]:18s} '
              f'{r["win_prob"]:7.1%} {r["top3_prob"]:7.1%} {r["expected_rank"]:8.2f} '
              f'{ps_row["mu"]:7.4f} {ps_row["sigma"]:7.4f}{tag}')

    print(f'\n  Layer1-2 選抜TOP5 (5番人気以内からQMC上位):')
    for rk, (_, r) in enumerate(top5.iterrows(), 1):
        print(f'    {rk}. [{int(r["umaban"]):2d}] {r["horse_name"]}')
    print(f'  ※穴馬発掘はLayer3ディベートで実施')

    # ==================================================
    # Layer 3: 議長プロンプト生成 + ディベート
    # ==================================================
    if not args.no_debate:
        print(f'\n[Layer3] Generating debate prompt...')
        prompt = build_prompt(
            race_name=f'{race_info["venue"]}{race_info["race_num"]}R {race_info["class_name"]}',
            course_name=f'{course_name}',
            distance=race_info['distance'],
            heads=race_info['heads'],
            date_str=race_info['date_str'],
            mc_results=mc,
            race_features=rf,
            nn_preds=ps,
            race_id=race_info['race_id'],
            top5=top5,
        )
        # プロンプトをファイルに保存
        out_name = f'output_{race_info["race_id"]}_prompt.txt'
        with open(out_name, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f'  Saved: {out_name}')
        print(f'\n--- 議長プロンプト (Layer 3) ---')
        print(prompt)

    total = time.time() - t_start
    print(f'\n  Total: {total:.1f}s')


if __name__ == '__main__':
    main()
