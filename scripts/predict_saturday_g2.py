"""
2026/4/12 土曜重賞予測
  - ニュージーランドトロフィー G2 (中山芝1600m)
  - 阪神牝馬S G2 (阪神芝1600m外回り)

アーキテクチャ: ベースNN(v9b) + コース別QMC + 議長プロンプト
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import re
import time

from src.predictor import Predictor, FEATURES_V9, CAT_FEATURES
from src.qmc_courses import qmc_sim
from src.prompts import build_prompt

np.random.seed(42)


def parse_html_entries(html_path):
    """TARGET HTML出馬表をパース → [{wakuban, umaban, horse_name_html, futan}, ...]"""
    with open(html_path, 'rb') as f:
        text = f.read().decode('cp932', errors='replace')
    entries = []
    for row in text.split('<TR>')[1:]:
        clean = re.sub(r'<[^>]+>', '\t', row)
        cells = [c.strip() for c in clean.split('\t') if c.strip()]
        if len(cells) >= 6 and cells[0].isdigit() and cells[1].isdigit():
            # TARGET HTML: [0]=枠, [1]=馬番, [2]=馬名, [3]=性齢, [4]=騎手, [5]=斤量
            try:
                futan = float(cells[5])
            except (ValueError, IndexError):
                futan = 57.0
            entries.append({
                'wakuban': int(cells[0]),
                'umaban': int(cells[1]),
                'horse_name_html': cells[2],
                'futan': futan,
            })
    return entries


def parse_csv_mapping(csv_path):
    """TARGET CSV → 馬名→ketto_num, 馬名→最新走情報"""
    df = pd.read_csv(csv_path, encoding='cp932', header=None)
    name_to_ketto = {}
    name_to_latest = {}
    for name in df[13].str.strip().unique():
        h = df[df[13].str.strip() == name]
        ketto = '20' + str(h.iloc[0][37])
        name_to_ketto[name] = ketto
        h2 = h.copy()
        h2['_date'] = (h2[0].astype(str).str.zfill(2) +
                       h2[1].astype(str).str.zfill(2) +
                       h2[2].astype(str).str.zfill(2))
        name_to_latest[name] = h2.sort_values('_date', ascending=False).iloc[0]
    return name_to_ketto, name_to_latest


def build_race_features(entries, name_to_ketto, name_to_latest,
                        feat, race_date, heads, default_futan=57.0):
    """出走馬の特徴量DataFrameを構築"""
    rows = []
    for entry in entries:
        hname = entry['horse_name_html']
        ketto = name_to_ketto.get(hname)
        if ketto is None:
            print(f'  WARNING: {hname} not found in CSV, skipping')
            continue

        horse_feat = feat[(feat['ketto_num'] == ketto) & (feat['date'] < race_date)]
        if len(horse_feat) > 0:
            row = horse_feat.sort_values('date').iloc[-1].copy()
        else:
            # キャッシュにない馬 → CSVから最小限の特徴量を構築
            print(f'  NOTE: {hname} ({ketto}) not in cache, building minimal features')
            row = pd.Series(dtype=float)
            row['ketto_num'] = ketto
            row['ema_finish'] = 5.0      # 保守的デフォルト
            row['ema_time_zscore'] = 0.0
            row['ema_agari'] = 0.0
            row['weighted_ema_finish'] = 5.0
            row['long_stretch_avg'] = np.nan
            row['same_dist_finish'] = np.nan
            row['same_surface_finish'] = np.nan
            row['win_rate'] = 0.0
            row['top3_rate'] = 0.0
            row['avg_run_style'] = 2.5
            row['avg_jyuni_3c'] = 8.0
            row['avg_jyuni_4c'] = 8.0
            row['jockey_win_rate'] = 0.08
            row['jockey_top3_rate'] = 0.25
            row['trainer_win_rate'] = 0.08
            row['trainer_top3_rate'] = 0.25
            row['prev_race_class'] = 3.0
            row['log_prize_money'] = 7.0
            row['past_count'] = 1.0
            row['is_turf'] = 1
            row['finish'] = 0
            row['prev_dist_diff'] = 0.0
            for cat in CAT_FEATURES:
                row[cat] = 'unknown'

        # 出馬表情報で上書き
        row['wakuban'] = entry['wakuban']
        row['umaban'] = entry['umaban']
        row['heads'] = heads
        row['horse_name'] = hname
        row['futan'] = entry.get('futan', default_futan)

        # 斤量: 牝馬は55kg (HTMLから判定が難しいのでCSVの性別で)
        csv_latest = name_to_latest.get(hname)
        if csv_latest is not None:
            # 前走日付→間隔日数
            try:
                prev_year = int(csv_latest[0])
                prev_month = int(csv_latest[1])
                prev_day = int(csv_latest[2])
                prev_date = pd.Timestamp(2000 + prev_year, prev_month, prev_day)
                row['interval_days'] = (race_date - prev_date).days
            except Exception:
                pass
            # 前走距離差
            try:
                prev_dist = int(csv_latest[11])
                row['prev_dist_diff'] = 1600 - prev_dist
            except Exception:
                pass
            # 馬体重 (CSV列28くらい)
            try:
                row['bataijyu'] = float(csv_latest[24])
            except Exception:
                pass
            try:
                row['zogen_sa'] = float(csv_latest[25])
            except Exception:
                pass

        row['odds'] = np.nan
        row['race_id'] = 'predict_target'
        row['date'] = race_date

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def predict_race(race_name, course_key, course_display, distance,
                 html_path, csv_path, feat, race_date, default_futan=57.0):
    """1レース分の予測を実行"""
    t0 = time.time()
    print(f'\n{"="*80}')
    print(f'  {race_name} ({course_display} {distance}m)')
    print(f'{"="*80}')

    # 1. 出馬表パース
    entries = parse_html_entries(html_path)
    name_to_ketto, name_to_latest = parse_csv_mapping(csv_path)
    heads = len(entries)
    print(f'  出走頭数: {heads}')

    # 2. 特徴量構築
    rf = build_race_features(entries, name_to_ketto, name_to_latest,
                             feat, race_date, heads, default_futan)
    print(f'  特徴量構築完了: {len(rf)}頭')

    # 3. 学習（レース日より前の芝データ直近20000件）
    use_f = [f for f in FEATURES_V9 if f in feat.columns]
    tr = feat[
        (feat['date'] < race_date) &
        (feat['is_turf'] == 1) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date').tail(20000).copy()
    print(f'  学習データ: {len(tr):,}件 ({tr["date"].min()} ~ {tr["date"].max()})')

    pred = Predictor(use_f, CAT_FEATURES)
    pred.train(tr, ep=50, lr=0.003, seed=42)

    # 4. 推論 + QMC
    nn_preds = pred.predict(rf)
    mc = qmc_sim(nn_preds, race_features=rf, course=course_key, n=100000)

    t1 = time.time()
    print(f'  完了: {t1 - t0:.0f}秒')

    # 5. 結果表示
    print(f'\n  {"順位":>4s} {"枠":>2s} {"番":>3s} {"馬名":20s} '
          f'{"勝率":>7s} {"複勝率":>7s} {"E[rank]":>8s} '
          f'{"μ":>7s} {"σ":>7s} '
          f'{"EMA着":>7s} {"タイムZ":>8s} {"末脚":>7s} {"脚質":>5s}')
    print(f'  {"-"*110}')
    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        rf_row = rf[rf['umaban'] == u].iloc[0]
        ps_row = nn_preds[nn_preds['umaban'] == u].iloc[0]
        print(f'  {rank:4d} {int(rf_row["wakuban"]):2d} {u:3d} {r["horse_name"]:20s} '
              f'{r["win_prob"]:7.2%} {r["top3_prob"]:7.2%} {r["expected_rank"]:8.2f} '
              f'{ps_row["mu"]:7.4f} {ps_row["sigma"]:7.4f} '
              f'{rf_row["ema_finish"]:7.2f} {rf_row["ema_time_zscore"]:8.3f} '
              f'{rf_row["ema_agari"]:7.2f} {rf_row["avg_run_style"]:5.2f}')

    # 6. プロンプト生成
    prompt = build_prompt(
        race_name=race_name,
        course_name=course_display,
        distance=distance,
        heads=heads,
        date_str=str(race_date.date()),
        mc_results=mc,
        race_features=rf,
        nn_preds=nn_preds,
    )

    return mc, rf, nn_preds, prompt


def main():
    t_start = time.time()

    # 特徴量キャッシュ読み込み
    cache_path = 'data/features_v9b_2026.pkl'
    print(f'Loading {cache_path}...')
    feat = pd.read_pickle(cache_path)
    print(f'  {len(feat):,} rows ({feat["date"].min()} ~ {feat["date"].max()})')

    race_date = pd.Timestamp('2026-04-11')

    # === NZT G2 ===
    mc_nzt, rf_nzt, ps_nzt, prompt_nzt = predict_race(
        race_name='第44回ニュージーランドトロフィー G2',
        course_key='nakayama_turf_1600',
        course_display='中山芝1600m',
        distance=1600,
        html_path='C:/TXT/ニュージーランドT2026.html',
        csv_path='C:/TXT/ニュージーランドT.csv',
        feat=feat,
        race_date=race_date,
        default_futan=57.0,
    )

    # === 阪神牝馬S G2 ===
    mc_hh, rf_hh, ps_hh, prompt_hh = predict_race(
        race_name='阪神牝馬ステークス G2',
        course_key='hanshin_turf_1600_outer',
        course_display='阪神芝1600m外回り',
        distance=1600,
        html_path='C:/TXT/阪神牝馬S　2026.html',
        csv_path='C:/TXT/阪神牝馬.csv',
        feat=feat,
        race_date=race_date,
        default_futan=55.0,
    )

    # プロンプトをファイル保存
    for name, prompt in [('NZT', prompt_nzt), ('阪神牝馬S', prompt_hh)]:
        path = f'output_prompt_{name}.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f'\n  プロンプト保存: {path}')

    total = time.time() - t_start
    print(f'\n  Total: {total:.0f}秒')
    print('Done.')


if __name__ == '__main__':
    main()
