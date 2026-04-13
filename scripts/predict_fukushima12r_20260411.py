"""
2026/4/11 福島12R 芝1200m 1勝クラス(牝) 16頭 予測
Pipeline: 特徴量キャッシュ → NN学習 → コース別QMC(fukushima_turf_1200)
"""
import sys
import io
import os
import warnings
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch
import re
import time

np.random.seed(42)

from src.predictor import Predictor, FEATURES_V9, CAT_FEATURES
from src.qmc_courses import qmc_sim

# === 定義 ===
CLASS_MAP = {'A': 5, 'B': 4, 'C': 4, 'L': 3, 'E': 3}
RACE_DATE = pd.Timestamp('2026-04-11')
COURSE = 'fukushima_turf_1200'
N_HEADS = 16

# ==========================================================
# Step 1: 出馬表HTML解析
# ==========================================================
print('[Step1] Parsing entry table...')
with open('C:/TXT/12.html', 'rb') as f:
    html = f.read().decode('cp932')

rows_html = html.split('<TR>')[1:]
entry_list = []
for row in rows_html:
    tds = re.findall(r'<TD[^>]*>(.*?)</TD>', row, re.DOTALL)
    if len(tds) < 10:
        continue
    cleaned = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
    waku_str = re.sub(r'[^0-9]', '', cleaned[0])
    umaban_str = re.sub(r'[^0-9]', '', cleaned[1])
    if not waku_str or not umaban_str:
        continue
    futan_raw = cleaned[10].replace('▲', '').replace('△', '').replace('◇', '').strip()
    entry_list.append({
        'wakuban': int(waku_str),
        'umaban': int(umaban_str),
        'horse_name': cleaned[6],
        'jockey': cleaned[9].lstrip('*'),
        'futan': float(futan_raw),
    })

print(f'  {len(entry_list)} horses parsed')

# ==========================================================
# Step 2: CSVから血統番号マッピング + 前走情報
# ==========================================================
print('\n[Step2] Loading CSV past performance data...')
with open('C:/TXT/2026041112R.csv', 'rb') as f:
    csv_raw = f.read().decode('cp932')
csv_lines = csv_raw.strip().split('\r\n')

# Build horse_id -> name mapping and latest race data
csv_horses = {}
for line in csv_lines:
    c = line.split(',')
    hid = c[37].strip()
    hname = c[13].strip()
    yr, mo, da = int(c[0]), int(c[1]), int(c[2])
    date_val = f'{2000+yr:04d}{mo:02d}{da:02d}'
    if hid not in csv_horses:
        csv_horses[hid] = {'name': hname, 'rows': []}
    csv_horses[hid]['rows'].append({
        'date_str': date_val,
        'venue': c[4].strip(),
        'distance': int(c[11]),
        'surface': c[9].strip(),
        'finish': int(c[19]) if c[19].strip().isdigit() else 0,
        'heads': int(c[18]) if c[18].strip().isdigit() else 16,
        'time_diff': float(c[23]) if c[23].strip() else 0,
        'last3f': float(c[32]) if c[32].strip() else 0,
        'corner3': int(c[20]) if c[20].strip().isdigit() else 0,
        'corner4': int(c[21]) if c[21].strip().isdigit() else 0,
        'body_weight': int(c[33]) if c[33].strip().isdigit() else 0,
        'jockey_code': c[38].strip() if len(c) > 38 else '',
        'trainer_code': c[39].strip() if len(c) > 39 else '',
    })

# Match entry list to horse IDs
name_to_hid = {}
for hid, hdata in csv_horses.items():
    name_to_hid[hdata['name']] = hid

print(f'  CSV horses: {len(csv_horses)}')

# ==========================================================
# Step 3: 特徴量キャッシュロード + 馬別最新レコード取得
# ==========================================================
print('\n[Step3] Loading feature cache...')
t0 = time.time()

cache_path = 'data/features_v9b_2026.pkl'
if os.path.exists(cache_path):
    feat = pd.read_pickle(cache_path)
    print(f'  Loaded: {len(feat):,} rows ({feat["date"].min()} ~ {feat["date"].max()})')
else:
    cache_path = 'data/features_v9b_cache.pkl'
    feat = pd.read_pickle(cache_path)
    print(f'  Loaded fallback: {len(feat):,} rows')

# Build race features for each entry horse
race_rows = []
missing = []

for entry in entry_list:
    hname = entry['horse_name']
    hid = name_to_hid.get(hname)

    if hid is None:
        print(f'  WARNING: {hname} not found in CSV')
        missing.append(hname)
        continue

    ketto = '20' + hid  # CSV stores without century prefix
    # Try also without prefix
    horse_feat = feat[(feat['ketto_num'] == ketto) & (feat['date'] < RACE_DATE)]
    if len(horse_feat) == 0:
        horse_feat = feat[(feat['ketto_num'] == hid) & (feat['date'] < RACE_DATE)]
    if len(horse_feat) == 0:
        print(f'  WARNING: {hname} (ketto={ketto}) has no feature records, using CSV fallback')
        missing.append(hname)
        continue

    latest_row = horse_feat.sort_values('date').iloc[-1].copy()

    # Override with today's entry info
    latest_row['wakuban'] = entry['wakuban']
    latest_row['umaban'] = entry['umaban']
    latest_row['heads'] = N_HEADS
    latest_row['futan'] = entry['futan']
    latest_row['horse_name'] = hname

    # Compute interval_days from latest CSV race
    csv_rows = csv_horses[hid]['rows']
    csv_rows_sorted = sorted(csv_rows, key=lambda x: x['date_str'], reverse=True)
    if csv_rows_sorted:
        latest_csv = csv_rows_sorted[0]
        ds = latest_csv['date_str']
        prev_date = pd.Timestamp(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        latest_row['interval_days'] = (RACE_DATE - prev_date).days
        latest_row['prev_dist_diff'] = 1200 - latest_csv['distance']

    latest_row['odds'] = np.nan
    latest_row['race_id'] = '20260411_fukushima_12'
    latest_row['date'] = RACE_DATE

    race_rows.append(latest_row)

rf = pd.DataFrame(race_rows).reset_index(drop=True)
print(f'  Race features built: {len(rf)} horses')
if missing:
    print(f'  Missing: {missing}')

t1 = time.time()
print(f'  Step3 done in {t1-t0:.0f}s')

# ==========================================================
# Step 4: NN学習
# ==========================================================
print(f'\n[Step4] Training model (seed=42, ep=50, lr=0.003)...')
use_f = [f for f in FEATURES_V9 if f in feat.columns]
print(f'  Features: {len(use_f)}')

tr = feat[
    (feat['date'] < RACE_DATE) &
    (feat['is_turf'] == 1) &
    (feat['past_count'] > 0) &
    (feat['finish'] > 0)
].sort_values('date').tail(20000).copy()
print(f'  Training data: {len(tr):,} samples ({tr["date"].min()} ~ {tr["date"].max()})')

pred = Predictor(use_f, CAT_FEATURES)
pred.train(tr, ep=50, lr=0.003, seed=42)
t2 = time.time()
print(f'  Training done in {t2-t1:.0f}s')

# ==========================================================
# Step 5: 推論 + QMC (福島芝1200m)
# ==========================================================
print(f'\n[Step5] Predict + QMC ({COURSE}, 100,000 sims)...')
ps = pred.predict(rf)
mc = qmc_sim(ps, race_features=rf, course=COURSE, n=100000)
t3 = time.time()
print(f'  QMC done in {t3-t2:.0f}s')

# ==========================================================
# 結果出力
# ==========================================================
print(f'\n{"="*80}')
print(f'  2026/4/11 福島12R 芝1200m 1勝クラス(牝) v9b + QMC予測結果')
print(f'  コースプロファイル: {COURSE} (直線292m, 逃げ先行有利)')
print(f'{"="*80}')

print(f'\n  {"順位":>4s} {"枠":>2s} {"番":>3s} {"馬名":18s} {"勝率":>7s} {"複勝率":>7s} {"期待着順":>8s} {"μ":>7s} {"σ":>7s}')
print(f'  {"-"*76}')
for rank, (_, r) in enumerate(mc.iterrows(), 1):
    u = int(r['umaban'])
    rf_row = rf[rf['umaban'] == u]
    if len(rf_row) == 0:
        continue
    rf_row = rf_row.iloc[0]
    ps_row = ps[ps['umaban'] == u].iloc[0]
    marker = ''
    if rank <= 2:
        marker = ' ★'
    elif rank <= 5:
        marker = ' ◎'
    print(f'  {rank:4d} {int(rf_row["wakuban"]):2d} {u:3d} {r["horse_name"]:18s} '
          f'{r["win_prob"]:7.1%} {r["top3_prob"]:7.1%} {r["expected_rank"]:8.2f} '
          f'{ps_row["mu"]:7.4f} {ps_row["sigma"]:7.4f}{marker}')

# 詳細: 脚質・末脚情報
print(f'\n  === 詳細特徴量 ===')
print(f'  {"番":>3s} {"馬名":18s} {"脚質":>6s} {"末脚EMA":>8s} {"直線適性":>8s} {"EMA着順":>8s} {"間隔日":>6s} {"前走距差":>7s}')
print(f'  {"-"*76}')
for _, r in mc.iterrows():
    u = int(r['umaban'])
    rf_row = rf[rf['umaban'] == u]
    if len(rf_row) == 0:
        continue
    rf_row = rf_row.iloc[0]
    ls_str = f'{rf_row["long_stretch_avg"]:8.1f}' if pd.notna(rf_row.get('long_stretch_avg')) else '     N/A'
    style_val = rf_row.get('avg_run_style', 2.5)
    style_name = {1: '逃げ', 2: '先行', 3: '差し', 4: '追込'}.get(
        round(style_val) if pd.notna(style_val) else 3, '不明')
    print(f'  {u:3d} {r["horse_name"]:18s} '
          f'{style_val:5.1f}{style_name} '
          f'{rf_row.get("ema_agari", 0):8.2f} '
          f'{ls_str} '
          f'{rf_row.get("ema_finish", 0):8.2f} '
          f'{rf_row.get("interval_days", 0):6.0f} '
          f'{rf_row.get("prev_dist_diff", 0):7.0f}')

# 買い目提案
print(f'\n  === 買い目提案 ===')
top5 = mc.head(5)
top5_umaban = top5['umaban'].astype(int).tolist()
print(f'  軸候補:   {top5_umaban[:2]}')
print(f'  相手候補: {top5_umaban[2:5]}')
print(f'  ワイド:   {top5_umaban[0]}-{top5_umaban[1]}, {top5_umaban[0]}-{top5_umaban[2]}')
print(f'  3連複:    {top5_umaban[0]}-{top5_umaban[1]}-{top5_umaban[2]}')

# アノマリー枠: 上位5頭以外で勝率最高の馬
non_top5 = mc.iloc[5:]
if len(non_top5) > 0:
    anomaly = non_top5.sort_values('win_prob', ascending=False).iloc[0]
    print(f'  穴馬注意: {int(anomaly["umaban"])}番 {anomaly["horse_name"]} (勝率{anomaly["win_prob"]:.1%})')

total = time.time() - t0
print(f'\n  Total: {total:.0f}s')
print('\nDone.')
