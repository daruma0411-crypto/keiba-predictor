"""
直接予測スクリプト: 25列CSV(10R1形式) + 特徴量キャッシュから予測
entry_parserを使わず、馬名+オッズで特徴量キャッシュとマッチさせる
"""
import sys, os, warnings
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
from src.qmc_courses import qmc_sim, COURSE_PROFILES
from src.prompts import build_prompt

# ============================================================
# 入力
# ============================================================
csv_path = sys.argv[1]
course_key = sys.argv[2] if len(sys.argv) > 2 else None

# CSV読み込み (25列フォーマット)
with open(csv_path, 'rb') as f:
    raw = f.read()
text = raw.decode('cp932')
lines = text.strip().split('\r\n')

entries = []
for line in lines:
    c = line.split(',')
    # 馬体重: col16が数値なら25列フォーマット、そうでなければ23列(馬体重なし)
    try:
        bataijyu = int(c[16])
    except (ValueError, IndexError):
        bataijyu = 0  # キャッシュから取得
    entries.append({
        'wakuban': int(c[0]),
        'umaban': int(c[2]),
        'horse_name': c[7].strip(),
        'sex': c[9].strip(),
        'age': int(c[10]),
        'jockey': c[12].strip(),
        'futan': float(c[13]),
        'odds': float(c[15]),
        'bataijyu': bataijyu,
    })

heads = len(entries)
print(f'出走馬: {heads}頭')
for e in entries:
    print(f'  {e["umaban"]:2d}番 枠{e["wakuban"]} {e["horse_name"]:15s} {e["odds"]:6.1f}倍 {e["jockey"]}')

# ============================================================
# 特徴量キャッシュロード
# ============================================================
print(f'\nLoading feature cache...')
feat = pd.read_pickle('data/features_v9b_2026.pkl')
feat['date'] = pd.to_datetime(feat['date'])
print(f'  Cache: {len(feat):,} rows ({feat["date"].min()} ~ {feat["date"].max()})')

# 馬名でマッチング（最新のレコードを使用）
race_rows = []
missing = []
for e in entries:
    # 馬名で検索（最新5走を取得し、最新を使う）
    cand = feat[feat['horse_name'] == e['horse_name']].sort_values('date')
    if len(cand) == 0:
        # cp932エンコード問題: bytesで比較
        name_bytes = e['horse_name'].encode('cp932', errors='ignore')
        cand = feat[feat['horse_name'].apply(
            lambda x: x.encode('cp932', errors='ignore') == name_bytes if isinstance(x, str) else False
        )].sort_values('date')

    if len(cand) == 0:
        missing.append(e['horse_name'])
        continue

    latest = cand.iloc[-1].copy()
    # 今回のレース情報で上書き
    latest['umaban'] = e['umaban']
    latest['wakuban'] = e['wakuban']
    latest['futan'] = e['futan']
    latest['odds'] = e['odds']
    if e['bataijyu'] > 0:
        latest['bataijyu'] = e['bataijyu']
    # else: keep cache value
    latest['heads'] = heads
    latest['horse_name'] = e['horse_name']
    race_rows.append(latest)

if missing:
    print(f'  ⚠ Missing: {missing}')

rf = pd.DataFrame(race_rows)
print(f'  Matched: {len(rf)}/{heads} horses')

# ============================================================
# NN学習 (PredictorV2)
# ============================================================
print(f'\nTraining PredictorV2...')
use_f = [f for f in FEATURES_V2 if f in feat.columns]
use_cf = [f for f in CAT_FEATURES_V2 if f in feat.columns]

tr = feat[
    (feat['is_turf'] == 1) &
    (feat['past_count'] > 0) &
    (feat['finish'] > 0)
].sort_values('date').tail(40000).copy()
print(f'  Training: {len(tr):,} samples')

pred = PredictorV2(use_f, use_cf)
pred.train(tr, ep=80, lr=0.003, seed=42)

# ============================================================
# 推論 + QMC
# ============================================================
if course_key and course_key in COURSE_PROFILES:
    cp = COURSE_PROFILES[course_key]
    print(f'\nQMC: {cp["name"]} (100,000 sims)...')
else:
    # デフォルト
    course_key = 'hanshin_turf_1600_outer'
    cp = COURSE_PROFILES[course_key]
    print(f'\nQMC: default profile (100,000 sims)...')

ps = pred.predict(rf)
mc = qmc_sim(ps, race_features=rf, course=course_key, n=100000)

# ============================================================
# 結果表示
# ============================================================
print(f'\n{"="*80}')
print(f'  阪神10R 芝2000m 3勝クラス {heads}頭')
print(f'  v2(D) + QMC予測結果')
print(f'{"="*80}')

if 'ninki' not in mc.columns:
    mc['ninki'] = mc['odds'].rank(method='first')

print(f'\n  {"順位":>4s} {"枠":>2s} {"番":>3s} {"馬名":15s} '
      f'{"勝率":>7s} {"複勝率":>7s} {"期待着順":>8s} {"μ":>7s} {"σ":>7s} {"odds":>7s}')
print(f'  {"-"*80}')

for rank, (_, r) in enumerate(mc.iterrows(), 1):
    u = int(r['umaban'])
    rf_row = rf[rf['umaban'] == u]
    if len(rf_row) == 0:
        continue
    rf_row = rf_row.iloc[0]
    ps_row = ps[ps['umaban'] == u]
    if len(ps_row) == 0:
        continue
    ps_row = ps_row.iloc[0]
    print(f'  {rank:4d} {int(rf_row["wakuban"]):2d} {u:3d} {r["horse_name"]:15s} '
          f'{r["win_prob"]:7.1%} {r["top3_prob"]:7.1%} {r["expected_rank"]:8.2f} '
          f'{ps_row["mu"]:7.4f} {ps_row["sigma"]:7.4f} {r["odds"]:7.1f}')

# TOP5
top5 = mc.head(5)
print(f'\n  QMC TOP5:')
for rk, (_, r) in enumerate(top5.iterrows(), 1):
    print(f'    {rk}. [{int(r["umaban"]):2d}] {r["horse_name"]} ({r["odds"]:.1f}倍)')

# ワイドBOX
from itertools import combinations
from src.debate_rules import select_with_rules

top5_list = [int(r['umaban']) for _, r in top5.iterrows()]
print(f'\n  ワイドBOX ({len(list(combinations(top5_list,2)))}点):')
for a, b in combinations(top5_list, 2):
    print(f'    {a}-{b}')

# ============================================================
# Layer 3: 議長プロンプト生成
# ============================================================
if '--no-debate' not in sys.argv:
    print(f'\n[Layer3] Generating debate prompt...')

    # 2系統選抜
    if 'ninki' not in mc.columns:
        mc_d = mc.copy()
        mc_d['ninki'] = mc_d['odds'].rank(method='first')
    else:
        mc_d = mc

    line_a = mc_d.head(5)
    try:
        line_b, pop_umabans, flagged_info = select_with_rules(mc_d, rf, n_pop=2, n_disc=3, cutoff=5)
    except Exception as e:
        print(f'  Warning: select_with_rules failed: {e}')
        line_b = None
        pop_umabans = None
        flagged_info = []

    # 前走テキスト生成
    past_text = ''
    fullset_path = os.environ.get('FULLSET_CSV', '')
    if fullset_path and os.path.exists(fullset_path):
        try:
            with open(fullset_path, 'rb') as f:
                raw_fs = f.read()
            text_fs = raw_fs.decode('cp932')
            from collections import defaultdict
            horse_runs = defaultdict(list)
            for fline in text_fs.strip().split('\r\n'):
                fc = fline.split(',')
                fname = fc[13].strip()
                venue = fc[4].strip()
                surface = fc[9].strip()
                dist_r = fc[11].strip()
                baba = fc[12].strip()
                finish_r = fc[20].strip()
                heads_r = fc[10].strip()
                race_nm = fc[7].strip()
                agari = fc[32].strip() if len(fc) > 32 else ''
                c3 = fc[30].strip() if len(fc) > 30 else ''
                c4 = fc[31].strip() if len(fc) > 31 else ''
                jockey_r = fc[16].strip() if len(fc) > 16 else ''
                weight_r = fc[33].strip() if len(fc) > 33 else ''
                horse_runs[fname].append(
                    f'{venue}{surface}{dist_r}m {baba} {finish_r}/{heads_r}着 {race_nm} 上り{agari} 通過{c3}-{c4} 騎手{jockey_r} 馬体重{weight_r}'
                )
            past_lines = []
            for e in entries:
                runs = horse_runs.get(e['horse_name'], [])
                if runs:
                    past_lines.append(f'{e["umaban"]}番 {e["horse_name"]}: ' + ' | '.join(runs))
            past_text = '\n'.join(past_lines)
            print(f'  Loaded past races for {len(past_lines)} horses from {fullset_path}')
        except Exception as e:
            print(f'  Warning: fullset parse failed: {e}')
    else:
        # フォールバック: 10r2形式
        r2_path = csv_path.replace('10R1', '10r2').replace('10r1', '10r2')
        if os.path.exists(r2_path) and r2_path != csv_path:
            try:
                with open(r2_path, 'rb') as f:
                    raw2 = f.read()
                text2 = raw2.decode('cp932')
                past_lines = []
                for fline in text2.strip().split('\r\n'):
                    fc = fline.split(',')
                    name = fc[7].strip()
                    runs = []
                    for start in range(11, len(fc)-10, 12):
                        if start+10 < len(fc):
                            try:
                                runs.append(f'{fc[start+2]}{fc[start+3]}{fc[start+4]}m {fc[start+5]} {fc[start+6]}/{fc[start+7]}着 {fc[start+8]} 上り{fc[start+9]}')
                            except (ValueError, IndexError):
                                pass
                    if runs:
                        past_lines.append(f'{fc[2]}番 {name}: ' + ' | '.join(runs))
                past_text = '\n'.join(past_lines)
            except Exception as e:
                print(f'  Warning: past race parse failed: {e}')

    race_name = os.environ.get('RACE_NAME', f'{cp["name"]} レース')
    course_name = cp['name']
    race_date = os.environ.get('RACE_DATE', '2026-04-19')
    prompt = build_prompt(
        race_name=race_name,
        course_name=course_name,
        distance=int(os.environ.get('RACE_DIST', '2000')),
        heads=heads,
        date_str=race_date,
        mc_results=mc,
        race_features=rf,
        nn_preds=ps,
        race_id=os.environ.get('RACE_ID', '20260419_race'),
        line_a=line_a,
        line_b=line_b,
        pop_umabans=pop_umabans,
        past_races_text=past_text,
    )

    out_name = f'output_{os.environ.get("RACE_ID","race")}_prompt.txt'
    with open(out_name, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f'  Saved: {out_name}')
    print(f'\n--- 議長プロンプト (Layer 3) ---')
    print(prompt)
