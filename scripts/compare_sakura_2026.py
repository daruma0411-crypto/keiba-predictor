"""
2026桜花賞: 手動QMC係数 vs ベイズ最適化係数の比較
既存predict_sakura_2026.pyのパイプラインを流用し、QMC部分だけ2パターン回す
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch, re
np.random.seed(42)

from src.predictor import Predictor, FEATURES_V9, CAT_FEATURES
from src.qmc_courses import qmc_sim, COURSE_PROFILES
from src.prompts import build_prompt

# 手動(旧)係数
OLD_PARAMS = {
    'name': '阪神芝1600m外回り (手動)',
    'straight': 473,
    'pace_base_per_runner': 0.30,
    'pace_noise': 0.15,
    'style_bonus': {'nige': +0.06, 'senkou': +0.02, 'sashi': -0.04, 'oikomi': -0.04},
    'gate_bias': {'inner_senkou': -0.010, 'outer_sashi': +0.010, 'inner_block': +0.010},
    'trouble_rate': 0.05, 'trouble_penalty': 0.15, 'noise_scale': 0.02,
}

t0 = time.time()

# === Step 1-2: 桜花賞出馬表構築 (predict_sakura_2026.pyと同じ) ===
print('[Step1] Loading features...')
feat = pd.read_pickle('data/features_v9b_2026.pkl')
RACE_DATE = pd.Timestamp('2026-04-12')

print('[Step2] Building sakura-sho race features...')
with open('C:/TXT/桜花賞５.html', 'r', encoding='cp932') as f:
    html = f.read()
rows_html = html.split('<TR>')[1:]
entry_list = []
for row in rows_html:
    text = re.sub(r'<[^>]+>', '|', row)
    cells = [c.strip() for c in text.split('|') if c.strip()]
    if len(cells) >= 10 and cells[0].isdigit() and cells[1].isdigit():
        entry_list.append({'wakuban': int(cells[0]), 'umaban': int(cells[1]), 'horse_name_html': cells[2]})
print(f'  HTML entries: {len(entry_list)}')

csv = pd.read_csv('C:/TXT/桜花賞.csv', encoding='cp932', header=None)
name_to_ketto = {}
name_to_latest = {}
for horse_name in csv[13].str.strip().unique():
    h_data = csv[csv[13].str.strip() == horse_name]
    ketto = '20' + str(h_data.iloc[0][37])
    name_to_ketto[horse_name] = ketto
    h_data = h_data.copy()
    h_data['_date'] = h_data[0].astype(str).str.zfill(2) + h_data[1].astype(str).str.zfill(2) + h_data[2].astype(str).str.zfill(2)
    name_to_latest[horse_name] = h_data.sort_values('_date', ascending=False).iloc[0]

sakura_rows = []
for entry in entry_list:
    hname = entry['horse_name_html']
    ketto = name_to_ketto.get(hname)
    if ketto is None: continue
    horse_feat = feat[(feat['ketto_num'] == ketto) & (feat['date'] < RACE_DATE)]
    if len(horse_feat) == 0: continue
    lr = horse_feat.sort_values('date').iloc[-1].copy()
    lr['wakuban'] = entry['wakuban']
    lr['umaban'] = entry['umaban']
    lr['heads'] = 18
    lr['futan'] = 55.0
    lr['horse_name'] = hname
    csv_latest = name_to_latest.get(hname)
    if csv_latest is not None:
        prev_date = pd.Timestamp(2000 + int(csv_latest[0]), int(csv_latest[1]), int(csv_latest[2]))
        lr['interval_days'] = (RACE_DATE - prev_date).days
        lr['prev_dist_diff'] = 1600 - int(csv_latest[11])
    lr['odds'] = np.nan
    lr['date'] = RACE_DATE
    sakura_rows.append(lr)

rf = pd.DataFrame(sakura_rows).reset_index(drop=True)
print(f'  {len(rf)} horses')

# === Step 3: NN学習 ===
print(f'\n[Step3] Training NN...')
use_f = [f for f in FEATURES_V9 if f in feat.columns]
tr = feat[(feat['date'] < RACE_DATE) & (feat['is_turf'] == 1) & (feat['past_count'] > 0) & (feat['finish'] > 0)].sort_values('date').tail(20000).copy()
pred = Predictor(use_f, CAT_FEATURES)
pred.train(tr, ep=50, lr=0.003, seed=42)
ps = pred.predict(rf)
t1 = time.time()
print(f'  Done in {t1-t0:.0f}s')

# === Step 4: QMC比較 ===
print(f'\n[Step4] Running QMC with both param sets...')

# 一時的にOLD_PARAMSでQMC実行
import src.qmc_courses as qmc_mod
saved = qmc_mod.COURSE_PROFILES['hanshin_turf_1600_outer'].copy()
qmc_mod.COURSE_PROFILES['_old'] = OLD_PARAMS
mc_old = qmc_sim(ps, race_features=rf, course='_old', n=100000)
mc_new = qmc_sim(ps, race_features=rf, course='hanshin_turf_1600_outer', n=100000)
del qmc_mod.COURSE_PROFILES['_old']

# === 結果比較 ===
print(f'\n{"="*90}')
print(f'  2026年 第86回桜花賞 QMC係数比較')
print(f'  阪神11R 芝1600m外 18頭 2026/4/12')
print(f'{"="*90}')

print(f'\n  {"":18s}  {"--- 手動係数 ---":>30s}   {"--- 最適化係数 ---":>30s}')
print(f'  {"順":>2s} {"枠":>2s} {"番":>3s} {"馬名":16s}  {"勝率":>6s} {"複勝":>6s} {"E[rk]":>6s}   {"勝率":>6s} {"複勝":>6s} {"E[rk]":>6s}  {"差分":>6s}')
print(f'  {"-"*86}')

# Merge by umaban
for rank_new, (_, rn) in enumerate(mc_new.iterrows(), 1):
    u = int(rn['umaban'])
    ro = mc_old[mc_old['umaban'] == u].iloc[0]
    rf_row = rf[rf['umaban'] == u].iloc[0]
    diff = ro['expected_rank'] - rn['expected_rank']
    arrow = '↑' if diff > 0.3 else ('↓' if diff < -0.3 else '→')
    print(f'  {rank_new:2d} {int(rf_row["wakuban"]):2d} {u:3d} {rn["horse_name"]:16s}  '
          f'{ro["win_prob"]:5.1%} {ro["top3_prob"]:5.1%} {ro["expected_rank"]:6.2f}   '
          f'{rn["win_prob"]:5.1%} {rn["top3_prob"]:5.1%} {rn["expected_rank"]:6.2f}  '
          f'{diff:+5.2f}{arrow}')

# 主な変化点
print(f'\n  === 主な変化点 ===')
for _, rn in mc_new.iterrows():
    u = int(rn['umaban'])
    ro = mc_old[mc_old['umaban'] == u].iloc[0]
    rf_row = rf[rf['umaban'] == u].iloc[0]
    style = rf_row.get('avg_run_style', 2.5)
    diff = ro['expected_rank'] - rn['expected_rank']
    if abs(diff) > 0.3:
        sname = {1: '逃げ', 2: '先行', 3: '差し', 4: '追込'}.get(round(style), '?')
        print(f'    {rn["horse_name"]:16s} 脚質{style:.1f}({sname}) E[rank] {ro["expected_rank"]:.2f}→{rn["expected_rank"]:.2f} ({diff:+.2f})')

# === Layer 3: 最適化係数で議長プロンプト生成 ===
print(f'\n[Layer3] Generating debate prompt (optimized params)...')
prompt = build_prompt(
    race_name='第86回桜花賞 G1',
    course_name='阪神芝1600m外回り (最適化済: 直線473m, 差しも届く)',
    distance=1600, heads=18, date_str='2026-04-12',
    mc_results=mc_new, race_features=rf, nn_preds=ps,
    race_id='20260412_hanshin_sakura',
)
with open('output_sakura2026_optimized_prompt.txt', 'w', encoding='utf-8') as f:
    f.write(prompt)
print(f'  Saved: output_sakura2026_optimized_prompt.txt')
print(prompt)

total = time.time() - t0
print(f'\n  Total: {total:.0f}s')
