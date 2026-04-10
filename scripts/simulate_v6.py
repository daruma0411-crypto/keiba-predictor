"""
桜花賞シミュレーション v6
v5からの改善:
1. 大敗馬ペナルティ強化 (ema_finish>8)
2. 前走大敗もペナルティ
3. 桜花賞スコアに「前走着順」をより重視
4. 人気（オッズ）情報を間接的に活用（過去走の人気傾向）
"""
import sys
import io
import os
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
np.random.seed(42)

from src.binary_parser import load_all_data, load_hanshin_data
from src.um_parser import load_um_data
from src.model import RacePredictor, monte_carlo_simulation
from src.sakura_model import compute_sakura_score

FEATURES = [
    'wakuban', 'futan', 'bataijyu', 'zogen_sa',
    'sex', 'barei', 'heads',
    'past_count', 'ema_time_zscore', 'ema_finish',
    'win_rate', 'top3_rate', 'avg_run_style',
    'same_dist_finish', 'same_surface_finish', 'interval_days',
    'jockey_win_rate', 'jockey_top3_rate',
    'trainer_win_rate', 'trainer_top3_rate',
    'avg_jyuni_3c', 'avg_jyuni_4c',
]
CAT = ['kisyu_code', 'chokyosi_code', 'banusi_code']

print('Loading data...')
df_h = load_hanshin_data(years=range(2015, 2026))
df_h = df_h[df_h['kakutei_jyuni'] > 0]
df_all = load_all_data(years=range(2014, 2026))
df_all = df_all[df_all['kakutei_jyuni'] > 0]
um = load_um_data(years=range(2010, 2026))
feat = pd.read_pickle('data/features_all_v4.pkl')

available = [f for f in FEATURES if f in feat.columns]
FEATURES = available

sakura = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
sakura_ids = {int(rid[:4]): rid for rid in sakura['race_id'].unique()}

results = []
for year in range(2016, 2026):
    if year not in sakura_ids:
        continue
    rid = sakura_ids[year]
    race_data = df_h[df_h['race_id'] == rid]
    race_date = race_data['date'].iloc[0]
    race_feat = feat[feat['race_id'] == rid].copy()
    if len(race_feat) == 0:
        continue

    train = feat[
        (feat['date'] < race_date) &
        (feat['is_turf'] == 1) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date').tail(20000).copy()

    predictor = RacePredictor(FEATURES, CAT)
    predictor.train(train, epochs=50, lr=0.003, seed=42)
    preds = predictor.predict(race_feat)

    np.random.seed(year)
    mc = monte_carlo_simulation(preds, race_features=race_feat, n_simulations=100000)

    # 桜花賞スコア
    ss = compute_sakura_score(race_feat, df_all, um)

    # === 合成スコア v6 ===
    merged = mc.merge(ss[['umaban', 'sakura_score', 'reasons']], on='umaban', how='left')

    # MC期待順位を正規化
    er = merged['expected_rank']
    merged['mc_score'] = 1 - (er - er.min()) / (er.max() - er.min()) if er.max() > er.min() else 0.5

    # 桜花賞スコア正規化
    ss_col = merged['sakura_score'].fillna(0)
    merged['sakura_norm'] = (ss_col - ss_col.min()) / (ss_col.max() - ss_col.min()) if ss_col.max() > ss_col.min() else 0.5

    # === ペナルティ/ボーナス ===
    penalty = np.zeros(len(merged))
    for i, (_, row) in enumerate(merged.iterrows()):
        uma = int(row['umaban'])
        h = race_feat[race_feat['umaban'] == uma]
        if len(h) == 0:
            continue
        hf = h.iloc[0]

        ef = hf.get('ema_finish', 5)
        wr = hf.get('win_rate', 0)
        t3r = hf.get('top3_rate', 0)
        pc = hf.get('past_count', 0)

        # 大敗馬ペナルティ（段階的）
        if pd.notna(ef):
            if ef > 12:
                penalty[i] -= 0.3
            elif ef > 10:
                penalty[i] -= 0.2
            elif ef > 8:
                penalty[i] -= 0.1

        # 勝率0%かつ複勝率0%の馬にペナルティ
        if pd.notna(wr) and pd.notna(t3r):
            if wr == 0 and t3r == 0 and pc >= 3:
                penalty[i] -= 0.15

        # 少キャリアで好成績の馬にボーナス
        if pd.notna(wr) and pc <= 5 and pc >= 1:
            if wr > 0.3:
                penalty[i] += 0.1
            if t3r > 0.5:
                penalty[i] += 0.05

        # 前走情報（桜花賞スコアから間接的に）
        # → sakura_scoreが高い馬は前走パターンが良い

    merged['penalty'] = penalty

    # 合成: MC 70% + 桜花賞 20% + ペナルティ 10%
    merged['final_score'] = (
        0.70 * merged['mc_score'] +
        0.20 * merged['sakura_norm'] +
        0.10 * (merged['penalty'] - merged['penalty'].min()) /
        (merged['penalty'].max() - merged['penalty'].min() + 0.001)
    )
    merged = merged.sort_values('final_score', ascending=False)

    # 結果
    actual = race_data.sort_values('kakutei_jyuni').head(3)
    actual_top3 = set(actual['umaban'].astype(int))

    print(f'\n{year}年桜花賞:')
    print(f'  予測TOP5:')
    for rank, (_, r) in enumerate(merged.head(5).iterrows(), 1):
        uma = int(r['umaban'])
        act = race_data[race_data['umaban'] == uma]
        af = int(act.iloc[0]['kakutei_jyuni']) if len(act) > 0 else '?'
        hit = '*' if uma in actual_top3 else ' '
        reasons = str(r.get('reasons', ''))[:30]
        print(f'    {hit}{rank}位 [{uma:2d}] {r["horse_name"]:16s} MC={r["mc_score"]:.2f} 桜={r["sakura_norm"]:.2f} P={r["penalty"]:+.2f} (実際{af}着) {reasons}')
    print(f'  実際:')
    for _, r in actual.iterrows():
        print(f'    {int(r["kakutei_jyuni"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]:16s} ({int(r["ninki"])}人気)')

    pred_1st = int(merged.iloc[0]['umaban'])
    actual_1st = int(actual.iloc[0]['umaban'])
    pred_1st_actual = int(race_data[race_data['umaban'] == pred_1st].iloc[0]['kakutei_jyuni'])
    pred_top5 = set(merged.head(5)['umaban'].astype(int))
    overlap = len(pred_top5 & actual_top3)
    win = pred_1st == actual_1st
    results.append({
        'year': year, 'win': win, 'overlap': overlap,
        't5': actual_1st in pred_top5, 'pred_finish': pred_1st_actual
    })
    sys.stdout.flush()

print(f'\n{"="*50}')
for r in results:
    m = '◎' if r['win'] else ('○' if r['t5'] else '×')
    print(f'  {r["year"]}: {m} 予測1位→{r["pred_finish"]}着 上位5に3着内{r["overlap"]}/3')
w = sum(r['win'] for r in results)
t5 = sum(r['t5'] for r in results)
avg = np.mean([r['overlap'] for r in results])
pred_top3 = sum(1 for r in results if r['pred_finish'] <= 3)
print(f'\n  単勝: {w}/10 | 1着TOP5内: {t5}/10 | 3着内一致: {avg:.1f}/3 | 予測1位が3着内: {pred_top3}/10')
