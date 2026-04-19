"""2018年桜花賞 v9b CSV出力"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import qmc, norm
np.random.seed(42)

from scripts.simulate_v9b import FEATURES_V9, CAT_FEATURES, HorseRaceModel, Predictor, qmc_sim

def run():
    from src.binary_parser import load_all_data, load_hanshin_data
    from src.um_parser import load_um_data

    df_h = load_hanshin_data(years=range(2015, 2026))
    df_h = df_h[df_h['kakutei_jyuni'] > 0]
    feat = pd.read_pickle('data/features_v9b_cache.pkl')

    sakura_all = df_h[(df_h['race_name'].str.contains('桜花賞', na=False)) & (df_h['grade_cd'] == 'A')]
    sakura_ids = {int(rid[:4]): rid for rid in sakura_all['race_id'].unique()}

    rid = sakura_ids[2018]
    rd = df_h[df_h['race_id'] == rid]
    rdate = rd['date'].iloc[0]
    rf = feat[feat['race_id'] == rid].copy()

    use_f = [f for f in FEATURES_V9 if f in feat.columns]
    tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()

    pred = Predictor(use_f, CAT_FEATURES)
    pred.train(tr, ep=50, lr=0.003, seed=42)
    ps = pred.predict(rf)
    mc = qmc_sim(ps, race_features=rf, n=100000)

    # Build merged dataframe
    rows = []
    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        hd = rd[rd['umaban'] == u].iloc[0]
        fd = rf[rf['umaban'] == u].iloc[0]
        mu_val = ps[ps['umaban']==u]['mu'].values[0]
        sig_val = ps[ps['umaban']==u]['sigma'].values[0]

        row = {
            'v9b_rank': rank,
            'horse_name': hd['horse_name'],
            'umaban': u,
            'wakuban': int(fd['wakuban']),
            'odds': r['odds'],
            'mu': mu_val,
            'sigma': sig_val,
            'win_prob': r['win_prob'],
            'top3_prob': r['top3_prob'],
            'expected_rank': r['expected_rank'],
            'ema_finish': fd['ema_finish'],
            'ema_time_zscore': fd['ema_time_zscore'],
            'ema_agari': fd['ema_agari'],
            'weighted_ema_finish': fd['weighted_ema_finish'],
            'long_stretch_avg': fd.get('long_stretch_avg', None),
            'same_dist_finish': fd.get('same_dist_finish', None),
            'same_surface_finish': fd.get('same_surface_finish', None),
            'prev_dist_diff': fd['prev_dist_diff'],
            'avg_run_style': fd['avg_run_style'],
            'avg_jyuni_3c': fd['avg_jyuni_3c'],
            'avg_jyuni_4c': fd['avg_jyuni_4c'],
            'futan': fd['futan'],
            'bataijyu': fd['bataijyu'],
            'zogen_sa': fd['zogen_sa'],
            'interval_days': fd['interval_days'],
            'past_count': fd['past_count'],
            'jockey_win_rate': fd['jockey_win_rate'],
            'jockey_top3_rate': fd['jockey_top3_rate'],
            'trainer_win_rate': fd['trainer_win_rate'],
            'trainer_top3_rate': fd['trainer_top3_rate'],
            'prev_race_class': fd['prev_race_class'],
            'log_prize_money': fd['log_prize_money'],
            'win_rate': fd['win_rate'],
            'top3_rate': fd['top3_rate'],
            'prob_1': r.get('prob_1', 0),
            'prob_2': r.get('prob_2', 0),
            'prob_3': r.get('prob_3', 0),
            'prob_4': r.get('prob_4', 0),
            'prob_5': r.get('prob_5', 0),
            'prob_6': r.get('prob_6', 0),
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv('output_2018_sakura.csv', index=False, encoding='utf-8-sig')
    print('Saved to output_2018_sakura.csv')

if __name__ == '__main__':
    run()
