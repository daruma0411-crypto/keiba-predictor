"""
QMCコース係数ベイズ最適化 (Optuna)

阪神芝1600m外の全レース + 桜花賞10年分を重み付きで評価し、
COURSE_PROFILES['hanshin_turf_1600_outer'] の9パラメータを最適化する。

使い方:
    py -3.13 scripts/optimize_qmc.py
    py -3.13 scripts/optimize_qmc.py --trials 150 --sakura-weight 5.0
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

import numpy as np
import pandas as pd
import torch
import optuna
from scipy.stats import qmc as scipy_qmc, norm
from src.predictor import Predictor, FEATURES_V9, CAT_FEATURES

np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# QMCシミュレーション（パラメータ可変版）
# ============================================================
def qmc_sim_parameterized(preds, rf, params, n=100000):
    """パラメータ辞書を受け取るQMC"""
    nh = len(preds)
    mu = preds['mu'].values.copy()
    sig = preds['sigma'].values.copy()

    rs = np.full(nh, 2.5)
    wk = np.arange(1, nh + 1)
    if rf is not None and len(rf) == nh:
        if 'avg_run_style' in rf.columns:
            r = rf['avg_run_style'].values
            rs = np.where(np.isnan(r), 2.5, r)
        if 'wakuban' in rf.columns:
            wk = rf['wakuban'].values.astype(float)

    nn_nige = np.sum(rs <= 1.5)
    pace_base = (nn_nige - 1.5) * params['pace_base_per_runner']

    nd = nh * 4 + 1
    sampler = scipy_qmc.Sobol(d=nd, scramble=True, seed=42)
    np2 = 2 ** int(np.ceil(np.log2(n)))
    sb = sampler.random(np2)[:n]
    sn = norm.ppf(np.clip(sb, 0.001, 0.999))

    j = 0
    ability = mu[np.newaxis, :] + sig[np.newaxis, :] * sn[:, j:j+nh]; j += nh
    pace = pace_base + params['pace_noise'] * sn[:, j]; j += 1
    luck_u = sb[:, j:j+nh]; j += nh
    block_u = sb[:, j:j+nh]; j += nh
    indiv_noise = sn[:, j:j+nh]; j += nh

    pace2 = pace[:, np.newaxis]

    is_nige   = (rs <= 1.5).astype(float)
    is_senkou = ((rs > 1.5) & (rs <= 2.5)).astype(float)
    is_sashi  = ((rs > 2.5) & (rs < 3.5)).astype(float)
    is_oikomi = (rs >= 3.5).astype(float)

    ability += pace2 * params['bonus_nige']   * is_nige[np.newaxis, :]
    ability += pace2 * params['bonus_senkou'] * is_senkou[np.newaxis, :]
    ability += pace2 * params['bonus_sashi']  * is_sashi[np.newaxis, :]
    ability += pace2 * params['bonus_oikomi'] * is_oikomi[np.newaxis, :]

    inner_senkou = ((wk <= 3) & (rs <= 2.5)).astype(float)
    ability += params['gate_inner_senkou'] * inner_senkou[np.newaxis, :]
    outer_sashi = ((wk >= 6) & (rs >= 3.5)).astype(float)
    ability += params['gate_outer_sashi'] * outer_sashi[np.newaxis, :]
    inner_block_flag = ((wk <= 3) & (rs >= 3.0)).astype(float)
    ability += (block_u < 0.10).astype(float) * inner_block_flag[np.newaxis, :] * params.get('gate_inner_block', 0.01)

    ability += (luck_u < params.get('trouble_rate', 0.05)).astype(float) * params.get('trouble_penalty', 0.15)
    ability += params.get('noise_scale', 0.02) * indiv_noise

    ranks = np.argsort(np.argsort(ability, axis=1), axis=1) + 1

    res = preds[['horse_name', 'umaban', 'odds']].copy()
    for pos in range(1, min(nh + 1, 19)):
        res[f'prob_{pos}'] = (ranks == pos).mean(axis=0)
    res['expected_rank'] = ranks.mean(axis=0)
    res['win_prob'] = res['prob_1']
    res['top3_prob'] = res[['prob_1', 'prob_2', 'prob_3']].sum(axis=1)
    return res.sort_values('expected_rank')


# ============================================================
# バックテスト用データ準備
# ============================================================
def prepare_backtest_data(feat):
    """阪神芝1600mのレース一覧 + 桜花賞フラグを返す"""
    # race_idフォーマット: YYYYMMDD_PP_KK_DD_RR (PP=place_code)
    feat = feat.copy()
    feat['_place'] = feat['race_id'].str.split('_').str[1]
    feat['_racenum'] = feat['race_id'].str.split('_').str[4]

    # 阪神(09) 芝 1600m
    mask = (
        (feat['_place'] == '09') &
        (feat['is_turf'] == 1) &
        (feat['kyori'] == 1600) &
        (feat['finish'] > 0) &
        (feat['past_count'] > 0)
    )
    hanshin_1600 = feat[mask].copy()
    race_ids = hanshin_1600['race_id'].unique()
    print(f'  阪神芝1600m レース数: {len(race_ids)}')

    # 桜花賞特定: 4月 + 11R + 3歳平均 + 16頭以上 + 年1回（最多頭数を採用）
    sakura_candidates = {}
    for rid in race_ids:
        rd = hanshin_1600[hanshin_1600['race_id'] == rid]
        dt = rd['date'].iloc[0]
        rnum = rd['_racenum'].iloc[0]
        n_heads = len(rd)
        avg_age = rd['barei'].mean()
        if (dt.month == 4 and rnum == '11' and n_heads >= 16 and avg_age <= 3.1):
            year = dt.year
            if year not in sakura_candidates or n_heads > sakura_candidates[year][1]:
                sakura_candidates[year] = (rid, n_heads)

    sakura_ids = set(v[0] for v in sakura_candidates.values())
    print(f'  うち桜花賞: {len(sakura_ids)}')
    for sid in sorted(sakura_ids):
        rd = hanshin_1600[hanshin_1600['race_id'] == sid]
        print(f'    {sid} ({rd["date"].iloc[0].date()}, {len(rd)}頭)')

    hanshin_1600.drop(columns=['_place', '_racenum'], inplace=True)
    return hanshin_1600, race_ids, sakura_ids


# ============================================================
# 目的関数
# ============================================================
class QMCObjective:
    def __init__(self, feat, hanshin_data, race_ids, sakura_ids,
                 sakura_weight=3.0, n_sims=30000):
        self.feat = feat
        self.hanshin_data = hanshin_data
        self.race_ids = race_ids
        self.sakura_ids = sakura_ids
        self.sakura_weight = sakura_weight
        self.n_sims = n_sims

        # 事前にレースごとのNN予測をキャッシュ（学習は1回だけ）
        self.race_cache = {}
        self._pretrain_all()

    def _pretrain_all(self, max_non_sakura=80):
        """各レースのNN予測を事前計算してキャッシュ（サンプリングで高速化）"""
        print('  Pre-training NN for each race year...')
        use_f = [f for f in FEATURES_V9 if f in self.feat.columns]

        # 桜花賞は全件、非桜花賞はサンプリング
        sakura_rids = [r for r in self.race_ids if r in self.sakura_ids]
        non_sakura_rids = [r for r in self.race_ids if r not in self.sakura_ids]
        rng = np.random.RandomState(42)
        if len(non_sakura_rids) > max_non_sakura:
            non_sakura_rids = list(rng.choice(non_sakura_rids, max_non_sakura, replace=False))
        target_rids = sakura_rids + non_sakura_rids
        print(f'  Target races: {len(sakura_rids)} sakura + {len(non_sakura_rids)} others = {len(target_rids)}')

        # 年ごとにグループ化してNN学習
        year_models = {}
        for rid in target_rids:
            rf = self.hanshin_data[self.hanshin_data['race_id'] == rid]
            if len(rf) < 5:
                continue
            rdate = rf['date'].iloc[0]
            year = rdate.year

            if year not in year_models:
                tr = self.feat[
                    (self.feat['date'] < rdate) &
                    (self.feat['is_turf'] == 1) &
                    (self.feat['past_count'] > 0) &
                    (self.feat['finish'] > 0)
                ].sort_values('date').tail(20000).copy()

                if len(tr) < 1000:
                    continue

                pred = Predictor(use_f, CAT_FEATURES)
                pred.train(tr, ep=50, lr=0.003, seed=42)
                year_models[year] = pred

            if year in year_models:
                pred = year_models[year]
                try:
                    ps = pred.predict(rf)
                    actual = rf[['umaban', 'finish']].copy()
                    actual['umaban'] = actual['umaban'].astype(int)
                    actual_top3 = set(actual.nsmallest(3, 'finish')['umaban'])
                    actual_1st = actual.nsmallest(1, 'finish')['umaban'].iloc[0]

                    is_sakura = rid in self.sakura_ids
                    self.race_cache[rid] = {
                        'ps': ps, 'rf': rf, 'actual_top3': actual_top3,
                        'actual_1st': actual_1st, 'is_sakura': is_sakura,
                    }
                except Exception:
                    pass

        print(f'  Cached {len(self.race_cache)} races ({sum(1 for v in self.race_cache.values() if v["is_sakura"])} sakura)')

    def __call__(self, trial):
        params = {
            'pace_base_per_runner': trial.suggest_float('pace_base_per_runner', 0.10, 0.50),
            'pace_noise': trial.suggest_float('pace_noise', 0.05, 0.25),
            'bonus_nige':   trial.suggest_float('bonus_nige',   -0.05, +0.15),
            'bonus_senkou': trial.suggest_float('bonus_senkou', -0.05, +0.10),
            'bonus_sashi':  trial.suggest_float('bonus_sashi',  -0.10, +0.05),
            'bonus_oikomi': trial.suggest_float('bonus_oikomi', -0.15, +0.05),
            'gate_inner_senkou': trial.suggest_float('gate_inner_senkou', -0.05, +0.01),
            'gate_outer_sashi':  trial.suggest_float('gate_outer_sashi',  -0.01, +0.05),
            'noise_scale': trial.suggest_float('noise_scale', 0.005, 0.05),
        }

        total_score = 0.0
        total_weight = 0.0

        for rid, cache in self.race_cache.items():
            mc = qmc_sim_parameterized(
                cache['ps'], cache['rf'], params, n=self.n_sims
            )
            top5 = set(mc.head(5)['umaban'].astype(int))

            # スコア: TOP5と実際の3着内の重複数
            overlap = len(top5 & cache['actual_top3'])
            # 1着的中ボーナス
            win_bonus = 1.0 if int(mc.iloc[0]['umaban']) == cache['actual_1st'] else 0.0
            race_score = overlap + win_bonus * 0.5

            weight = self.sakura_weight if cache['is_sakura'] else 1.0
            total_score += race_score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='QMCコース係数ベイズ最適化')
    parser.add_argument('--trials', type=int, default=100, help='Optuna試行回数')
    parser.add_argument('--sakura-weight', type=float, default=3.0, help='桜花賞の重み')
    parser.add_argument('--sims', type=int, default=30000, help='QMCシミュレーション回数(最適化中)')
    args = parser.parse_args()

    t0 = time.time()
    print(f'=== QMC係数ベイズ最適化 ===')
    print(f'  Trials: {args.trials}, Sakura weight: {args.sakura_weight}, Sims: {args.sims}')

    # データロード
    print(f'\n[1] Loading feature cache...')
    feat = pd.read_pickle('data/features_v9b_2026.pkl')
    print(f'  {len(feat):,} rows')

    # バックテスト対象レース
    print(f'\n[2] Preparing backtest data...')
    hanshin_data, race_ids, sakura_ids = prepare_backtest_data(feat)

    # 目的関数準備（NN事前学習）
    print(f'\n[3] Pre-training NN models...')
    objective = QMCObjective(
        feat, hanshin_data, race_ids, sakura_ids,
        sakura_weight=args.sakura_weight, n_sims=args.sims,
    )

    t1 = time.time()
    print(f'  Prep done in {t1-t0:.0f}s')

    # 現在の係数でスコア計算
    print(f'\n[4] Scoring current (hand-tuned) params...')
    current_params = {
        'pace_base_per_runner': 0.30,
        'pace_noise': 0.15,
        'bonus_nige':   +0.06,
        'bonus_senkou': +0.02,
        'bonus_sashi':  -0.04,
        'bonus_oikomi': -0.04,
        'gate_inner_senkou': -0.010,
        'gate_outer_sashi':  +0.010,
        'noise_scale': 0.02,
    }
    current_score = 0.0
    total_w = 0.0
    for rid, cache in objective.race_cache.items():
        mc = qmc_sim_parameterized(cache['ps'], cache['rf'], current_params, n=args.sims)
        top5 = set(mc.head(5)['umaban'].astype(int))
        overlap = len(top5 & cache['actual_top3'])
        win_bonus = 1.0 if int(mc.iloc[0]['umaban']) == cache['actual_1st'] else 0.0
        w = args.sakura_weight if cache['is_sakura'] else 1.0
        current_score += (overlap + win_bonus * 0.5) * w
        total_w += w
    current_score /= total_w
    print(f'  Current score: {current_score:.4f}')

    # Optuna最適化
    print(f'\n[5] Running Bayesian optimization ({args.trials} trials)...')
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

    # 現在のパラメータをenqueue
    study.enqueue_trial(current_params)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    t2 = time.time()
    print(f'\n  Optimization done in {t2-t1:.0f}s')

    # 結果
    best = study.best_params
    best_score = study.best_value
    improvement = (best_score - current_score) / current_score * 100

    print(f'\n{"="*60}')
    print(f'  最適化結果')
    print(f'{"="*60}')
    print(f'  Before (手動): {current_score:.4f}')
    print(f'  After (最適):  {best_score:.4f} ({improvement:+.1f}%)')
    print(f'\n  最適パラメータ:')
    print(f"    'hanshin_turf_1600_outer': {{")
    print(f"        'name': '阪神芝1600m外回り (最適化済)',")
    print(f"        'straight': 473,")
    print(f"        'pace_base_per_runner': {best['pace_base_per_runner']:.4f},")
    print(f"        'pace_noise': {best['pace_noise']:.4f},")
    print(f"        'style_bonus': {{")
    print(f"            'nige':    {best['bonus_nige']:+.4f},")
    print(f"            'senkou':  {best['bonus_senkou']:+.4f},")
    print(f"            'sashi':   {best['bonus_sashi']:+.4f},")
    print(f"            'oikomi':  {best['bonus_oikomi']:+.4f},")
    print(f"        }},")
    print(f"        'gate_bias': {{")
    print(f"            'inner_senkou': {best['gate_inner_senkou']:+.5f},")
    print(f"            'outer_sashi':  {best['gate_outer_sashi']:+.5f},")
    print(f"            'inner_block':  +0.01000,")
    print(f"        }},")
    print(f"        'trouble_rate': 0.05,")
    print(f"        'trouble_penalty': 0.15,")
    print(f"        'noise_scale': {best['noise_scale']:.4f},")
    print(f"    }}")

    # 桜花賞のみのスコアも表示
    print(f'\n  === 桜花賞のみのスコア ===')
    for label, params in [('手動', current_params), ('最適化', best)]:
        sakura_scores = []
        for rid, cache in objective.race_cache.items():
            if not cache['is_sakura']:
                continue
            mc = qmc_sim_parameterized(cache['ps'], cache['rf'], params, n=args.sims)
            top5 = set(mc.head(5)['umaban'].astype(int))
            overlap = len(top5 & cache['actual_top3'])
            win_bonus = 1.0 if int(mc.iloc[0]['umaban']) == cache['actual_1st'] else 0.0

            year = cache['rf']['date'].iloc[0].year
            print(f'    {label} {year}: TOP5∩馬券内={overlap}/3 1着{"○" if win_bonus else "×"}')
            sakura_scores.append(overlap + win_bonus * 0.5)
        if sakura_scores:
            print(f'    {label} 平均: {np.mean(sakura_scores):.2f}')
        print()

    total = time.time() - t0
    print(f'  Total: {total:.0f}s')


if __name__ == '__main__':
    main()
