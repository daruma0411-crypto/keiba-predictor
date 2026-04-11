"""
コース別QMCシミュレーション
各コースの物理特性（直線長、回り、枠順バイアス）を反映した係数でMCを実行する
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc, norm


# ============================================================
# コースプロファイル定義
# ============================================================
# 各係数の意味:
#   pace_base_per_runner: 逃げ馬1頭増えるごとのペース上昇係数
#   pace_noise: ペースのランダム変動幅
#   style_bonus:  脚質別のハイペース時補正 (逃げ, 先行, 差し, 追込)
#                 正=ハイペースで有利, 負=ハイペースで不利
#   gate_bias:    枠順バイアス (inner_senkou, outer_sashi, inner_block)
#                 inner_senkou: 内枠×先行の有利度
#                 outer_sashi:  外枠×追込の不利度
#                 inner_block:  内枠×差しの包まれリスク
#   trouble_rate: 出遅れ・不利の発生確率
#   noise_scale:  個体ノイズの大きさ

COURSE_PROFILES = {
    # --- 中山 芝1600m (NZT等) ---
    # 直線310m: 短い→逃げ先行が残りやすい
    # 内回り・小回り→内枠先行が圧倒的有利
    # 差し追込は物理的に届きにくい
    'nakayama_turf_1600': {
        'name': '中山芝1600m',
        'straight': 310,
        'pace_base_per_runner': 0.25,   # 桜花賞(0.30)より低い→ペース上がりにくい
        'pace_noise': 0.12,             # 桜花賞(0.15)より低い→ペース安定
        'style_bonus': {
            'nige':    +0.10,  # 逃げ: 桜花賞+0.06→大幅強化。直線短く残りやすい
            'senkou':  +0.05,  # 先行: 桜花賞+0.02→強化
            'sashi':   -0.02,  # 差し: 桜花賞-0.04→やや緩和(1600mなのでまだ届く)
            'oikomi':  -0.07,  # 追込: 新設。直線310mでは致命的に届かない
        },
        'gate_bias': {
            'inner_senkou': -0.025,  # 内枠先行の有利(桜花賞-0.01→大幅強化)
            'outer_sashi':  +0.020,  # 外枠追込の不利(桜花賞+0.01→強化)
            'inner_block':  +0.005,  # 内枠差しの包まれ(桜花賞の半分→小回りで逃げやすい)
        },
        'trouble_rate': 0.05,
        'trouble_penalty': 0.15,
        'noise_scale': 0.02,
    },

    # --- 阪神 芝1600m外回り (阪神牝馬S, 桜花賞等) ---
    # 直線473m: 長い→差し追込が届く
    # 外回り→枠順バイアス小さい
    # 現行桜花賞QMCとほぼ同じ係数
    'hanshin_turf_1600_outer': {
        'name': '阪神芝1600m外回り',
        'straight': 473,
        'pace_base_per_runner': 0.30,
        'pace_noise': 0.15,
        'style_bonus': {
            'nige':    +0.06,
            'senkou':  +0.02,
            'sashi':   -0.04,
            'oikomi':  -0.04,  # 桜花賞と同じ。直線長く追込も効く
        },
        'gate_bias': {
            'inner_senkou': -0.010,
            'outer_sashi':  +0.010,
            'inner_block':  +0.010,
        },
        'trouble_rate': 0.05,
        'trouble_penalty': 0.15,
        'noise_scale': 0.02,
    },
}


# ============================================================
# 汎用QMCシミュレーション
# ============================================================
def qmc_sim(preds, race_features=None, course='hanshin_turf_1600_outer', n=100000):
    """
    コースプロファイルに基づくQMCシミュレーション

    Parameters
    ----------
    preds : DataFrame
        NNの予測結果 (mu, sigma, horse_name, umaban, odds)
    race_features : DataFrame
        出走馬の特徴量 (avg_run_style, wakuban等)
    course : str
        COURSE_PROFILESのキー
    n : int
        シミュレーション回数

    Returns
    -------
    DataFrame
        各馬の勝率・複勝率・期待順位等
    """
    prof = COURSE_PROFILES[course]
    rf = race_features

    nh = len(preds)
    mu = preds['mu'].values.copy()
    sig = preds['sigma'].values.copy()

    # 脚質・枠番の取得
    rs = np.full(nh, 2.5)  # デフォルト: 差し
    wk = np.arange(1, nh + 1)
    if rf is not None and len(rf) == nh:
        if 'avg_run_style' in rf.columns:
            r = rf['avg_run_style'].values
            rs = np.where(np.isnan(r), 2.5, r)
        if 'wakuban' in rf.columns:
            wk = rf['wakuban'].values.astype(float)

    # --- Sobol列でサンプリング ---
    nn_nige = np.sum(rs <= 1.5)
    pace_base = (nn_nige - 1.5) * prof['pace_base_per_runner']

    nd = nh * 4 + 1
    sampler = qmc.Sobol(d=nd, scramble=True, seed=42)
    np2 = 2 ** int(np.ceil(np.log2(n)))
    sb = sampler.random(np2)[:n]
    sn = norm.ppf(np.clip(sb, 0.001, 0.999))

    j = 0
    # 基本能力分布
    ability = mu[np.newaxis, :] + sig[np.newaxis, :] * sn[:, j:j+nh]
    j += nh

    # ペース変動
    pace = pace_base + prof['pace_noise'] * sn[:, j]
    j += 1

    # 不利乱数
    luck_u = sb[:, j:j+nh]
    j += nh
    block_u = sb[:, j:j+nh]
    j += nh

    # 個体ノイズ
    indiv_noise = sn[:, j:j+nh]
    j += nh

    pace2 = pace[:, np.newaxis]

    # --- 脚質分類 ---
    is_nige   = (rs <= 1.5).astype(float)
    is_senkou = ((rs > 1.5) & (rs <= 2.5)).astype(float)
    is_sashi  = ((rs > 2.5) & (rs < 3.5)).astype(float)
    is_oikomi = (rs >= 3.5).astype(float)

    # --- 脚質×ペース補正 ---
    sb_ = prof['style_bonus']
    ability += pace2 * sb_['nige']   * is_nige[np.newaxis, :]
    ability += pace2 * sb_['senkou'] * is_senkou[np.newaxis, :]
    ability += pace2 * sb_['sashi']  * is_sashi[np.newaxis, :]
    ability += pace2 * sb_['oikomi'] * is_oikomi[np.newaxis, :]

    # --- 枠順バイアス ---
    gb = prof['gate_bias']
    # 内枠(1-3)×先行以前 → 有利
    inner_senkou = ((wk <= 3) & (rs <= 2.5)).astype(float)
    ability += gb['inner_senkou'] * inner_senkou[np.newaxis, :]

    # 外枠(6+)×追込 → 不利
    outer_sashi = ((wk >= 6) & (rs >= 3.5)).astype(float)
    ability += gb['outer_sashi'] * outer_sashi[np.newaxis, :]

    # 内枠×差し → 包まれリスク
    inner_block_flag = ((wk <= 3) & (rs >= 3.0)).astype(float)
    ability += (block_u < 0.10).astype(float) * inner_block_flag[np.newaxis, :] * gb['inner_block']

    # --- 出遅れ・不利 ---
    ability += (luck_u < prof['trouble_rate']).astype(float) * prof['trouble_penalty']

    # --- 個体ノイズ ---
    ability += prof['noise_scale'] * indiv_noise

    # --- 順位算出 ---
    ranks = np.argsort(np.argsort(ability, axis=1), axis=1) + 1

    # --- 結果集計 ---
    res = preds[['horse_name', 'umaban', 'odds']].copy()
    for pos in range(1, min(nh + 1, 19)):
        res[f'prob_{pos}'] = (ranks == pos).mean(axis=0)
    res['expected_rank'] = ranks.mean(axis=0)
    res['win_prob'] = res['prob_1']
    res['top3_prob'] = res[['prob_1', 'prob_2', 'prob_3']].sum(axis=1)
    if 'odds' in res.columns:
        res['ev_win'] = res['win_prob'] * res['odds']

    return res.sort_values('expected_rank')


def list_courses():
    """利用可能なコースプロファイル一覧"""
    for key, prof in COURSE_PROFILES.items():
        print(f"  {key:35s} → {prof['name']} (直線{prof['straight']}m)")
