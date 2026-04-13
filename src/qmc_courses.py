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

    # --- 中山 芝1200m ---
    # 直線310m: 短い→逃げ先行有利
    # 内回り急坂→最後の急坂で前が止まることも
    # スプリント→ペース速い
    'nakayama_turf_1200': {
        'name': '中山芝1200m',
        'straight': 310,
        'pace_base_per_runner': 0.22,
        'pace_noise': 0.12,
        'style_bonus': {
            'nige':    +0.10,
            'senkou':  +0.05,
            'sashi':   -0.03,
            'oikomi':  -0.08,
        },
        'gate_bias': {
            'inner_senkou': -0.025,
            'outer_sashi':  +0.020,
            'inner_block':  +0.005,
        },
        'trouble_rate': 0.06,
        'trouble_penalty': 0.17,
        'noise_scale': 0.025,
    },

    # --- 福島 芝1200m ---
    # 直線292m: JRA最短→逃げ先行が圧倒的有利
    # 小回り平坦コース→内枠先行が非常に有利
    # 1200mスプリント→ペース緩みにくく前残り傾向
    'fukushima_turf_1200': {
        'name': '福島芝1200m',
        'straight': 292,
        'pace_base_per_runner': 0.20,   # スプリントなので逃げ馬数の影響は小さめ
        'pace_noise': 0.10,             # 1200mはペース安定
        'style_bonus': {
            'nige':    +0.12,  # 直線最短→逃げ超有利
            'senkou':  +0.06,  # 先行も有利
            'sashi':   -0.04,  # 差しは届きにくい
            'oikomi':  -0.10,  # 追込は致命的
        },
        'gate_bias': {
            'inner_senkou': -0.030,  # 内枠先行→非常に有利（小回り）
            'outer_sashi':  +0.025,  # 外枠追込→非常に不利
            'inner_block':  +0.003,  # 1200mなのでブロックリスク低い
        },
        'trouble_rate': 0.06,       # スプリントは出遅れ影響大
        'trouble_penalty': 0.18,    # 出遅れペナルティ大きめ
        'noise_scale': 0.025,       # スプリントはバラつきやや大
    },

    # --- 阪神 芝2000m内回り (忘れな草賞, 大阪杯等) ---
    # 直線356m: 中山より長いが東京より短い
    # 内回り・急坂あり → 先行有利だが差しも届く
    # 2000mは中距離→ペース読みが重要
    'hanshin_turf_2000': {
        'name': '阪神芝2000m内回り',
        'straight': 356,
        'pace_base_per_runner': 0.25,
        'pace_noise': 0.15,
        'style_bonus': {
            'nige':    +0.08,
            'senkou':  +0.04,
            'sashi':   -0.01,
            'oikomi':  -0.05,
        },
        'gate_bias': {
            'inner_senkou': -0.020,
            'outer_sashi':  +0.015,
            'inner_block':  +0.008,
        },
        'trouble_rate': 0.05,
        'trouble_penalty': 0.15,
        'noise_scale': 0.02,
    },

    # --- 阪神 芝1600m外回り (阪神牝馬S, 桜花賞等) ---
    # 直線473m: 長い→差し追込が届く
    # 外回り→枠順バイアス小さい
    # 2026-04-11 ベイズ最適化済 (Optuna 100trials, 桜花賞weight=3.0)
    'hanshin_turf_1600_outer': {
        'name': '阪神芝1600m外回り (最適化済)',
        'straight': 473,
        'pace_base_per_runner': 0.3674,
        'pace_noise': 0.1988,
        'style_bonus': {
            'nige':    +0.0846,
            'senkou':  +0.0535,
            'sashi':   +0.0183,  # 最適化で差しペナ消滅→直線473mで差しも届く
            'oikomi':  -0.0610,
        },
        'gate_bias': {
            'inner_senkou': -0.00251,
            'outer_sashi':  -0.00359,  # 最適化で外枠追込ペナ消滅
            'inner_block':  +0.010,
        },
        'trouble_rate': 0.05,
        'trouble_penalty': 0.15,
        'noise_scale': 0.0475,  # 最適化で波乱許容度UP
    },
}


# ============================================================
# 汎用QMCシミュレーション
# ============================================================
def qmc_sim(preds, race_features=None, course='hanshin_turf_1600_outer', n=100000):
    """
    コースプロファイルに基づくQMCシミュレーション（ペース寄与分離モデル）

    ペースモデル:
      - 前半ペース: 逃げ馬の頭数と競り合いの激しさで決まる
      - 後半ペース: 前半ペースの反動（前半速い→後半遅い）
      - 各馬の消耗: 脚質とペースの関係で個別に計算
        - 逃げ馬がハイペースを作る → 逃げ馬自身が最も消耗
        - 先行馬は巻き込まれ度合いで消耗
        - 差し追込は前半温存、後半で加速

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
    nn_senkou = np.sum((rs > 1.5) & (rs <= 2.5))

    # 次元: 各馬能力(nh) + 前半ペース(1) + 後半ペース(1) + 逃げ競り(1)
    #       + 出遅れ(nh) + 包まれ(nh) + 個体ノイズ(nh)
    nd = nh * 4 + 3
    sampler = qmc.Sobol(d=nd, scramble=True, seed=42)
    np2 = 2 ** int(np.ceil(np.log2(n)))
    sb = sampler.random(np2)[:n]
    sn = norm.ppf(np.clip(sb, 0.001, 0.999))

    j = 0
    # 基本能力分布
    ability = mu[np.newaxis, :] + sig[np.newaxis, :] * sn[:, j:j+nh]
    j += nh

    # === ペース寄与分離モデル ===
    # 前半ペース: 逃げ馬の数で基本ペースが決まり、乱数で変動
    pace_front = (nn_nige - 1.5) * prof['pace_base_per_runner'] + prof['pace_noise'] * sn[:, j]
    j += 1

    # 逃げ馬同士の競り合い強度 (逃げ2頭以上で発動)
    rivalry = sn[:, j] * 0.5  # -0.5〜+0.5の範囲で競り合いの激しさが変動
    j += 1
    if nn_nige >= 2:
        rivalry_intensity = np.clip(rivalry, 0, None)  # 正の時だけ競り合い激化
    else:
        rivalry_intensity = np.zeros(n)  # 単騎逃げなら競り合いなし

    # 後半ペース: 前半の反動（前半速い→後半で前の馬が止まる）
    pace_rear = -pace_front * 0.6 + prof['pace_noise'] * 0.5 * sn[:, j]
    j += 1

    # 不利乱数
    luck_u = sb[:, j:j+nh]
    j += nh
    block_u = sb[:, j:j+nh]
    j += nh

    # 個体ノイズ
    indiv_noise = sn[:, j:j+nh]
    j += nh

    # --- 脚質分類 ---
    is_nige   = (rs <= 1.5).astype(float)
    is_senkou = ((rs > 1.5) & (rs <= 2.5)).astype(float)
    is_sashi  = ((rs > 2.5) & (rs < 3.5)).astype(float)
    is_oikomi = (rs >= 3.5).astype(float)

    pf = pace_front[:, np.newaxis]  # (n, 1)
    pr = pace_rear[:, np.newaxis]
    ri = rivalry_intensity[:, np.newaxis]

    sb_ = prof['style_bonus']
    straight_ratio = prof['straight'] / 1600.0  # 直線比率（長いほど差しに有利）

    # --- 馬ごとのペース影響（分離モデル） ---

    # 逃げ馬: 前半ペースを作る側 → 自身が最も消耗
    # ハイペース(pf>0)で逃げると消耗大、スロー(pf<0)なら楽逃げ
    # 競り合い(ri>0)があるとさらに消耗
    nige_effect = (
        pf * sb_['nige']           # ペース効果（旧来）
        - np.abs(pf) * 0.03 * is_nige[np.newaxis, :]    # 逃げ馬はペースが極端なほど消耗
        - ri * 0.08 * is_nige[np.newaxis, :]              # 競り合い消耗（逃げ馬のみ）
    )

    # 先行馬: 逃げ馬のペースに巻き込まれるが、逃げほどではない
    senkou_effect = (
        pf * sb_['senkou']
        - ri * 0.03 * is_senkou[np.newaxis, :]  # 競り合いの巻き込まれ（逃げの半分以下）
    )

    # 差し馬: 前半は温存、後半で加速
    # ハイペース(pf>0)→後半(pr<0で前が止まる)→差し有利
    sashi_effect = (
        pr * sb_['sashi'] * (-1)    # 後半ペース反動が大きいほど差し有利
        + pf * 0.02 * straight_ratio * is_sashi[np.newaxis, :]  # ハイペース×直線長→差し有利
    )

    # 追込馬: 差しよりさらに後半依存
    oikomi_effect = (
        pr * sb_['oikomi'] * (-1)
        + pf * 0.03 * straight_ratio * is_oikomi[np.newaxis, :]
        - (1 - straight_ratio) * 0.02 * is_oikomi[np.newaxis, :]  # 直線短いと届かない
    )

    ability += nige_effect * is_nige[np.newaxis, :]
    ability += senkou_effect * is_senkou[np.newaxis, :]
    ability += sashi_effect * is_sashi[np.newaxis, :]
    ability += oikomi_effect * is_oikomi[np.newaxis, :]

    # --- 枠順バイアス ---
    gb = prof['gate_bias']
    inner_senkou = ((wk <= 3) & (rs <= 2.5)).astype(float)
    ability += gb['inner_senkou'] * inner_senkou[np.newaxis, :]

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
