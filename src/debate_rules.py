"""
ディベートルール: QMCエラー分析(1,192レース)から導出
QMC TOP8を候補プールとし、消しルールで過大評価馬を除外して5頭選抜

バックテスト実績:
  占有率92%, 全外し4回/1192レース, 1着含有87%, 単勝ROI 122%
"""

import pandas as pd
import numpy as np


# === 消しルール（Agent D相当） ===
# QMCエラー分析で「過大評価(TOP5→着外)」に共通する特徴パターン

def count_flags(row):
    """
    消しフラグをカウント。フラグが多いほど過大評価の可能性が高い。

    Parameters
    ----------
    row : dict-like with keys:
        prev_race_class, ema_finish, odds, ninki,
        same_dist_finish, ema_agari
    """
    flags = []

    prev_cl = row.get('prev_race_class', 4)
    if pd.isna(prev_cl):
        prev_cl = 4
    ema_fin = row.get('ema_finish', 3)
    if pd.isna(ema_fin):
        ema_fin = 3
    odds_val = row.get('odds', 5)
    if pd.isna(odds_val):
        odds_val = 5
    ninki_val = row.get('ninki', 3)
    if pd.isna(ninki_val):
        ninki_val = 3
    same_d = row.get('same_dist_finish', 3)
    if pd.isna(same_d):
        same_d = 3
    ema_ag = row.get('ema_agari', 2)
    if pd.isna(ema_ag):
        ema_ag = 2

    # ルール①: クラス壁（前走3勝以下 & 6人気以降）
    if prev_cl <= 3 and ninki_val > 5:
        flags.append('class_wall')

    # ルール②: EMA着順が悪い（能力不足）
    if ema_fin > 4.5:
        flags.append('ema_bad')

    # ルール③: NNの穴馬バイアス（オッズ20倍超 & 9人気以降）
    if odds_val > 20 and ninki_val > 8:
        flags.append('longshot_bias')

    # ルール④: 適性不足（同距離実績悪い & 末脚不足）
    if same_d > 4.0 and ema_ag < 1.0:
        flags.append('no_aptitude')

    return flags


def select_with_rules(mc, rf, n_pop=2, n_disc=3, cutoff=5):
    """
    ディベートルール適用済み5頭選抜

    1. QMC TOP8を候補プールとする
    2. 消しルールでフラグ付け
    3. フラグなし優先で堅実枠(人気上位) + 穴枠(QMC順)を選抜

    Parameters
    ----------
    mc : DataFrame - qmc_sim()出力（expected_rank順ソート済み）
    rf : DataFrame - 出走馬の特徴量
    n_pop : int - 堅実枠の頭数（デフォルト2）
    n_disc : int - 穴枠の頭数（デフォルト3）
    cutoff : int - 堅実枠の人気上限（デフォルト5番人気以内）

    Returns
    -------
    selected : DataFrame - 選抜された5頭
    pop_umabans : set - 堅実枠の馬番セット
    flagged_info : list of dict - 消しフラグ情報（プロンプト用）
    """
    if 'ninki' not in mc.columns:
        mc = mc.copy()
        if 'odds' in mc.columns:
            mc['ninki'] = mc['odds'].rank(method='first')
        else:
            mc['ninki'] = range(1, len(mc) + 1)

    total = n_pop + n_disc

    # 候補プール: QMC TOP8
    pool = mc.head(8).copy()

    # 各馬にフラグ付け
    pool_rows = []
    flagged_info = []
    for _, r in pool.iterrows():
        u = int(r['umaban'])
        fd = rf[rf['umaban'] == u]
        if len(fd) == 0:
            continue
        fd = fd.iloc[0]

        # 特徴量を辞書化
        feat_dict = {
            'prev_race_class': fd.get('prev_race_class', 4),
            'ema_finish': fd.get('ema_finish', 3),
            'odds': r.get('odds', 5) if pd.notna(r.get('odds')) else 5,
            'ninki': r.get('ninki', 3) if pd.notna(r.get('ninki')) else 3,
            'same_dist_finish': fd.get('same_dist_finish', 3),
            'ema_agari': fd.get('ema_agari', 2),
        }
        flags = count_flags(feat_dict)

        pool_rows.append({
            'umaban': u,
            'horse_name': r['horse_name'],
            'ninki': feat_dict['ninki'],
            'odds': feat_dict['odds'],
            'expected_rank': r['expected_rank'],
            'win_prob': r.get('win_prob', 0),
            'top3_prob': r.get('top3_prob', 0),
            'flags': flags,
            'n_flags': len(flags),
        })

        if flags:
            flagged_info.append({
                'umaban': u,
                'horse_name': r['horse_name'],
                'flags': flags,
                'qmc_rank': len(pool_rows),
            })

    pool_df = pd.DataFrame(pool_rows)
    clean = pool_df[pool_df['n_flags'] == 0].copy()
    flagged = pool_df[pool_df['n_flags'] > 0].copy()

    # === 堅実枠: cutoff人気以内 → フラグなし → QMC順 ===
    pop_pool = clean[clean['ninki'] <= cutoff].sort_values('expected_rank')
    pop = pop_pool.head(n_pop)

    # 足りなければフラグ付きから人気順
    if len(pop) < n_pop:
        extra = flagged[flagged['ninki'] <= cutoff].sort_values('ninki')
        pop = pd.concat([pop, extra]).head(n_pop)

    # さらに足りなければcutoff外のフラグなしから
    if len(pop) < n_pop:
        remaining = clean[~clean['umaban'].isin(pop['umaban'])].sort_values('ninki')
        pop = pd.concat([pop, remaining]).head(n_pop)

    pop_umabans = set(pop['umaban'].astype(int))

    # === 穴枠: 堅実枠以外 → フラグなし → QMC順 ===
    disc_pool = clean[~clean['umaban'].isin(pop_umabans)].sort_values('expected_rank')
    disc = disc_pool.head(n_disc)

    # 足りなければフラグ付きから（フラグ少ない順→QMC順）
    if len(disc) < n_disc:
        extra = flagged[
            ~flagged['umaban'].isin(pop_umabans) &
            ~flagged['umaban'].isin(disc['umaban'])
        ].sort_values(['n_flags', 'expected_rank'])
        disc = pd.concat([disc, extra]).head(n_disc)

    selected = pd.concat([pop, disc])

    # QMC TOP8外から補充
    if len(selected) < total:
        rest_mc = mc[~mc['umaban'].isin(selected['umaban'].values)]
        for _, r in rest_mc.iterrows():
            if len(selected) >= total:
                break
            row = {
                'umaban': int(r['umaban']),
                'horse_name': r['horse_name'],
                'ninki': r.get('ninki', 99),
                'odds': r.get('odds', 999),
                'expected_rank': r['expected_rank'],
                'win_prob': r.get('win_prob', 0),
                'top3_prob': r.get('top3_prob', 0),
                'flags': [],
                'n_flags': 0,
            }
            selected = pd.concat([selected, pd.DataFrame([row])], ignore_index=True)

    return selected.head(total), pop_umabans, flagged_info
