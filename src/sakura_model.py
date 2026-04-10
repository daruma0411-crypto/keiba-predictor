"""
桜花賞特化型予測モデル
過去10年の好走パターンをスコア化して予測に加算する
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_sakura_score(race_feat, df_all, um_data):
    """
    桜花賞出走馬に対して、過去の好走パターンに基づくスコアを計算する

    Parameters
    ----------
    race_feat : pd.DataFrame
        桜花賞出走馬の特徴量（ketto_num, horse_name, umaban等を含む）
    df_all : pd.DataFrame
        全レースデータ（前走情報取得用）
    um_data : pd.DataFrame
        UM_DATA（血統情報、ketto_numでindexed）

    Returns
    -------
    pd.DataFrame
        各馬のsakura_score等を含むDataFrame
    """
    race_date = race_feat['date'].iloc[0]
    scores = []

    for _, row in race_feat.iterrows():
        horse_id = row['ketto_num']
        horse_name = row['horse_name']
        umaban = row['umaban']
        score = 0.0
        reasons = []

        # === 1. 前走パターン ===
        horse_history = df_all[
            (df_all['ketto_num'] == horse_id) &
            (df_all['date'] < race_date) &
            (df_all['kakutei_jyuni'] > 0)
        ].sort_values('date')

        if len(horse_history) > 0:
            prev = horse_history.iloc[-1]
            prev_race = prev.get('race_name', '')
            prev_finish = prev.get('kakutei_jyuni', 99)
            prev_kyori = prev.get('kyori', 0)

            # 前走がチューリップ賞
            if 'チューリップ' in str(prev_race):
                score += 3.0
                if prev_finish <= 3:
                    score += 2.0
                    reasons.append(f'前走チューリップ賞{int(prev_finish)}着')
                else:
                    reasons.append(f'前走チューリップ賞{int(prev_finish)}着')

            # 前走が阪神JF
            elif 'ジュベナイル' in str(prev_race) or 'ＪＦ' in str(prev_race):
                score += 4.0
                if prev_finish <= 2:
                    score += 3.0
                    reasons.append(f'前走阪神JF{int(prev_finish)}着')
                else:
                    reasons.append(f'前走阪神JF{int(prev_finish)}着')

            # 前走がフィリーズレビュー
            elif 'フィリーズ' in str(prev_race):
                score += 1.5
                if prev_finish <= 2:
                    score += 1.5
                    reasons.append(f'前走フィリーズR{int(prev_finish)}着')

            # 前走がクイーンC
            elif 'クイーン' in str(prev_race):
                score += 2.0
                if prev_finish <= 2:
                    score += 2.0
                    reasons.append(f'前走クイーンC{int(prev_finish)}着')

            # 前走が1600m重賞で好走
            elif prev_kyori == 1600 and prev_finish <= 3:
                score += 2.0
                reasons.append(f'前走1600m{int(prev_finish)}着')

            # 前走3着以内ボーナス（ローテ問わず）
            if prev_finish <= 3:
                score += 1.0

            # 前走大敗ペナルティ
            if prev_finish >= 10:
                score -= 3.0
                reasons.append(f'前走{int(prev_finish)}着大敗')

        else:
            score -= 2.0
            reasons.append('過去走なし')

        # === 2. 血統パターン ===
        if horse_id in um_data.index:
            um_row = um_data.loc[horse_id]
            sire = um_row.get('sire_name', '')
            bms = um_row.get('bms_name', '')
            breeder = um_row.get('breeder_name', '')

            # 桜花賞好走種牡馬
            sakura_sires = {
                'ディープインパクト': 3.0, 'ダイワメジャー': 2.5,
                'キズナ': 1.5, 'キングカメハメハ': 1.5,
                'キタサンブラック': 1.5, 'ドゥラメンテ': 1.5,
                'エピファネイア': 1.5, 'ハーツクライ': 1.0,
                'ロードカナロア': 1.0, 'モーリス': 1.0,
            }
            if sire in sakura_sires:
                score += sakura_sires[sire]
                reasons.append(f'種牡馬{sire}')

            # ノーザンファーム
            if 'ノーザン' in str(breeder):
                score += 2.0
                reasons.append('ノーザンF')
            elif '社台' in str(breeder):
                score += 1.0
                reasons.append('社台系')

        # === 3. 騎手パターン ===
        jockey = row.get('kisyu_name', '')
        sakura_jockeys = {
            'ルメール': 2.0, '川田将雅': 1.5, '武豊': 1.5,
            '池添謙一': 1.5, 'Ｍ．デム': 1.5, '福永祐一': 1.0,
            'モレイラ': 1.5, '北村友一': 1.0, '戸崎圭太': 1.0,
        }
        # 騎手名が部分一致でもOK
        for jname, jscore in sakura_jockeys.items():
            if jname in str(jockey):
                score += jscore
                reasons.append(f'騎手{jockey}')
                break

        # === 4. 基本能力 ===
        # NNからのema_finish（小さいほど強い）
        ema_f = row.get('ema_finish', 10)
        if pd.notna(ema_f):
            if ema_f <= 3:
                score += 3.0
                reasons.append(f'過去走平均{ema_f:.1f}着')
            elif ema_f <= 5:
                score += 1.5
            elif ema_f >= 10:
                score -= 2.0
                reasons.append(f'過去走平均{ema_f:.1f}着(弱)')

        # 勝率
        wr = row.get('win_rate', 0)
        if pd.notna(wr) and wr > 0:
            score += wr * 5.0

        scores.append({
            'ketto_num': horse_id,
            'horse_name': horse_name,
            'umaban': umaban,
            'sakura_score': score,
            'reasons': '; '.join(reasons) if reasons else 'なし',
        })

    return pd.DataFrame(scores)


def combined_prediction(nn_preds, sakura_scores, nn_weight=0.5, sakura_weight=0.5):
    """
    NNの予測と桜花賞スコアを組み合わせて最終予測を出す

    Parameters
    ----------
    nn_preds : pd.DataFrame
        RacePredictor.predict()の出力（mu, sigma, horse_name, umaban）
    sakura_scores : pd.DataFrame
        compute_sakura_score()の出力（sakura_score）
    nn_weight : float
        NNスコアの重み
    sakura_weight : float
        桜花賞スコアの重み

    Returns
    -------
    pd.DataFrame
        combined_score順にソートされた予測結果
    """
    merged = nn_preds.merge(sakura_scores[['umaban', 'sakura_score', 'reasons']],
                            on='umaban', how='left')

    # NNのmuを正規化（0-1、小さいほど強い → 反転）
    mu_min = merged['mu'].min()
    mu_max = merged['mu'].max()
    if mu_max > mu_min:
        merged['nn_score'] = 1 - (merged['mu'] - mu_min) / (mu_max - mu_min)
    else:
        merged['nn_score'] = 0.5

    # sakura_scoreを正規化（0-1）
    ss_min = merged['sakura_score'].min()
    ss_max = merged['sakura_score'].max()
    if ss_max > ss_min:
        merged['sakura_norm'] = (merged['sakura_score'] - ss_min) / (ss_max - ss_min)
    else:
        merged['sakura_norm'] = 0.5

    # 合成スコア
    merged['combined_score'] = (
        nn_weight * merged['nn_score'] +
        sakura_weight * merged['sakura_norm']
    )

    return merged.sort_values('combined_score', ascending=False)
