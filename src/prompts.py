"""
議長プロンプトテンプレート
v9b出力データを流し込んで、3エージェントディベート用のプロンプトを生成する
"""

import pandas as pd
import numpy as np


MODERATOR_PROMPT = """あなたはプロのクオンツファンドにおける「AI予測モデルの最終モデレーター（議長）」です。

以下のデータは、私の構築した競馬予測モデル（v9b）が弾き出した、最新の推論シミュレーション結果です。
NN（ニューラルネットワーク）とQMC（量子モンテカルロ）による出力が含まれています。

【レース情報】
{race_info}

【v9b出力データ（全馬）】
{horse_data}

■ ⚠️ 絶対厳守ルール
各エージェントおよび議長は、「上記で提供されたv9bの数値データ」のみを根拠として議論してください。外部知識の持ち込みは固く禁じます。

■ 召喚する3人のエージェントと【必須分析タスク】
1. 【Agent A：絶対能力派】
役割：タイム偏差値と末脚EMAを最重視する。
★必須タスク：NNの評価（μや勝率）が低くても、「タイム偏差値と末脚EMAの絶対値が突出している馬」を見つけ出し、ポテンシャル枠として必ず提示すること。

2. 【Agent B：舞台適性派】
役割：直線長コース適性、距離差などを重視する。
★必須タスク：NNの評価が高くても、「直線長適性が悪い（数値が大きい）」または「コース形態に合わない脚質」の馬を厳しく減点すること。

3. 【Agent C：展開・リスク派】
役割：不利確率、脚質分布、σ（不確実性）から展開を読み解く。
★必須タスク（偽物狩り）：v9bの予測順位（E[rank]や勝率）が上位であっても、「絶対指標（タイム・末脚）が平凡」かつ「ペース展開に泣くリスク」がある人気馬を見つけ、必ず論破すること。

■ ディベートの進行手順
【ステップ1：各エージェントの個別見解】
A、B、Cが順番に自身の「最上位推奨馬」と「危険視する上位候補」を発表。

【ステップ2：リスク検証と激しい反論（クロスチェック）】
Agent Cを中心に、AとBの推奨馬に対してデータを用いた徹底的な粗探しと反論を実施。

【ステップ3：議長（あなた）による最終結論】
3者の議論を統合し、安全な妥協案（合成の誤謬）を避け、以下の明確な役割を持たせた【最終推奨5頭】を決定してください。

* 【軸馬枠（1〜2頭）】：μ、タイム偏差、適性、展開リスクの全てで死角が少ない馬。
* 【アノマリー枠（必須）】：「σ（不確実性）が極めて大きい」＋「タイムや末脚の絶対値はトップクラス」＋「しかしNNのμは低評価（データ不足によるディスカウント）」という条件を満たす、隠れた怪物候補。
* 【大穴展開枠（必須）】：絶対能力は劣るが、Agent Cが予測したペース（例：スローの単騎逃げ、ハイペースの追込など）においてのみ、激走するオッズ妙味（EV）の高い馬。

クオンツ投資家が納得できる、論理的で詳細な選定レポートを出力してください。"""


def format_race_info(race_name, course_name, distance, heads, date_str, race_id=''):
    """レース情報の文字列を生成"""
    return (
        f"レース名: {race_name}\n"
        f"コース: {course_name}\n"
        f"距離: {distance}m\n"
        f"出走頭数: {heads}頭\n"
        f"日付: {date_str}\n"
        f"レースID: {race_id}"
    )


def format_horse_data(mc_results, race_features, nn_preds):
    """
    QMC結果 + 特徴量 + NN予測を統合して、プロンプト用のテキストテーブルを生成

    Parameters
    ----------
    mc_results : DataFrame  - qmc_sim()の出力（expected_rank順にソート済み）
    race_features : DataFrame - 出走馬の特徴量
    nn_preds : DataFrame - Predictor.predict()の出力

    Returns
    -------
    str : プロンプトに埋め込むテーブル文字列
    """
    lines = []

    # ヘッダー
    lines.append(
        f"{'順位':>4s} {'馬番':>4s} {'枠':>2s} {'馬名':18s} "
        f"{'odds':>7s} {'μ':>7s} {'σ':>7s} "
        f"{'勝率':>7s} {'複勝率':>7s} {'E[rank]':>8s} "
        f"{'EMA着順':>8s} {'タイムZ':>8s} {'末脚EMA':>8s} {'加重EMA':>8s} "
        f"{'直線適性':>8s} {'同距離':>8s} {'同馬場':>8s} {'距離差':>6s} "
        f"{'脚質':>6s} {'3角位':>6s} {'4角位':>6s} "
        f"{'馬体重':>6s} {'増減':>4s} {'間隔日':>6s} {'走数':>4s} "
        f"{'騎手勝':>7s} {'騎手複':>7s} {'師勝':>7s} {'師複':>7s} "
        f"{'前走CL':>6s} {'賞金log':>8s} {'勝率実':>7s} {'複勝実':>7s}"
    )
    lines.append("-" * 250)

    for rank, (_, r) in enumerate(mc_results.iterrows(), 1):
        u = int(r['umaban'])
        fd = race_features[race_features['umaban'] == u]
        ps = nn_preds[nn_preds['umaban'] == u]
        if len(fd) == 0 or len(ps) == 0:
            continue
        fd = fd.iloc[0]
        ps = ps.iloc[0]

        ls = f"{fd['long_stretch_avg']:8.2f}" if pd.notna(fd.get('long_stretch_avg')) else '     N/A'
        sd = f"{fd['same_dist_finish']:8.2f}" if pd.notna(fd.get('same_dist_finish')) else '     N/A'
        ss = f"{fd['same_surface_finish']:8.2f}" if pd.notna(fd.get('same_surface_finish')) else '     N/A'

        lines.append(
            f"{rank:4d} {u:4d} {int(fd['wakuban']):2d} {r['horse_name']:18s} "
            f"{r['odds']:7.1f} {ps['mu']:7.4f} {ps['sigma']:7.4f} "
            f"{r['win_prob']:7.2%} {r['top3_prob']:7.2%} {r['expected_rank']:8.2f} "
            f"{fd['ema_finish']:8.3f} {fd['ema_time_zscore']:8.3f} {fd['ema_agari']:8.3f} {fd['weighted_ema_finish']:8.3f} "
            f"{ls} {sd} {ss} {fd['prev_dist_diff']:6.0f} "
            f"{fd['avg_run_style']:6.2f} {fd['avg_jyuni_3c']:6.2f} {fd['avg_jyuni_4c']:6.2f} "
            f"{fd['bataijyu']:6.0f} {fd['zogen_sa']:4.0f} {fd['interval_days']:6.0f} {fd['past_count']:4.0f} "
            f"{fd['jockey_win_rate']:7.3f} {fd['jockey_top3_rate']:7.3f} "
            f"{fd['trainer_win_rate']:7.3f} {fd['trainer_top3_rate']:7.3f} "
            f"{fd['prev_race_class']:6.0f} {fd['log_prize_money']:8.3f} "
            f"{fd['win_rate']:7.3f} {fd['top3_rate']:7.3f}"
        )

    return "\n".join(lines)


def build_prompt(race_name, course_name, distance, heads, date_str,
                 mc_results, race_features, nn_preds, race_id=''):
    """
    3層アーキテクチャの最終出力: 議長プロンプトを完成させる

    使い方:
        from src.predictor import Predictor
        from src.qmc_courses import qmc_sim
        from src.prompts import build_prompt

        pred = Predictor()
        pred.train(train_data)
        nn_preds = pred.predict(race_data)
        mc = qmc_sim(nn_preds, race_features=race_data, course='nakayama_turf_1600')

        prompt = build_prompt(
            race_name='ニュージーランドトロフィー G2',
            course_name='中山芝1600m',
            distance=1600,
            heads=len(race_data),
            date_str='2026-04-12',
            mc_results=mc,
            race_features=race_data,
            nn_preds=nn_preds,
        )
        print(prompt)  # → これをLLMに渡す
    """
    race_info = format_race_info(race_name, course_name, distance, heads, date_str, race_id)
    horse_data = format_horse_data(mc_results, race_features, nn_preds)
    return MODERATOR_PROMPT.format(race_info=race_info, horse_data=horse_data)
