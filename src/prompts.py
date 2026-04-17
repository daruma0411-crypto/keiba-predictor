"""
議長プロンプトテンプレート
v9b出力データを流し込んで、3エージェントディベート用のプロンプトを生成する
"""

import pandas as pd
import numpy as np


MODERATOR_PROMPT = """あなたはプロのクオンツファンドにおける「AI予測モデルの最終モデレーター（議長）」です。

以下のデータは、私の構築した競馬予測モデル（v2/D構成: 4層NN+ListNet, 40k学習, 80epoch）が弾き出した、最新の推論シミュレーション結果です。
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
【ステップ0：全馬スクリーニング（必須・省略厳禁）】
ディベート開始前に、A・B・Cの3エージェントがそれぞれ**全馬**を自分の基準で評価し、以下の表を作成すること。
QMC順位に関係なく、全馬を漏れなくチェックすること。特にQMC下位の馬こそ丁寧に見ること。

| 馬番 | 馬名 | A評価 | B評価 | C評価 | 注目理由(あれば) |

評価基準:
- A(絶対能力): タイムZ・上がり3F EMA・着差EMAの絶対値で ◎/○/△/× の4段階
- B(舞台適性): 距離差・同距離着順・脚質×コース形態で ◎/○/△/× の4段階
- C(展開リスク): 脚質×想定ペース・σ・不利確率で ◎/○/△/× の4段階

★この表で、QMC下位なのにいずれかのエージェントが◎をつけた馬は「要注目馬」として必ずステップ1以降で議論すること。

【ステップ1：各エージェントの個別見解】
A、B、Cが順番に自身の「最上位推奨馬」と「危険視する上位候補」を発表。
ステップ0の「要注目馬」にも必ず言及すること。

【ステップ2：リスク検証と激しい反論（クロスチェック）】
Agent Cを中心に、AとBの推奨馬に対してデータを用いた徹底的な粗探しと反論を実施。

【ステップ3：議長（あなた）による最終結論】
3者の議論を統合し、安全な妥協案（合成の誤謬）を避け、最終推奨を決定してください。

■ Layer1-2が2系統の選抜を出力済みです。

【系統A：占有率重視（V1従来）】QMC上位5頭
{line_a_info}
→ 10年バックテスト: 占有率80%, 単勝ROI 74.8%

【系統B：ROI重視（堅2穴3）】★本命(5人気以内QMC上位2頭) + ☆穴(6人気以降QMC上位3頭)
{line_b_info}
→ 10年バックテスト: 占有率50%, 単勝ROI 139.6%, 馬連ROI 56.9%

■ 議長の仕事
1. 今回のレースが「堅実決着型」か「波乱型」かを、脚質分布・ペース想定・出走馬の層から判断
2. 堅実決着型 → 系統Aベースで買い目構築（ワイドBOX等）
3. 波乱型 → 系統Bベースで買い目構築（単勝BOX・馬連流し等）
4. 両系統から最終推奨馬を選定

■ 最終出力フォーマット
【展開判断】堅実決着型 or 波乱型（理由）
【採用系統】A or B（理由）
【最終推奨】
  ◎（本命）○（対抗）▲（単穴）△（連下）★（穴）
  ×（消し馬）＋消し理由
【推奨買い目】
  系統A採用時: ワイドBOX / 3連複フォーメーション等
  系統B採用時: 単勝BOX(5点) / ★穴からの馬連流し等
【投資配分】1レースあたりの推奨投資額と点数

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
    # 前走クラス数値→名称マッピング
    CLASS_NAMES = {
        8: 'G1', 7: 'G2', 6: 'G3', 5: 'Listed', 4: 'OP',
        3: '3勝', 2: '2勝', 1: '未勝利/1勝',
    }

    lines = []

    # ヘッダー
    lines.append(
        f"{'順位':>4s} {'馬番':>4s} {'枠':>2s} {'馬名':18s} "
        f"{'μ':>7s} {'σ':>7s} "
        f"{'勝率':>7s} {'複勝率':>7s} {'E[rank]':>8s} "
        f"{'EMA着順':>8s} {'タイムZ':>8s} {'着差EMA':>8s} "
        f"{'上3F_EMA':>8s} {'上3F順':>8s} {'上3F_Z':>8s} "
        f"{'PaceF':>8s} {'PaceD':>8s} {'H着差':>7s} {'S着差':>7s} "
        f"{'同距離':>8s} {'同馬場面':>8s} {'同馬場状':>8s} {'距離差':>6s} "
        f"{'脚質':>6s} {'1角位':>6s} {'2角位':>6s} {'3角位':>6s} "
        f"{'馬体重':>6s} {'増減':>4s} {'間隔日':>6s} {'走数':>4s} "
        f"{'騎手勝':>7s} {'騎手複':>7s} {'騎手D勝':>7s} {'騎手D複':>7s} "
        f"{'師勝':>7s} {'師複':>7s} {'師D勝':>7s} {'師D複':>7s} "
        f"{'前走CL':>10s} {'賞金log':>8s} {'勝率実':>7s} {'複勝実':>7s}"
    )
    lines.append("-" * 300)

    def _fmt(val, width=8, decimals=3):
        """NaN安全なフォーマット"""
        if pd.isna(val):
            return ' ' * (width - 3) + 'N/A'
        return f"{val:{width}.{decimals}f}"

    for rank, (_, r) in enumerate(mc_results.iterrows(), 1):
        u = int(r['umaban'])
        fd = race_features[race_features['umaban'] == u]
        ps = nn_preds[nn_preds['umaban'] == u]
        if len(fd) == 0 or len(ps) == 0:
            continue
        fd = fd.iloc[0]
        ps = ps.iloc[0]

        # 前走クラス名
        prev_cl = fd.get('prev_race_class', 1)
        cl_name = CLASS_NAMES.get(int(prev_cl), f'{int(prev_cl)}')
        cl_str = f"{int(prev_cl)}({cl_name})"

        lines.append(
            f"{rank:4d} {u:4d} {int(fd['wakuban']):2d} {r['horse_name']:18s} "
            f"{ps['mu']:7.4f} {ps['sigma']:7.4f} "
            f"{r['win_prob']:7.2%} {r['top3_prob']:7.2%} {r['expected_rank']:8.2f} "
            f"{_fmt(fd.get('ema_finish'))} {_fmt(fd.get('ema_time_zscore'))} {_fmt(fd.get('ema_time_diff'))} "
            f"{_fmt(fd.get('ema_agari_3f'))} {_fmt(fd.get('ema_agari_rank'))} {_fmt(fd.get('ema_agari_zscore'))} "
            f"{_fmt(fd.get('ema_pace_front'))} {_fmt(fd.get('ema_pace_diff'))} "
            f"{_fmt(fd.get('pace_h_time_diff'), 7)} {_fmt(fd.get('pace_s_time_diff'), 7)} "
            f"{_fmt(fd.get('same_dist_finish'))} {_fmt(fd.get('same_surface_finish'))} {_fmt(fd.get('same_baba_finish'))} "
            f"{fd.get('prev_dist_diff', 0):6.0f} "
            f"{fd.get('avg_run_style', 0):6.2f} "
            f"{_fmt(fd.get('avg_jyuni_1c'), 6, 2)} {_fmt(fd.get('avg_jyuni_2c'), 6, 2)} {_fmt(fd.get('avg_jyuni_3c'), 6, 2)} "
            f"{fd.get('bataijyu', 0):6.0f} {fd.get('zogen_sa', 0):4.0f} "
            f"{fd.get('interval_days', 0):6.0f} {fd.get('past_count', 0):4.0f} "
            f"{fd.get('jockey_win_rate', 0):7.3f} {fd.get('jockey_top3_rate', 0):7.3f} "
            f"{fd.get('jockey_dist_win_rate', 0):7.3f} {fd.get('jockey_dist_top3_rate', 0):7.3f} "
            f"{fd.get('trainer_win_rate', 0):7.3f} {fd.get('trainer_top3_rate', 0):7.3f} "
            f"{fd.get('trainer_dist_win_rate', 0):7.3f} {fd.get('trainer_dist_top3_rate', 0):7.3f} "
            f"{cl_str:>10s} {fd.get('log_prize_money', 0):8.3f} "
            f"{fd.get('win_rate', 0):7.3f} {fd.get('top3_rate', 0):7.3f}"
        )

    return "\n".join(lines)


def build_prompt(race_name, course_name, distance, heads, date_str,
                 mc_results, race_features, nn_preds, race_id='',
                 line_a=None, line_b=None, pop_umabans=None):
    """
    3層アーキテクチャの最終出力: 議長プロンプトを完成させる

    Parameters
    ----------
    line_a : DataFrame - 系統A(V1従来QMC上位5頭)
    line_b : DataFrame - 系統B(堅2穴3)
    pop_umabans : set - 系統Bの本命馬番セット
    """
    race_info = format_race_info(race_name, course_name, distance, heads, date_str, race_id)
    horse_data = format_horse_data(mc_results, race_features, nn_preds)

    # 系統A
    if line_a is not None:
        a_lines = []
        for rk, (_, r) in enumerate(line_a.iterrows(), 1):
            a_lines.append(f"  {rk}. [{int(r['umaban']):2d}] {r['horse_name']}")
        line_a_info = "\n".join(a_lines)
    else:
        a_lines = []
        for rk, (_, r) in enumerate(mc_results.head(5).iterrows(), 1):
            a_lines.append(f"  {rk}. [{int(r['umaban']):2d}] {r['horse_name']}")
        line_a_info = "\n".join(a_lines)

    # 系統B
    if line_b is not None and pop_umabans is not None:
        b_lines = []
        for rk, (_, r) in enumerate(line_b.iterrows(), 1):
            u = int(r['umaban'])
            tag = '\u2605' if u in pop_umabans else '\u2606'
            odds_str = f"{r['odds']:.1f}" if pd.notna(r.get('odds')) else '?'
            b_lines.append(f"  {rk}. {tag}[{u:2d}] {r['horse_name']} (odds:{odds_str})")
        line_b_info = "\n".join(b_lines)
    else:
        line_b_info = "  (未計算)"

    return MODERATOR_PROMPT.format(
        race_info=race_info, horse_data=horse_data,
        line_a_info=line_a_info, line_b_info=line_b_info,
    )
