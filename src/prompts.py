"""
議長プロンプトテンプレート v2
6エージェントディベート + 2系統選抜 + 入れ替えステップ
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

{past_races_section}

■ ⚠️ 絶対厳守ルール
- 上記で提供されたv9bの数値データ・前5走データ・オッズのみを根拠として議論すること。外部知識の持ち込み禁止。
- NNはオッズを知らない。QMC順位と人気順のズレこそがNNの付加価値であり、議論の出発点。

■ 召喚する6人のエージェントと【必須分析タスク】

1. 【Agent A：絶対能力派】
役割：タイム偏差値(タイムZ)・着差EMA・上がり3Fを最重視。
★必須タスク：QMC順位が低くても、「タイムZや上がり3Fの絶対値が突出している馬」を見つけ出し、ポテンシャル枠として必ず提示。前5走の上がりタイムも精査すること。

2. 【Agent B：舞台適性派】
役割：距離適性・コース形態×脚質・馬場状態を重視。
★必須タスク：前5走の距離/場所/脚質パターンから今回コースへの適性を判断。距離延長/短縮ショック、小回り/大箱の適性差を明確にすること。

3. 【Agent C：展開・ペース派】
役割：脚質分布からペースを予測し、展開利/不利を判断。
★必須タスク：逃げ馬の数、先行馬の密集度からスロー/ハイを予測。「このペースなら誰が有利か」を具体的に述べること。

4. 【Agent D：逆張り派（最重要）】
役割：QMC上位馬・人気馬の粗探し。消し馬を指名する。
★必須タスク：
  - QMC上位5頭それぞれに対し「なぜこの馬は過大評価か」を論じること
  - 特に「前走クラスが低い(3勝以下→重賞/L初挑戦)」「タイムZが低い」「σが小さいのにμも悪い(安定して凡走)」馬を狙い撃ち
  - 逆に「QMC下位だが人気上位」の馬はNNが見落としている可能性を指摘
  - 最低1頭の「消し馬」を明確に指名すること

5. 【Agent E：血統・ローテ派】
役割：前5走のローテーション（間隔・クラス推移・場所）から調子の上がり下がりを判断。
★必須タスク：「前走から上昇中」vs「下降中」を全馬で判定。休み明け(間隔90日超)の馬は特に慎重に。

6. 【Agent F：データファクト派】
役割：馬体重増減・騎手/調教師の勝率・走数などの統計ファクトを客観提示。
★必須タスク：
  - 馬体重の大幅増減(±10kg超)がある馬を警告
  - 騎手勝率・調教師勝率が全馬平均を大きく上回る/下回る馬を指摘
  - 走数が多い馬(30走超)の消耗リスクを評価

■ ディベートの進行手順

【ステップ0：全馬スクリーニング（必須・省略厳禁）】
6エージェント全員が全馬を評価し、以下の表を作成。

| 馬番 | 馬名 | 人気 | A(能力) | B(適性) | C(展開) | D(逆張り) | E(ローテ) | F(データ) | 注目理由 |

評価: ◎/○/△/× の4段階
★QMC下位なのに◎が2つ以上ある馬 = 「要注目馬」
★QMC上位なのに×が2つ以上ある馬 = 「要警戒馬」

【ステップ1：各エージェントの主張】
A→B→C→D→E→Fの順に「推奨馬」「消し馬」「要注目馬への見解」を発表。
Agent Dは必ずQMC上位への反論を述べること。

【ステップ2：クロスチェック（激しい反論）】
Agent Dを中心に、他エージェントの推奨馬に対してデータを用いた粗探しを実施。
各エージェントは反論に対して再反論すること。

【ステップ3：入れ替え判断（必須）】
Layer1-2の選抜5頭に対し、議長が以下を判断：
  - 堅実枠(★)の2頭のうち、入れ替えるべき馬はいるか？
    → 「5番人気以内でQMC選抜外だが、ディベートで高評価の馬」があれば入れ替え
  - 穴枠(☆)の3頭のうち、入れ替えるべき馬はいるか？
    → 「6番人気以降でQMC選抜外だが、ディベートで高評価の馬」があれば入れ替え
  ★入れ替えには明確な根拠（Agent Dの消し理由 + 代替馬の優位点）が必要

【ステップ4：議長による最終結論】

■ Layer1-2が2系統の選抜を出力済みです。

【系統A：占有率重視（V1従来）】QMC上位5頭
{line_a_info}
→ 10年バックテスト: 占有率80%, 単勝ROI 74.8%

【系統B：ROI重視（堅2穴3）】★本命(5人気以内QMC上位2頭) + ☆穴(6人気以降QMC上位3頭)
{line_b_info}
→ 10年バックテスト: 占有率50%, 単勝ROI 139.6%, 馬連ROI 56.9%

■ 最終出力フォーマット

【展開判断】堅実決着型 or 波乱型（理由）
【採用系統】A or B（理由）
【入れ替え】あり/なし
  入れ替えありの場合: OUT→INと理由
【最終推奨5頭】
  ★◎（本命）★○（対抗）＝堅実枠2頭
  ☆▲（単穴）☆△（連下）☆★（大穴）＝穴枠3頭
  ×（消し馬）＋消し理由
【推奨買い目】
  系統A採用時: ワイドBOX / 3連複フォーメーション等
  系統B採用時: 単勝BOX(5点) / ★穴→☆本命の馬連流し等
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
    """
    CLASS_NAMES = {
        8: 'G1', 7: 'G2', 6: 'G3', 5: 'Listed', 4: 'OP',
        3: '3勝', 2: '2勝', 1: '未勝利/1勝',
    }

    lines = []
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
                 line_a=None, line_b=None, pop_umabans=None,
                 past_races_text=''):
    """
    6エージェントディベート用プロンプトを完成させる

    Parameters
    ----------
    line_a : DataFrame - 系統A(V1従来QMC上位5頭)
    line_b : DataFrame - 系統B(堅2穴3)
    pop_umabans : set - 系統Bの本命馬番セット
    past_races_text : str - 前5走データのテキスト（外部で生成して渡す）
    """
    race_info = format_race_info(race_name, course_name, distance, heads, date_str, race_id)
    horse_data = format_horse_data(mc_results, race_features, nn_preds)

    # 前5走セクション
    if past_races_text:
        past_races_section = f"【前5走データ】\n{past_races_text}"
    else:
        past_races_section = ""

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
        past_races_section=past_races_section,
        line_a_info=line_a_info, line_b_info=line_b_info,
    )
