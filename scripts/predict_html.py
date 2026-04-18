"""
HTML + 148列CSV ベースの汎用予測スクリプト v3
TARGET JVの出馬表HTMLとフルセットCSVから全自動で予測する

使い方:
    py -3 scripts/predict_html.py C:/TXT/出馬表.html C:/TXT/フルセット.csv
    py -3 scripts/predict_html.py C:/TXT/出馬表.html C:/TXT/フルセット.csv --no-debate
"""
import sys, os, re, warnings, argparse, time
from collections import defaultdict

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd
import numpy as np
import torch
from itertools import combinations

np.random.seed(42)

from src.predictor_v2 import PredictorV2
from src.features_v2 import FEATURES_V2, CAT_FEATURES_V2
from src.qmc_courses import qmc_sim, COURSE_PROFILES
from src.prompts import build_prompt
from src.debate_rules import select_with_rules


# ============================================================
# 場名マップ
# ============================================================
VENUE_MAP = {
    '札幌': 'sapporo', '函館': 'hakodate', '福島': 'fukushima',
    '新潟': 'niigata', '東京': 'tokyo', '中山': 'nakayama',
    '中京': 'chukyo', '阪神': 'hanshin', '京都': 'kyoto',
    '小倉': 'kokura',
}

# ============================================================
# Step 1: HTML パーサー
# ============================================================
def strip_tags(html):
    """HTMLタグを除去してテキストのみ返す"""
    return re.sub(r'<[^>]+>', '', html).strip()


def parse_html(html_path):
    """
    TARGET JV出馬表HTMLからレース情報と出走馬リストを取得

    Returns: race_info dict, entries list
    """
    with open(html_path, 'rb') as f:
        raw = f.read()
    text = raw.decode('cp932', errors='replace')

    # --- レース情報 ---
    # タイトル: 出馬表・中山 9R 袖ケ浦特･2勝 or 出馬表・福島12R １勝ｸﾗｽ
    title_m = re.search(r'<TITLE>出馬表・(\S+?)\s*(\d+)R\s*(.*?)</TITLE>', text)
    if not title_m:
        # 全角R対応
        title_m = re.search(r'<TITLE>出馬表・(\S+?)\s*(\d+)Ｒ\s*(.*?)</TITLE>', text)
    venue = title_m.group(1) if title_m else ''
    race_num = int(title_m.group(2)) if title_m else 0
    race_name_raw = title_m.group(3).strip() if title_m else ''

    # 日付
    date_m = re.search(r'(\d+)年\s*(\d+)月(\d+)日', text)
    if date_m:
        year = int(date_m.group(1))
        month = int(date_m.group(2))
        day = int(date_m.group(3))
        race_date = pd.Timestamp(year, month, day)
    else:
        race_date = pd.Timestamp.now()

    # 芝/ダ + 距離: 芝1200m (C) or ダ1800m
    course_m = re.search(r'(芝|ダ)(\d+)m', text)
    surface = course_m.group(1) if course_m else '芝'
    distance = int(course_m.group(2)) if course_m else 1600

    # 頭数
    heads_m = re.search(r'(\d+)頭', text)
    heads = int(heads_m.group(1)) if heads_m else 0

    venue_key = VENUE_MAP.get(venue, venue.lower())
    surface_key = 'turf' if surface == '芝' else 'dirt'
    course_key = f'{venue_key}_{surface_key}_{distance}'

    race_info = {
        'date': race_date,
        'date_str': f'{race_date:%Y-%m-%d}',
        'venue': venue,
        'venue_key': venue_key,
        'race_num': race_num,
        'race_name': race_name_raw,
        'surface': surface,
        'surface_key': surface_key,
        'distance': distance,
        'heads': heads,
        'course_key': course_key,
        'race_id': f'{race_date:%Y%m%d}_{venue_key}_{race_num:02d}',
    }

    # --- 出走馬パース ---
    # テーブル行を取得
    rows = re.findall(r'<TR>\s*(.*?)\s*</TR>', text, re.DOTALL)
    entries = []
    for row in rows:
        cells = re.findall(r'<TD[^>]*>(.*?)</TD>', row, re.DOTALL)
        if len(cells) < 12:
            continue
        try:
            # 枠番: "B4" や " 4" のようなフォーマットがある
            waku_text = re.sub(r'[A-Za-z]', '', strip_tags(cells[0]).strip())
            wakuban = int(waku_text) if waku_text else 0
            umaban = int(strip_tags(cells[1]).strip())
        except ValueError:
            continue

        horse_name = strip_tags(cells[6]).strip().lstrip('*')
        sex_age = strip_tags(cells[8]).strip()
        jockey = strip_tags(cells[9]).strip().lstrip('*')
        futan_text = re.sub(r'[^\d.]', '', strip_tags(cells[10]).strip())
        futan = float(futan_text) if futan_text else 55.0
        odds_text = re.sub(r'[^\d.]', '', strip_tags(cells[11]).strip())
        try:
            odds = float(odds_text)
        except ValueError:
            odds = 99.9

        entries.append({
            'wakuban': wakuban,
            'umaban': umaban,
            'horse_name': horse_name,
            'jockey': jockey,
            'futan': futan,
            'odds': odds,
        })

    race_info['heads'] = len(entries)
    return race_info, entries


# ============================================================
# Step 2: 148列フルセットCSVパーサー
# ============================================================
def parse_fullset_csv(csv_path):
    """
    148列フルセットCSVから全走データをパース

    Returns: dict[horse_name] -> list of race dicts (newest first)
    """
    with open(csv_path, 'rb') as f:
        raw = f.read()
    text = raw.decode('cp932', errors='replace')
    lines = text.strip().split('\r\n')

    horse_runs = defaultdict(list)
    for line in lines:
        c = line.split(',')
        if len(c) < 120:
            continue

        name = c[11].strip()
        try:
            finish = int(c[16]) if c[16].strip() else 0
        except ValueError:
            finish = 0

        run = {
            'date': f'{c[0]}/{c[1]}/{c[2]}',
            'venue': c[4].strip(),
            'surface': c[8].strip(),
            'distance': c[9].strip(),
            'baba': c[10].strip(),
            'finish': finish,
            'heads': c[19].strip(),
            'race_name': c[7].strip(),
            'jockey': c[14].strip(),
            'futan': c[15].strip(),
            'wakuban': c[17].strip(),
            'umaban': c[18].strip(),
            'ninki': c[21].strip(),
            'time': c[23].strip(),
            'weight': c[29].strip(),
            'weight_diff': c[116].strip() if len(c) > 116 else '',
            'trainer': c[30].strip(),
            'trainer_loc': c[31].strip() if len(c) > 31 else '',
            'agari_3f': c[45].strip() if len(c) > 45 else '',
            'agari_rank': c[46].strip() if len(c) > 46 else '',
            'ten_3f': c[44].strip() if len(c) > 44 else '',
            'first_half': c[42].strip() if len(c) > 42 else '',
            'second_half': c[43].strip() if len(c) > 43 else '',
            'run_style': c[109].strip() if len(c) > 109 else '',
            'corner': c[22].strip() if len(c) > 22 else '',
            'sire': c[37].strip() if len(c) > 37 else '',
            'dam': c[38].strip() if len(c) > 38 else '',
            'broodmare_sire': c[39].strip() if len(c) > 39 else '',
            'pace_diff': c[20].strip() if len(c) > 20 else '',
        }
        horse_runs[name].append(run)

    return horse_runs


def format_past_races(entries, horse_runs, max_runs=10):
    """前走データをディベート用テキストに整形"""
    lines = []
    for e in entries:
        runs = horse_runs.get(e['horse_name'], [])
        if not runs:
            continue
        run_texts = []
        for r in runs[:max_runs]:
            run_texts.append(
                f'{r["venue"]}{r["surface"]}{r["distance"]}m {r["baba"]} '
                f'{r["finish"]}/{r["heads"]}着 {r["race_name"]} '
                f'上り{r["agari_3f"]} 上順{r["agari_rank"]} '
                f'テン{r["ten_3f"]} 脚質:{r["run_style"]} '
                f'体重{r["weight"]}({r["weight_diff"]}) '
                f'騎手{r["jockey"]}'
            )
        lines.append(f'{e["umaban"]}番 {e["horse_name"]}({horse_runs.get(e["horse_name"],[{}])[0].get("sire","")}):\n  ' + '\n  '.join(run_texts))
    return '\n'.join(lines)


# ============================================================
# Step 3: コースプロファイル自動選択
# ============================================================
def find_course(race_info):
    """race_infoからCOURSE_PROFILESのキーを自動選択"""
    key = race_info['course_key']
    if key in COURSE_PROFILES:
        return key

    # 部分一致
    for k in COURSE_PROFILES:
        if race_info['venue_key'] in k and str(race_info['distance']) in k:
            return k

    # 同じ場の最近接距離
    venue_profiles = {k: v for k, v in COURSE_PROFILES.items() if race_info['venue_key'] in k}
    if venue_profiles:
        closest = min(venue_profiles.keys(),
                      key=lambda k: abs(int(re.search(r'\d+', k.split('_')[-1]).group()) - race_info['distance']))
        print(f'  WARNING: Exact profile not found for {key}, using {closest}')
        return closest

    print(f'  WARNING: No profile found for {key}, using default')
    return 'hanshin_turf_1600_outer'


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='HTML+148列CSV 汎用予測 v3')
    parser.add_argument('html_path', help='TARGET出馬表HTMLパス')
    parser.add_argument('csv_path', help='148列フルセットCSVパス')
    parser.add_argument('--sims', type=int, default=100000, help='QMCシミュレーション回数')
    parser.add_argument('--no-debate', action='store_true', help='Layer3ディベートをスキップ')
    args = parser.parse_args()

    t_start = time.time()

    # ==================================================
    # Step 1: HTML解析
    # ==================================================
    print(f'[Step1] Parsing HTML: {args.html_path}')
    race_info, entries = parse_html(args.html_path)
    print(f'  {race_info["date_str"]} {race_info["venue"]}{race_info["race_num"]}R '
          f'{race_info["surface"]}{race_info["distance"]}m {race_info["race_name"]} '
          f'{race_info["heads"]}頭')
    print()
    for e in entries:
        print(f'  {e["umaban"]:2d}番 枠{e["wakuban"]} {e["horse_name"]:15s} {e["odds"]:6.1f}倍 {e["jockey"]}')

    # ==================================================
    # Step 2: 148列CSV解析
    # ==================================================
    print(f'\n[Step2] Parsing fullset CSV: {args.csv_path}')
    horse_runs = parse_fullset_csv(args.csv_path)
    for e in entries:
        n_runs = len(horse_runs.get(e['horse_name'], []))
        if n_runs == 0:
            print(f'  WARNING: {e["horse_name"]} not found in CSV')

    total_runs = sum(len(v) for v in horse_runs.values())
    print(f'  {len(horse_runs)}馬, {total_runs}走分のデータ読み込み完了')

    # ==================================================
    # Step 3: 特徴量キャッシュロード + マッチング
    # ==================================================
    print(f'\n[Step3] Loading feature cache & matching...')
    candidates = [
        'data/features_v2_cache.pkl',
        'data/features_v9b_2026.pkl',
        'data/features_v9b_cache.pkl',
    ]
    cache_path = next((p for p in candidates if os.path.exists(p)), None)
    if cache_path is None:
        print('  ERROR: No feature cache found.')
        sys.exit(1)

    feat = pd.read_pickle(cache_path)
    feat['date'] = pd.to_datetime(feat['date'])
    print(f'  Cache: {cache_path} ({len(feat):,} rows)')

    # 馬名でマッチング
    race_rows = []
    missing = []
    for e in entries:
        cand = feat[feat['horse_name'] == e['horse_name']].sort_values('date')
        if len(cand) == 0:
            missing.append(e['horse_name'])
            continue
        latest = cand.iloc[-1].copy()
        latest['umaban'] = e['umaban']
        latest['wakuban'] = e['wakuban']
        latest['futan'] = e['futan']
        latest['odds'] = e['odds']
        latest['heads'] = race_info['heads']
        latest['horse_name'] = e['horse_name']
        race_rows.append(latest)

    if missing:
        print(f'  WARNING: Missing from cache: {missing}')

    rf = pd.DataFrame(race_rows)
    print(f'  Matched: {len(rf)}/{race_info["heads"]} horses')

    t1 = time.time()

    # ==================================================
    # Step 4: NN学習
    # ==================================================
    print(f'\n[Step4/Layer1] Training PredictorV2...')
    use_f = [f for f in FEATURES_V2 if f in feat.columns]
    use_cf = [f for f in CAT_FEATURES_V2 if f in feat.columns]

    is_turf = 1 if race_info['surface_key'] == 'turf' else 0
    tr = feat[
        (feat['is_turf'] == is_turf) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date').tail(40000).copy()
    print(f'  Training: {len(tr):,} samples')

    pred = PredictorV2(use_f, use_cf)
    pred.train(tr, ep=80, lr=0.003, seed=42)
    t2 = time.time()
    print(f'  Done in {t2 - t1:.1f}s')

    # ==================================================
    # Step 5: QMC
    # ==================================================
    course = find_course(race_info)
    cp = COURSE_PROFILES[course]
    print(f'\n[Step5/Layer2] QMC: {cp["name"]} ({args.sims:,} sims)...')

    ps = pred.predict(rf)
    mc = qmc_sim(ps, race_features=rf, course=course, n=args.sims)
    t3 = time.time()
    print(f'  Done in {t3 - t2:.1f}s')

    # ==================================================
    # 結果出力
    # ==================================================
    print(f'\n{"="*80}')
    print(f'  {race_info["date_str"]} {race_info["venue"]}{race_info["race_num"]}R '
          f'{race_info["surface"]}{race_info["distance"]}m {race_info["race_name"]} '
          f'{race_info["heads"]}頭')
    print(f'  v2(D) + QMC予測結果 | コース: {cp["name"]}')
    print(f'{"="*80}')

    if 'ninki' not in mc.columns:
        mc['ninki'] = mc['odds'].rank(method='first')

    print(f'\n  {"順位":>4s} {"枠":>2s} {"番":>3s} {"馬名":15s} '
          f'{"勝率":>7s} {"複勝率":>7s} {"期待着順":>8s} {"μ":>7s} {"σ":>7s} {"odds":>7s}')
    print(f'  {"-"*80}')

    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        rf_row = rf[rf['umaban'] == u]
        if len(rf_row) == 0:
            continue
        rf_row = rf_row.iloc[0]
        ps_row = ps[ps['umaban'] == u]
        if len(ps_row) == 0:
            continue
        ps_row = ps_row.iloc[0]
        print(f'  {rank:4d} {int(rf_row["wakuban"]):2d} {u:3d} {r["horse_name"]:15s} '
              f'{r["win_prob"]:7.1%} {r["top3_prob"]:7.1%} {r["expected_rank"]:8.2f} '
              f'{ps_row["mu"]:7.4f} {ps_row["sigma"]:7.4f} {r["odds"]:7.1f}')

    # 2系統選抜
    mc_d = mc.copy()
    if 'ninki' not in mc_d.columns:
        mc_d['ninki'] = mc_d['odds'].rank(method='first')
    line_a = mc_d.head(5)

    try:
        line_b, pop_umabans, flagged_info = select_with_rules(mc_d, rf, n_pop=2, n_disc=3, cutoff=5)
    except Exception as e:
        print(f'  Warning: select_with_rules failed: {e}')
        line_b = None
        pop_umabans = None
        flagged_info = []

    # TOP5
    top5 = mc.head(5)
    print(f'\n  QMC TOP5:')
    for rk, (_, r) in enumerate(top5.iterrows(), 1):
        print(f'    {rk}. [{int(r["umaban"]):2d}] {r["horse_name"]} ({r["odds"]:.1f}倍)')

    # ワイドBOX
    top5_list = [int(r['umaban']) for _, r in top5.iterrows()]
    print(f'\n  ワイドBOX ({len(list(combinations(top5_list,2)))}点):')
    for a, b in combinations(top5_list, 2):
        print(f'    {a}-{b}')

    # ==================================================
    # Layer 3: ディベートプロンプト
    # ==================================================
    if not args.no_debate:
        print(f'\n[Layer3] Generating debate prompt...')

        # 前走テキスト生成（148列CSV全走データ）
        past_text = format_past_races(entries, horse_runs, max_runs=10)

        prompt = build_prompt(
            race_name=f'{race_info["venue"]}{race_info["race_num"]}R {race_info["race_name"]}',
            course_name=cp['name'],
            distance=race_info['distance'],
            heads=race_info['heads'],
            date_str=race_info['date_str'],
            mc_results=mc,
            race_features=rf,
            nn_preds=ps,
            race_id=race_info['race_id'],
            line_a=line_a,
            line_b=line_b,
            pop_umabans=pop_umabans,
            past_races_text=past_text,
        )

        out_name = f'output_{race_info["race_id"]}_prompt.txt'
        with open(out_name, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f'  Saved: {out_name}')
        print(f'\n--- 議長プロンプト (Layer 3) ---')
        print(prompt)

    total = time.time() - t_start
    print(f'\n  Total: {total:.1f}s')


if __name__ == '__main__':
    main()
