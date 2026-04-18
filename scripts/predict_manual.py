"""
手動入力レース予測: 出馬表CSVがentry_parser形式でない場合用
馬名リストとV1キャッシュからマッチしてNN+QMC実行
"""
import sys, io, os, warnings, argparse
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch
np.random.seed(42)

from src.predictor_v2 import PredictorV2
from src.features_v2 import FEATURES_V2, CAT_FEATURES_V2
from src.qmc_courses import qmc_sim, COURSE_PROFILES
from src.prompts import build_prompt
from src.debate_rules import select_with_rules


def parse_odds_csv(path):
    """syunnrai1形式: col2=馬番, col7=馬名, col15=オッズ"""
    with open(path, 'rb') as f:
        text = f.read().decode('cp932')
    result = {}
    for line in text.strip().split('\r\n'):
        c = line.split(',')
        name = c[7].strip()
        umaban = int(c[2].strip())
        try:
            odds = float(c[15].strip())
        except:
            odds = 999
        result[name] = {'umaban': umaban, 'odds': odds}
    return result


def parse_past5_csv(path):
    """syunrai2形式: col0=枠, col2=馬番, col7=馬名, col11~=前5走(各12列)"""
    with open(path, 'rb') as f:
        text = f.read().decode('cp932')
    horses = []
    for line in text.strip().split('\r\n'):
        c = line.split(',')
        umaban = int(c[2].strip())
        name = c[7].strip()
        races = []
        starts = [11, 23, 35, 47, 59]
        for s in starts:
            if s + 11 >= len(c):
                break
            try:
                heads = c[s].strip()
                fin = c[s+1].strip()
                venue = c[s+3].strip()
                surface = c[s+4].strip()
                dist = c[s+5].strip()
                baba = c[s+6].strip()
                cls = c[s+9].strip()
                agari = c[s+10].strip()
                style = c[s+11].strip()
                if venue:
                    surface_ja = {'T': '芝', 'D': 'ダ', 'S': '障'}.get(surface, surface)
                    races.append(f'{venue}{surface_ja}{dist}m{baba} {cls} {fin}着/{heads}頭 上り{agari} {style}')
            except:
                pass
        horses.append({'umaban': umaban, 'name': name, 'past5': races})
    return horses


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('odds_csv', help='オッズCSV (col2=馬番, col7=馬名, col15=オッズ)')
    parser.add_argument('past5_csv', help='前5走CSV (72列)')
    parser.add_argument('--race-name', default='不明', help='レース名')
    parser.add_argument('--course', default=None, help='コースキー')
    parser.add_argument('--distance', type=int, default=1200, help='距離')
    parser.add_argument('--venue', default='中山', help='場所')
    parser.add_argument('--surface', default='芝', help='芝/ダ')
    parser.add_argument('--date', default='2026-04-12', help='日付')
    args = parser.parse_args()

    # データ読み込み
    odds_data = parse_odds_csv(args.odds_csv)
    past5_data = parse_past5_csv(args.past5_csv)
    horse_names = [h['name'] for h in past5_data]

    print(f'[Info] {args.race_name} {args.venue}{args.surface}{args.distance}m {len(horse_names)}頭')

    # キャッシュからマッチ
    feat = pd.read_pickle('data/features_v9b_2026.pkl')
    race_date = pd.Timestamp(args.date)

    rows = []
    missing = []
    for h in past5_data:
        name = h['name']
        uma = h['umaban']
        match = feat[feat['horse_name'].str.strip() == name]
        if len(match) == 0:
            missing.append(name)
            continue
        latest = match.sort_values('date').iloc[-1].copy()
        latest['umaban'] = uma
        # オッズ上書き
        if name in odds_data:
            latest['odds'] = odds_data[name]['odds']
        rows.append(latest)

    if missing:
        print(f'  WARNING: Missing {missing}')

    rf = pd.DataFrame(rows)
    rf['ninki'] = rf['odds'].rank(method='first')

    # NN学習
    is_turf = 1 if args.surface == '芝' else 0
    use_f = [f for f in FEATURES_V2 if f in feat.columns]
    use_cf = [f for f in CAT_FEATURES_V2 if f in feat.columns]

    tr = feat[
        (feat['date'] < race_date) &
        (feat['is_turf'] == is_turf) &
        (feat['past_count'] > 0) &
        (feat['finish'] > 0)
    ].sort_values('date').tail(40000).copy()

    print(f'[Layer1] Training PredictorV2 ({len(tr):,} samples)...')
    pred = PredictorV2(use_f, use_cf)
    pred.train(tr, ep=80, lr=0.003, seed=42)

    ps = pred.predict(rf)

    # QMC
    course = args.course
    if course and course in COURSE_PROFILES:
        print(f'[Layer2] QMC ({course})...')
        mc = qmc_sim(ps, race_features=rf, course=course, n=100000)
    else:
        print(f'[Layer2] QMC (default)...')
        mc = qmc_sim(ps, race_features=rf, course='nakayama_turf_1200' if args.distance == 1200 else 'hanshin_turf_1600_outer', n=100000)

    # 人気情報をmcに追加
    ninki_map = {int(r['umaban']): int(r['ninki']) for _, r in rf.iterrows()}
    mc['ninki'] = mc['umaban'].astype(int).map(ninki_map)

    # 2系統選抜
    line_a = mc.head(5)
    line_b, pop_umabans, flagged_info = select_with_rules(mc, rf, n_pop=2, n_disc=3, cutoff=5)

    # 結果表示
    print(f'\n{"="*80}')
    print(f'  {args.race_name} {args.venue}{args.surface}{args.distance}m')
    print(f'{"="*80}')

    print(f'\n  {"QMC":>3s} {"番":>3s} {"馬名":18s} {"人気":>4s} {"odds":>7s} {"勝率":>7s} {"複勝":>7s} {"E[rank]":>8s} {"μ":>7s} {"σ":>7s}  選抜')
    print(f'  {"-"*90}')
    a_set = set(line_a['umaban'].astype(int))
    b_set = set(line_b['umaban'].astype(int))
    for rank, (_, r) in enumerate(mc.iterrows(), 1):
        u = int(r['umaban'])
        nk = int(r['ninki']) if pd.notna(r.get('ninki')) else 0
        ps_row = ps[ps['umaban'] == u]
        mu_val = ps_row.iloc[0]['mu'] if len(ps_row) > 0 else 0
        sig_val = ps_row.iloc[0]['sigma'] if len(ps_row) > 0 else 0
        tags = []
        if u in a_set:
            tags.append('A')
        if u in b_set:
            tags.append('B★' if u in pop_umabans else 'B☆')
        tag = ' ' + ','.join(tags) if tags else ''
        print(f'  {rank:3d} {u:3d} {r["horse_name"]:18s} {nk:3d}人気 {r["odds"]:7.1f} {r["win_prob"]:7.1%} {r["top3_prob"]:7.1%} {r["expected_rank"]:8.2f} {mu_val:7.4f} {sig_val:7.4f}{tag}')

    print(f'\n  系統A(V1従来): QMC上位5頭')
    for rk, (_, r) in enumerate(line_a.iterrows(), 1):
        u = int(r['umaban'])
        nk = ninki_map.get(u, 0)
        print(f'    {rk}. [{u:2d}] {r["horse_name"]} ({nk}人気/{r["odds"]:.1f}倍)')

    print(f'\n  系統B(堅2穴3):')
    for rk, (_, r) in enumerate(line_b.iterrows(), 1):
        u = int(r['umaban'])
        nk = ninki_map.get(u, 0)
        tag = '★' if u in pop_umabans else '☆'
        print(f'    {rk}. {tag}[{u:2d}] {r["horse_name"]} ({nk}人気/{r["odds"]:.1f}倍)')

    # 前5走テキスト生成
    past5_text_lines = []
    for h in past5_data:
        uma = h['umaban']
        name = h['name']
        od = odds_data.get(name, {}).get('odds', 0)
        nk = ninki_map.get(uma, 0)
        past5_text_lines.append(f'{uma:2d}番 {name}（{nk}人気/{od}倍）')
        for i, r in enumerate(h['past5'], 1):
            past5_text_lines.append(f'  前{i}走: {r}')
        past5_text_lines.append('')

    # プロンプト生成
    prompt = build_prompt(
        race_name=f'{args.venue}11R {args.race_name}',
        course_name=f'{args.venue}{args.surface}{args.distance}m',
        distance=args.distance,
        heads=len(rf),
        date_str=args.date,
        mc_results=mc,
        race_features=rf,
        nn_preds=ps,
        race_id=f'{args.date.replace("-","")}_{args.venue}_11',
        line_a=line_a,
        line_b=line_b,
        pop_umabans=pop_umabans,
        past_races_text='\n'.join(past5_text_lines),
    )

    out_name = f'output_{args.race_name}_prompt.txt'
    with open(out_name, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f'\n  Prompt saved: {out_name}')


if __name__ == '__main__':
    run()
