"""
ディベートルールのバックテスト: QMCエラー分析から導出したルールを適用して
占有率/ROIがどう変わるか検証（全重賞/L 1,192レース）
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np

def apply_rules(mc, rf, feat_cache, n_pop=2, n_disc=3, cutoff=5):
    """
    QMC結果にディベートルールを適用して最終5頭を選抜

    ルール:
    1. QMC TOP8を候補プールとする（TOP5ではなく）
    2. 候補プールから消しルールで除外
    3. 残りから堅実枠(人気上位)とNNの穴枠を選抜
    """
    if 'ninki' not in mc.columns:
        if 'odds' in mc.columns:
            mc = mc.copy()
            mc['ninki'] = mc['odds'].rank(method='first')
        else:
            mc = mc.copy()
            mc['ninki'] = range(1, len(mc) + 1)

    # 候補プール: QMC TOP8
    pool = mc.head(8).copy()

    # 各馬の特徴量をマージ
    pool_with_feat = []
    for _, r in pool.iterrows():
        u = int(r['umaban'])
        fd = rf[rf['umaban'] == u]
        if len(fd) == 0:
            continue
        fd = fd.iloc[0]

        # 消しフラグ判定
        flags = []
        prev_cl = fd.get('prev_race_class', 4)
        ema_fin = fd.get('ema_finish', 3)
        odds_val = r.get('odds', 5) if pd.notna(r.get('odds')) else 5
        ninki_val = r.get('ninki', 3) if pd.notna(r.get('ninki')) else 3
        past_cnt = fd.get('past_count', 10)
        ema_ag = fd.get('ema_agari', 2)
        same_d = fd.get('same_dist_finish', 3)

        # 消しルール①: 前走クラス低い & 人気薄
        if prev_cl <= 3 and ninki_val > 5:
            flags.append('class_wall')
        # 消しルール②: EMA着順が悪い
        if ema_fin > 4.5:
            flags.append('ema_bad')
        # 消しルール③: オッズ高すぎ（NNの穴馬バイアス）
        if odds_val > 20 and ninki_val > 8:
            flags.append('longshot_bias')
        # 消しルール④: 同距離実績悪い & 末脚不足
        if same_d > 4.0 and ema_ag < 1.0:
            flags.append('no_aptitude')

        pool_with_feat.append({
            'umaban': u,
            'horse_name': r['horse_name'],
            'ninki': ninki_val,
            'odds': odds_val,
            'expected_rank': r['expected_rank'],
            'win_prob': r['win_prob'],
            'top3_prob': r['top3_prob'],
            'flags': flags,
            'n_flags': len(flags),
        })

    pool_df = pd.DataFrame(pool_with_feat)

    # 消しフラグなしの馬を優先、フラグありは後回し
    clean = pool_df[pool_df['n_flags'] == 0].copy()
    flagged = pool_df[pool_df['n_flags'] > 0].copy()

    # 堅実枠: 人気上位(<=cutoff)からQMC順でn_pop頭
    pop_pool = clean[clean['ninki'] <= cutoff].sort_values('expected_rank')
    pop = pop_pool.head(n_pop)

    # 足りなければフラグ付きからも補充（人気順）
    if len(pop) < n_pop:
        extra = flagged[flagged['ninki'] <= cutoff].sort_values('ninki')
        pop = pd.concat([pop, extra]).head(n_pop)

    # さらに足りなければcutoff外からも
    if len(pop) < n_pop:
        remaining = clean[~clean['umaban'].isin(pop['umaban'])].sort_values('ninki')
        pop = pd.concat([pop, remaining]).head(n_pop)

    pop_umabans = set(pop['umaban'].astype(int))

    # 穴枠: 堅実枠以外からQMC順でn_disc頭（消しフラグなし優先）
    disc_pool = clean[~clean['umaban'].isin(pop_umabans)].sort_values('expected_rank')
    disc = disc_pool.head(n_disc)

    # 足りなければフラグ付きから（フラグ少ない順→QMC順）
    if len(disc) < n_disc:
        extra = flagged[~flagged['umaban'].isin(pop_umabans) & ~flagged['umaban'].isin(disc['umaban'])].sort_values(['n_flags', 'expected_rank'])
        disc = pd.concat([disc, extra]).head(n_disc)

    # QMC TOP8外からも候補補充（消し馬を外した結果、枠が空いた場合）
    selected = pd.concat([pop, disc])
    if len(selected) < n_pop + n_disc:
        rest = mc[~mc['umaban'].isin(selected['umaban'].values)].head(n_pop + n_disc - len(selected))
        # restにninki追加
        if 'ninki' not in rest.columns:
            rest = rest.copy()
            rest['ninki'] = rest['odds'].rank(method='first') if 'odds' in rest.columns else range(1, len(rest)+1)
        rest_rows = []
        for _, r in rest.iterrows():
            rest_rows.append({
                'umaban': int(r['umaban']),
                'horse_name': r['horse_name'],
                'ninki': r.get('ninki', 99),
                'odds': r.get('odds', 999),
                'expected_rank': r['expected_rank'],
                'win_prob': r.get('win_prob', 0),
                'top3_prob': r.get('top3_prob', 0),
                'flags': [],
                'n_flags': 0,
            })
        selected = pd.concat([selected, pd.DataFrame(rest_rows)])

    return selected.head(n_pop + n_disc), pop_umabans


def run():
    print('Loading saved QMC data...')

    # 保存済みのpklから読み込み（analyze_qmc_errors.pyで保存済み）
    df_over = pd.read_pickle('data/qmc_overrated.pkl')
    df_miss = pd.read_pickle('data/qmc_missed.pkl')
    df_correct = pd.read_pickle('data/qmc_correct.pkl')

    # 全データ結合
    df_over['category'] = 'overrated'
    df_miss['category'] = 'missed'
    df_correct['category'] = 'correct'
    df_all = pd.concat([df_over, df_miss, df_correct], ignore_index=True)

    races = df_all['race_id'].unique()
    print(f'Total races: {len(races)}')

    # === 各方式でバックテスト ===
    methods = {
        'V1従来(TOP5)': 'v1',
        '堅2穴3(cutoff5)': 'split',
        'ルール適用(堅2穴3)': 'rules',
        'ルール適用(堅3穴2)': 'rules32',
    }

    results = {m: {'overlap_2': 0, 'overlap_1': 0, 'overlap_0': 0,
                    'win_hit': 0, 'ana_hit': 0, 'total': 0,
                    'tansho_pay': 0, 'tansho_cost': 0} for m in methods}

    for rid in races:
        race = df_all[df_all['race_id'] == rid]
        actual_top3 = set(race[race['in_actual_top3']]['umaban'].astype(int))
        actual_1st = race[race['actual_finish'] == 1]
        if len(actual_1st) == 0:
            continue
        a1 = int(actual_1st.iloc[0]['umaban'])
        a1_odds = float(actual_1st.iloc[0]['odds']) if pd.notna(actual_1st.iloc[0].get('odds')) else 1

        # QMC順にソート
        race_sorted = race.sort_values('expected_rank')

        for method_name, method_key in methods.items():
            if method_key == 'v1':
                sel = set(race_sorted.head(5)['umaban'].astype(int))
            elif method_key == 'split':
                # 堅2穴3（ルールなし）
                if 'ninki' in race_sorted.columns:
                    pop = race_sorted[race_sorted['ninki'] <= 5].head(2)
                    disc = race_sorted[race_sorted['ninki'] > 5].head(3)
                    sel_df = pd.concat([pop, disc])
                    sel = set(sel_df['umaban'].astype(int))
                else:
                    sel = set(race_sorted.head(5)['umaban'].astype(int))
            elif method_key in ('rules', 'rules32'):
                n_pop = 2 if method_key == 'rules' else 3
                n_disc = 3 if method_key == 'rules' else 2
                # ルール適用
                pool = race_sorted.head(8).copy()
                clean_uma = []
                flagged_uma = []

                for _, r in pool.iterrows():
                    u = int(r['umaban'])
                    flags = 0
                    prev_cl = r.get('prev_race_class', 4)
                    ema_fin = r.get('ema_finish', 3)
                    odds_val = r.get('odds', 5) if pd.notna(r.get('odds')) else 5
                    ninki_val = r.get('ninki', 3) if pd.notna(r.get('ninki')) else 3
                    same_d = r.get('same_dist_finish', 3) if pd.notna(r.get('same_dist_finish')) else 3
                    ema_ag = r.get('ema_agari', 2) if pd.notna(r.get('ema_agari')) else 2

                    if prev_cl <= 3 and ninki_val > 5: flags += 1
                    if ema_fin > 4.5: flags += 1
                    if odds_val > 20 and ninki_val > 8: flags += 1
                    if same_d > 4.0 and ema_ag < 1.0: flags += 1

                    if flags == 0:
                        clean_uma.append((u, ninki_val, r['expected_rank']))
                    else:
                        flagged_uma.append((u, ninki_val, r['expected_rank'], flags))

                # 堅実枠
                pop_candidates = [x for x in clean_uma if x[1] <= 5]
                pop_candidates.sort(key=lambda x: x[2])  # QMC順
                pop_sel = [x[0] for x in pop_candidates[:n_pop]]

                # 足りなければフラグ付きから
                if len(pop_sel) < n_pop:
                    extra = [x for x in flagged_uma if x[1] <= 5]
                    extra.sort(key=lambda x: x[1])
                    for x in extra:
                        if len(pop_sel) >= n_pop: break
                        if x[0] not in pop_sel: pop_sel.append(x[0])

                # まだ足りなければ
                if len(pop_sel) < n_pop:
                    for x in sorted(clean_uma, key=lambda x: x[1]):
                        if len(pop_sel) >= n_pop: break
                        if x[0] not in pop_sel: pop_sel.append(x[0])

                # 穴枠
                disc_candidates = [x for x in clean_uma if x[0] not in pop_sel]
                disc_candidates.sort(key=lambda x: x[2])  # QMC順
                disc_sel = [x[0] for x in disc_candidates[:n_disc]]

                if len(disc_sel) < n_disc:
                    extra = [x for x in flagged_uma if x[0] not in pop_sel and x[0] not in disc_sel]
                    extra.sort(key=lambda x: (x[3], x[2]))
                    for x in extra:
                        if len(disc_sel) >= n_disc: break
                        disc_sel.append(x[0])

                sel = set(pop_sel + disc_sel)
                # まだ足りなければTOP8外から
                if len(sel) < n_pop + n_disc:
                    for _, r in race_sorted.iterrows():
                        if len(sel) >= n_pop + n_disc: break
                        u = int(r['umaban'])
                        if u not in sel: sel.add(u)

            overlap = len(sel & actual_top3)
            r_data = results[method_name]
            r_data['total'] += 1
            if overlap >= 2: r_data['overlap_2'] += 1
            elif overlap >= 1: r_data['overlap_1'] += 1
            else: r_data['overlap_0'] += 1
            if a1 in sel: r_data['win_hit'] += 1

            # 単勝ROI
            r_data['tansho_cost'] += len(sel) * 100
            if a1 in sel:
                r_data['tansho_pay'] += a1_odds * 100

            # 穴馬的中
            for u in sel:
                if u in actual_top3:
                    horse_odds = race[race['umaban'] == u]['odds'].values
                    if len(horse_odds) > 0 and pd.notna(horse_odds[0]) and horse_odds[0] >= 10:
                        r_data['ana_hit'] += 1

    # === 結果表示 ===
    print(f'\n{"="*90}')
    print(f'  ディベートルール バックテスト結果 ({len(races)}レース)')
    print(f'{"="*90}')

    print(f'\n  {"方式":<22s} {"占有率":>6s} {"2/3+":>6s} {"1/3":>5s} {"0/3":>5s} {"1着含有":>7s} {"穴的中":>6s} {"単勝ROI":>8s}')
    print(f'  {"-"*75}')

    for method_name in methods:
        r = results[method_name]
        n = r['total']
        ovr = r['overlap_2'] / n * 100
        o1 = r['overlap_1']
        o0 = r['overlap_0']
        win = r['win_hit'] / n * 100
        roi = r['tansho_pay'] / r['tansho_cost'] * 100 if r['tansho_cost'] > 0 else 0
        ana = r['ana_hit']
        print(f'  {method_name:<22s} {ovr:>5.1f}% {r["overlap_2"]:>5d} {o1:>5d} {o0:>5d} {win:>6.1f}% {ana:>5d} {roi:>7.1f}%')


if __name__ == '__main__':
    run()
