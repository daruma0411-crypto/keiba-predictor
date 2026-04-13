"""
出馬表CSVパーサー（汎用入力）
TARGET JVエクスポートの出馬表CSV（1行1馬）を読み込み、
特徴量キャッシュと紐付けてレース用DataFrameを生成する。

CSVフォーマット (cp932, ヘッダなし, 33列):
  [0]  日付(YYMMDD)    [1] 場名    [2] R番号    [3] 馬番
  [4]  クラス          [5] 芝/ダ   [6] 距離      [7] 馬名
  [8]  性別            [9] 馬齢    [10] 騎手     [11] 斤量
  [12] 調教師         [13] 栗/美   [14] 馬主     [15] 生産者
  [16] 父             [17] 母      [18] 馬ID     [19] ?
  [20] 母父           [21] 毛色    [22] 枠番     [23] ?
  [24] ?              [25] ?       [26] 頭数     [27] 賞金?
  [28] ?              [29] ?       [30] オッズ?   [31] ?
  [32] レースID
"""

import pandas as pd
import numpy as np


# 場名 → コースキー変換テーブル (芝/ダ + 距離で決定)
VENUE_MAP = {
    '札幌': 'sapporo', '函館': 'hakodate', '福島': 'fukushima',
    '新潟': 'niigata', '東京': 'tokyo', '中山': 'nakayama',
    '中京': 'chukyo', '阪神': 'hanshin', '京都': 'kyoto',
    '小倉': 'kokura',
}


def parse_entry_csv(csv_path):
    """
    出馬表CSVを読み込み、レース情報と馬リストを返す

    Parameters
    ----------
    csv_path : str
        CSVファイルパス

    Returns
    -------
    race_info : dict
        レース情報 (date, venue, race_num, class_name, surface, distance, heads)
    entries : list of dict
        各馬の情報 (umaban, wakuban, horse_name, futan, ketto_num, sire, dam,
                    broodmare_sire, jockey, trainer, sex, age, odds)
    """
    with open(csv_path, 'rb') as f:
        raw = f.read()
    text = raw.decode('cp932')
    lines = text.strip().split('\r\n')

    entries = []
    race_info = None

    for line in lines:
        c = line.split(',')
        if len(c) < 30:
            continue

        # レース情報は1行目から取得
        if race_info is None:
            date_str = c[0].strip()
            yy = int(date_str[:2])
            mm = int(date_str[2:4])
            dd = int(date_str[4:6])
            race_date = pd.Timestamp(2000 + yy, mm, dd)
            venue = c[1].strip()
            race_num = int(c[2].strip())
            distance = int(c[6].strip())
            surface = c[5].strip()  # 芝 or ダ
            class_name = c[4].strip()
            heads = int(c[26].strip())

            venue_key = VENUE_MAP.get(venue, venue.lower())
            surface_key = 'turf' if surface == '芝' else 'dirt'
            course_key = f'{venue_key}_{surface_key}_{distance}'

            race_info = {
                'date': race_date,
                'date_str': f'{race_date:%Y-%m-%d}',
                'venue': venue,
                'venue_key': venue_key,
                'race_num': race_num,
                'class_name': class_name,
                'surface': surface,
                'surface_key': surface_key,
                'distance': distance,
                'heads': heads,
                'course_key': course_key,
                'race_id': f'{race_date:%Y%m%d}_{venue_key}_{race_num:02d}',
            }

        # 馬情報
        umaban = int(c[3].strip())
        wakuban = int(c[22].strip())
        horse_name = c[7].strip()
        futan = float(c[11].strip())
        ketto_num = c[18].strip()
        sire = c[16].strip()
        dam = c[17].strip()
        broodmare_sire = c[20].strip()
        jockey = c[10].strip()
        trainer = c[12].strip()
        sex = c[8].strip()
        age = int(c[9].strip())

        # オッズ (col[30], 0なら不明)
        try:
            odds_raw = float(c[30].strip())
            odds = odds_raw if odds_raw > 0 else np.nan
        except (ValueError, IndexError):
            odds = np.nan

        entries.append({
            'umaban': umaban,
            'wakuban': wakuban,
            'horse_name': horse_name,
            'futan': futan,
            'ketto_num': ketto_num,
            'sire': sire,
            'dam': dam,
            'broodmare_sire': broodmare_sire,
            'jockey': jockey,
            'trainer': trainer,
            'sex': sex,
            'age': age,
            'odds': odds,
        })

    return race_info, entries


def build_race_features(csv_path, feat_cache, prev_race_csv=None):
    """
    出馬表CSVと特徴量キャッシュからレース用DataFrameを構築する

    Parameters
    ----------
    csv_path : str
        出馬表CSVパス
    feat_cache : DataFrame
        特徴量キャッシュ (features_v9b_*.pkl)
    prev_race_csv : str, optional
        過去走CSV (interval_days/prev_dist_diff計算用、なければキャッシュから推定)

    Returns
    -------
    race_info : dict
        レース情報
    rf : DataFrame
        レース用特徴量DataFrame（Predictor.predict()に渡せる形式）
    missing : list
        特徴量キャッシュにヒットしなかった馬名リスト
    """
    race_info, entries = parse_entry_csv(csv_path)
    race_date = race_info['date']
    distance = race_info['distance']

    # Grade/Class score maps (module-level constants used in loop)
    _GRADE_SCORE = {'A': 8, 'B': 7, 'C': 6, 'L': 5, 'E': 4}
    _CLASS_SCORE = {
        '01': 4, '11': 4, '21': 4, '31': 4, '41': 4,
        '02': 3, '12': 3, '22': 3, '32': 3, '42': 3,
        '04': 2, '14': 2, '24': 2, '34': 2, '44': 2,
        '03': 1, '13': 1, '23': 1, '33': 1, '43': 1,
    }

    # grade_cd/class_cd をrace_idから引くためのキャッシュ（1回だけロード）
    _raw_cache = {}

    def _get_race_class(race_id, race_date_val):
        """race_idからgrade/classスコアを返す（生データから取得）"""
        grade_cd = ''
        class_cd = ''
        # まずfeat_cacheから探す
        if 'grade_cd' in feat_cache.columns:
            r = feat_cache[feat_cache['race_id'] == race_id]
            if len(r) > 0:
                grade_cd = str(r.iloc[0].get('grade_cd', '')).strip()
                class_cd = str(r.iloc[0].get('class_cd', '')).strip()
        # キャッシュにない場合、生データから
        if not grade_cd and not class_cd:
            yr = race_date_val.year if pd.notna(race_date_val) else 2026
            if yr not in _raw_cache:
                try:
                    from src.binary_parser import load_all_data
                    _raw_cache[yr] = load_all_data(years=[yr])
                except Exception:
                    _raw_cache[yr] = pd.DataFrame()
            raw = _raw_cache[yr]
            if len(raw) > 0:
                race = raw[raw['race_id'] == race_id]
                if len(race) > 0:
                    grade_cd = str(race.iloc[0].get('grade_cd', '')).strip()
                    class_cd = str(race.iloc[0].get('class_cd', '')).strip()
        if grade_cd in _GRADE_SCORE:
            return _GRADE_SCORE[grade_cd]
        if class_cd in _CLASS_SCORE:
            return _CLASS_SCORE[class_cd]
        return 1

    race_rows = []
    missing = []

    for entry in entries:
        hid = entry['ketto_num']
        ketto_10 = '20' + hid if len(hid) == 8 else hid

        horse_feat = feat_cache[
            (feat_cache['ketto_num'] == ketto_10) & (feat_cache['date'] < race_date)
        ]
        if len(horse_feat) == 0:
            horse_feat = feat_cache[
                (feat_cache['ketto_num'] == hid) & (feat_cache['date'] < race_date)
            ]
        if len(horse_feat) == 0:
            missing.append(entry['horse_name'])
            continue

        latest_row = horse_feat.sort_values('date').iloc[-1]
        row = latest_row.copy()

        # ============================================================
        # キャッシュ最新レコードの「結果」を織り込んで特徴量を更新
        # キャッシュの値は「そのレース出走時点」の特徴量であり、
        # 「そのレースの結果」はまだ反映されていない。
        # ここで1走分の結果を反映して「次走予測用」に更新する。
        # ============================================================
        alpha = 0.3  # EMA平滑化係数 (features.pyと同じ)
        latest_finish = latest_row.get('finish', 0)
        latest_pc = latest_row.get('past_count', 0)

        if latest_finish > 0 and latest_pc > 0:
            new_pc = latest_pc + 1
            row['past_count'] = new_pc

            # ema_finish: 指数移動平均に最新着順を織り込む
            old_ema = latest_row.get('ema_finish', latest_finish)
            row['ema_finish'] = alpha * latest_finish + (1 - alpha) * old_ema

            # ema_time_zscore: 同様
            # (time_zscoreはレース内標準化値。最新レースのzscoreはキャッシュにない
            #  ため近似として前回値を維持)

            # win_rate / top3_rate: 累積率を更新
            old_wins = latest_row.get('win_rate', 0) * latest_pc
            old_top3 = latest_row.get('top3_rate', 0) * latest_pc
            is_win = 1 if latest_finish == 1 else 0
            is_top3 = 1 if latest_finish <= 3 else 0
            row['win_rate'] = (old_wins + is_win) / new_pc
            row['top3_rate'] = (old_top3 + is_top3) / new_pc

            # avg_run_style: 通過順位から脚質推定を更新
            c3 = latest_row.get('avg_jyuni_3c', 0)
            c4 = latest_row.get('avg_jyuni_4c', 0)
            if c3 > 0 and c4 > 0:
                avg_corner = (c3 + c4) / 2
                if avg_corner <= 3:
                    style = 1.0  # 逃げ
                elif avg_corner <= 6:
                    style = 2.0  # 先行
                elif avg_corner <= 10:
                    style = 3.0  # 差し
                else:
                    style = 4.0  # 追込
                old_style = latest_row.get('avg_run_style', style)
                row['avg_run_style'] = (old_style * latest_pc + style) / new_pc

            # avg_jyuni_3c/4c: 移動平均更新
            for corner_col in ['avg_jyuni_3c', 'avg_jyuni_4c', 'avg_jyuni_1c', 'avg_jyuni_2c']:
                raw_col = corner_col.replace('avg_', '')  # jyuni_3c etc.
                old_avg = latest_row.get(corner_col, 0)
                if old_avg > 0:
                    # 近似: 新しい通過順は不明なので前回値を維持
                    pass

            # same_dist_finish / same_surface_finish: 同距離/同馬場なら更新
            latest_kyori = latest_row.get('kyori', 0)
            if latest_kyori == distance:
                old_sd = latest_row.get('same_dist_finish', np.nan)
                if pd.notna(old_sd):
                    row['same_dist_finish'] = (old_sd * (new_pc - 1) + latest_finish) / new_pc
                else:
                    row['same_dist_finish'] = float(latest_finish)
                row['has_same_dist'] = 1

        # 出馬表情報で上書き
        row['wakuban'] = entry['wakuban']
        row['umaban'] = entry['umaban']
        row['heads'] = race_info['heads']
        row['futan'] = entry['futan']
        row['horse_name'] = entry['horse_name']
        row['odds'] = entry['odds']
        row['date'] = race_date
        row['race_id'] = race_info['race_id']

        # interval_days
        latest_date = horse_feat['date'].max()
        if pd.notna(latest_date):
            row['interval_days'] = (race_date - latest_date).days

        # prev_dist_diff
        if 'kyori' in latest_row.index and pd.notna(latest_row.get('kyori')):
            row['prev_dist_diff'] = distance - latest_row['kyori']
        else:
            row['prev_dist_diff'] = 0

        # prev_race_class: 最新レコードのレース自体のクラスで更新
        row['prev_race_class'] = _get_race_class(
            latest_row.get('race_id', ''),
            latest_row.get('date', race_date)
        )

        # weighted_ema_finishは廃止（FEATURES_V9から除外済み）

        race_rows.append(row)

    rf = pd.DataFrame(race_rows).reset_index(drop=True)

    # --- Fix stale kisyu_code (jockey code) when jockey changed ---
    # The feature cache row inherits the jockey from the horse's LAST race.
    # If the entry CSV lists a different jockey, the kisyu_code will be wrong.
    # Build a reverse lookup (jockey name → kisyu_code) from the cache if
    # kisyu_name is available (added to cache in features.py build_all_features).
    if 'kisyu_name' in feat_cache.columns and len(rf) > 0:
        # Build name→code mapping: for each jockey name, take the most recent code
        jn = feat_cache[['kisyu_name', 'kisyu_code', 'date']].dropna(subset=['kisyu_name', 'kisyu_code'])
        jn = jn[jn['kisyu_name'] != '']
        if len(jn) > 0:
            jn_sorted = jn.sort_values('date')
            jockey_name_to_code = dict(zip(jn_sorted['kisyu_name'], jn_sorted['kisyu_code']))

            for i, entry in enumerate(entries):
                if i >= len(rf):
                    break
                csv_jockey = entry['jockey']
                if csv_jockey and csv_jockey in jockey_name_to_code:
                    new_code = jockey_name_to_code[csv_jockey]
                    old_code = rf.at[i, 'kisyu_code'] if 'kisyu_code' in rf.columns else None
                    if old_code != new_code:
                        rf.at[i, 'kisyu_code'] = new_code
                else:
                    # Jockey name not found in cache - could be a new jockey or name mismatch
                    pass
    else:
        # kisyu_name not in cache - cannot verify jockey changes
        # Note: kisyu_code, chokyosi_code, banusi_code are inherited from cache.
        # If jockey changed since last race, kisyu_code will be stale.
        # Rebuild the feature cache (with kisyu_name) to enable jockey code updates.
        pass

    # --- Ensure derived binary flags exist ---
    if len(rf) > 0:
        if 'has_same_dist' not in rf.columns and 'same_dist_finish' in rf.columns:
            rf['has_same_dist'] = rf['same_dist_finish'].notna().astype(int)
        if 'has_long_stretch' not in rf.columns and 'long_stretch_avg' in rf.columns:
            rf['has_long_stretch'] = rf['long_stretch_avg'].notna().astype(int)
        elif 'has_long_stretch' not in rf.columns:
            rf['has_long_stretch'] = 0

    return race_info, rf, missing
