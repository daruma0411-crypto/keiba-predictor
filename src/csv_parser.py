"""
TARGET CSVデータパーサー
TARGETのCSVエクスポート(131列, cp932)を読み込み、DataFrameに変換する。
バイナリパーサー(binary_parser.py)の後継。上がり3F/ペース/馬場等が取得可能。

CSVフォーマット:
  前半(0-48): 馬個別データ + レース共通ペースデータ
  後半(49-130): レースデータ(重複) + 馬個別データ(重複) + 血統等
"""

import pandas as pd
import numpy as np
from pathlib import Path


# カラム定義 — 2024桜花賞ワイドラトゥール(uma=1,waku=1,fin=6,odds=208.5)で全列照合済み
# 詳細: data/csv_column_map.md 参照
COL_MAP = {
    # === レース情報 (col 0-10) ===
    'year': 0,           # 下2桁 (24=2024)
    'month': 1,
    'day': 2,
    'kai': 3,            # 回次
    'venue': 4,          # 場所名 (阪神, 東京, etc.)
    'nichi': 5,          # 日次
    'race_num': 6,       # R番号
    'class_name': 7,     # クラス名 (桜花賞G1, 1勝ｸﾗｽ, etc.)
    'surface': 8,        # 芝 / ダ
    'kyori': 9,          # 距離(m)
    'baba': 10,          # 馬場状態 (良/稍/重/不良)

    # === 馬個別データ 前半 (col 11-39) ===
    'horse_name': 11,
    'sex': 12,           # 牡/牝/セ
    'barei': 13,         # 馬齢
    'jockey_name': 14,
    'futan': 15,         # 斤量(kg)
    'heads': 16,         # 頭数
    'umaban': 17,        # 馬番 (1-18) ★V1照合済み
    'finish': 18,        # 確定着順 ★V1照合済み。col19も同値(重複)
    # col19 = finish重複 (マッピング不要)
    'time_diff': 20,     # 着差(秒)
    'ninki': 21,         # 人気
    'time_sec': 22,      # 走破タイム(秒)
    'time_disp': 23,     # タイム表示 (1.32.5等)
    # col24-25: 不明/未使用
    'jyuni_1c': 26,      # 通過順1角 (0=1角なし)
    'jyuni_2c': 27,      # 通過順2角
    'jyuni_3c': 28,      # 通過順3角
    'bataijyu': 29,      # 馬体重(kg)
    'trainer_name': 30,  # 調教師名
    'tozai': 31,         # 所属 (美/栗)
    # col32 = 調教師コード (col117と同値)
    'kisyu_code': 33,    # 騎手コード
    'owner_name': 34,    # 馬主名
    'breeder': 35,       # 生産者名
    'sire': 36,          # 父
    'dam': 37,           # 母
    'broodmare_sire': 38, # 母父
    'birthday': 39,       # 生年月日 (YYYYMMDD)

    # === ペース・上がり (col 40-48) ===
    'odds': 40,          # 単勝オッズ ★V1照合済み
    'pace_front': 41,    # 前半タイム(秒) レース共通
    'pace_rear': 42,     # 後半タイム(秒) 馬個別
    'pace_front_std': 43, # 前半基準タイム(秒)
    'agari_3f': 44,      # 上がり3F(秒) 精密値
    'agari_3f_disp': 45, # 上がり3F(秒) 表示値
    'agari_rank': 46,    # 上がり3F順位
    'ave_3f': 47,        # Ave-3F (レース平均?)
    'agari_diff': 48,    # 上がり3F差

    # === レース詳細 後半 (col 49-79) ===
    'class_name_2': 57,  # クラス名(重複)
    'race_name': 58,     # レース名のみ (桜花賞)
    # col60 = 不明(数値 195等)
    'grade_cd': 61,      # グレードCD (A=G1, B=G2, C=G3, L=Listed, E=OP)
    'surface_2': 62,     # 芝ダ(重複)
    'track_cd': 63,      # トラックコード
    'track_cd_jv': 64,   # トラックコード(JV)
    'corner_count': 65,  # コーナー回数
    'kyori_2': 66,       # 距離(重複)
    'course_type': 67,   # コース区分 (B=外回り等)
    'baba_2': 68,        # 馬場(重複)
    'weather': 69,       # 天候
    'heads_2': 70,       # 頭数(重複)
    'full_gate': 71,     # フルゲート頭数
    'start_time': 72,    # 発走時刻
    'age_limit_cd': 78,  # 競走種別コード
    'weight_cd': 79,     # 重量コード (定量/別定/ハンデ)

    # === 馬個別データ 後半 (col 80-130) ===
    'race_id_old': 80,   # レースID(旧)
    'race_id_new': 81,   # レースID(新) YYYYMMDD+venue+kai+nichi+race+umaban
    'umaban_2': 82,      # 馬番(重複) ★V1照合済み
    'wakuban_2': 83,     # 枠番 ★V1照合済み (前半に枠番列なし、後半のみ)
    'horse_name_2': 84,
    'sex_2': 85,
    'barei_2': 86,
    'jockey_name_2': 87,
    'futan_2': 88,
    'blinker': 89,       # ブリンカー
    'finish_2': 90,      # 着順(重複)
    'nyusen_jyuni': 91,  # 入線着順
    'ijyo_cd': 92,       # 異常コード
    'time_diff_2': 93,   # 着差(重複)
    'ninki_2': 94,       # 人気(重複)
    'odds_2': 95,        # オッズ(重複)
    'time_sec_2': 96,    # タイム(重複)
    # col97-99: タイム表示/不明
    # col100-101: jyuni_1c重複
    'jyuni_2c_2': 102,   # 2角(重複)
    'jyuni_3c_2': 103,   # 3角(重複)
    'running_style': 104, # 脚質 (逃げ/先行/差し/追込)
    'agari_3f_2': 105,   # 上がり3F表示値(重複)
    'agari_rank_2': 106, # 上がり順位(重複)
    'ave_3f_2': 107,     # Ave-3F(重複)
    'pace_rear_2': 108,  # 後半タイム(重複)
    'pci_or_diff': 109,  # PCI or 上がり差
    'bataijyu_2': 110,   # 馬体重(重複)
    'zogen_sa': 111,     # 増減(kg)
    'trainer_name_2': 112,
    'tozai_2': 113,
    'prize': 114,        # 賞金(万円)
    'ketto_num': 115,    # 血統登録番号
    'kisyu_code_2': 116, # 騎手コード(重複) ※col32=調教師CDと同値
    'chokyosi_code': 117, # 調教師コード ※col33=騎手CDと同値
    'banusi_name': 118,  # 馬主名
    'seisansha': 119,    # 生産者名
    'sire_2': 120,       # 父(重複)
    'dam_2': 121,        # 母(重複)
    'bms_2': 122,        # 母父(重複)
    'sire_type': 123,    # 父タイプ (ナスルーラ系等)
    'bms_type': 124,     # 母父タイプ
    'keiro': 125,        # 毛色
    'birthday_2': 126,   # 生年月日(重複)
}


def load_csv_data(filepath='C:/TXT/10ne_deta', surface_filter=None):
    """
    TARGET CSVデータを読み込んでDataFrameに変換する

    Parameters
    ----------
    filepath : str
        CSVファイルパス
    surface_filter : str, optional
        '芝' or 'ダ' でフィルタ。Noneなら全件

    Returns
    -------
    DataFrame
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    text = raw.decode('cp932')
    lines = text.strip().split('\r\n')

    records = []
    for line in lines:
        c = line.split(',')
        if len(c) < 120:
            continue

        try:
            yr = int(c[0].strip())
            mo = int(c[1].strip())
            da = int(c[2].strip())
            date = pd.Timestamp(2000 + yr, mo, da)
        except (ValueError, IndexError):
            continue

        surface = c[COL_MAP['surface']].strip()
        if surface_filter and surface != surface_filter:
            continue

        # 着順
        try:
            finish = int(c[COL_MAP['finish']].strip())
        except (ValueError, IndexError):
            finish = 0

        if finish <= 0:
            continue

        # 数値変換ヘルパー
        def safe_float(col_name, default=np.nan):
            try:
                v = c[COL_MAP[col_name]].strip()
                if v == '' or v == '0':
                    return default if col_name not in ('finish', 'heads', 'umaban', 'wakuban') else float(v) if v != '' else default
                return float(v)
            except (ValueError, IndexError, KeyError):
                return default

        def safe_int(col_name, default=0):
            try:
                v = c[COL_MAP[col_name]].strip()
                return int(v) if v and v != '' else default
            except (ValueError, IndexError, KeyError):
                return default

        def safe_str(col_name, default=''):
            try:
                return c[COL_MAP[col_name]].strip()
            except (IndexError, KeyError):
                return default

        # レースID生成
        venue = safe_str('venue')
        race_num = safe_int('race_num')
        kai = safe_int('kai')
        nichi = safe_int('nichi')
        race_id = f'{date:%Y%m%d}_{venue}_{kai:02d}_{nichi:02d}_{race_num:02d}'

        # 脚質コード変換
        rs_str = safe_str('running_style')
        rs_map = {'逃げ': 1, '先行': 2, '差し': 3, '追込': 4, '': 0}
        running_style = rs_map.get(rs_str, 0)

        record = {
            'date': date,
            'race_id': race_id,
            'venue': venue,
            'race_num': race_num,
            'class_name': safe_str('class_name'),
            'surface': surface,
            'kyori': safe_int('kyori'),
            'baba': safe_str('baba'),
            'weather': safe_str('weather'),
            'course_type': safe_str('course_type'),
            'corner_count': safe_int('corner_count'),
            'heads': safe_int('heads'),
            'full_gate': safe_int('full_gate'),

            # 馬個別
            'horse_name': safe_str('horse_name'),
            'ketto_num': safe_str('ketto_num'),
            'sex': safe_str('sex'),
            'barei': safe_int('barei'),
            'finish': finish,
            'wakuban': safe_int('wakuban_2'),  # col83: 枠番(1-8) ★前半に枠番列なし、後半のみ
            'umaban': safe_int('umaban'),       # col17: 馬番(1-18)
            'futan': safe_float('futan'),
            'bataijyu': safe_float('bataijyu_2'),
            'zogen_sa': safe_float('zogen_sa'),
            'odds': safe_float('odds'),
            'ninki': safe_int('ninki'),
            'time_sec': safe_float('time_sec'),
            'time_diff': safe_float('time_diff'),
            'blinker': safe_str('blinker'),

            # 通過順
            'jyuni_1c': safe_int('jyuni_1c'),
            'jyuni_2c': safe_int('jyuni_2c'),
            'jyuni_3c': safe_int('jyuni_3c'),
            'jyuni_4c': safe_int('jyuni_3c'),  # 4角通過順はCSVに列なし→3角で代替（3角-4角は高相関、特に1600m以下）
            'running_style': running_style,
            'running_style_str': rs_str,

            # 上がり3F (最重要)
            'agari_3f': safe_float('agari_3f'),
            'agari_3f_disp': safe_float('agari_3f_disp'),
            'agari_rank': safe_int('agari_rank'),
            'agari_diff': safe_float('agari_diff'),
            'ave_3f': safe_float('ave_3f'),

            # ペース
            'pace_front': safe_float('pace_front'),
            'pace_rear': safe_float('pace_rear'),
            'pace_front_std': safe_float('pace_front_std'),

            # 賞金・クラス
            'prize': safe_float('prize'),
            'grade_cd': safe_str('grade_cd'),
            'age_limit_cd': safe_str('age_limit_cd'),
            'weight_cd': safe_str('weight_cd'),

            # 騎手・調教師
            'jockey_name': safe_str('jockey_name'),
            'kisyu_code': safe_str('kisyu_code'),
            'trainer_name': safe_str('trainer_name'),
            'chokyosi_code': safe_str('chokyosi_code'),
            'banusi_name': safe_str('banusi_name'),

            # 血統
            'sire': safe_str('sire'),
            'dam': safe_str('dam'),
            'broodmare_sire': safe_str('broodmare_sire'),
            'sire_type': safe_str('sire_type'),
            'bms_type': safe_str('bms_type'),

            # タイプ区分
            'is_turf': 1 if surface == '芝' else 0,
        }

        records.append(record)

    df = pd.DataFrame(records)
    print(f'Loaded: {len(df):,} records, {df["race_id"].nunique():,} races')
    print(f'  Period: {df["date"].min()} ~ {df["date"].max()}')
    print(f'  Surface: 芝={df["is_turf"].sum():,} / ダ={(~df["is_turf"].astype(bool)).sum():,}')

    return df


if __name__ == '__main__':
    df = load_csv_data()
    print(f'\nColumns: {df.columns.tolist()}')
    print(f'\nSample:')
    print(df.head(3).to_string())
