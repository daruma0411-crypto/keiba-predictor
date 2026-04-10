"""
TARGET frontier JV バイナリデータパーサー
SE_DATA/SU*.DAT (馬毎レース成績) + SR*.DAT (レース情報) を直接パースする
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import re


# JRA場コード
PLACE_CODES = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}

# 性別コード
SEX_CODES = {'1': '牡', '2': '牝', '3': 'セ'}

# 脚質区分
RUNNING_STYLE_CODES = {
    '1': '逃げ', '2': '先行', '3': '差し', '4': '追込', '5': '自在',
}

SE_RECORD_SIZE = 555


def parse_se_record(rec):
    """
    SEレコード(555バイト)をパースして辞書を返す

    フィールド定義 (JVData仕様 + TARGET拡張):
    -----------------------------------------------
    Offset  Size  Field
    0       2     RecordSpec ('SE')
    2       1     DataKubun
    3       8     MakeDate (YYYYMMDD)
    11      8     RaceDate (YYYYMMDD)
    19      2     JyoCD (場コード)
    21      2     Kaession (回)
    23      2     Nichiji (日)
    25      2     RaceNum (レース番号)
    27      1     Wakuban (枠番)
    28      2     Umaban (馬番)
    30      10    KettoNum (血統登録番号)
    40      36    Bamei (馬名, cp932)
    76      2     UmaKigoCD (馬記号コード)
    78      1     SexCD (性別)
    79      1     HinsyuCD (品種)
    80      2     KeiroCD (毛色)
    82      2     Barei (馬齢)
    84      1     TozaiCD (東西)
    85      5     ChokyosiCode (調教師コード)
    90      8     ChokyosiRyakusyo (調教師名略称, cp932)
    98      6     BanusiCode (馬主コード)
    104     40    BanusiName (馬主名, cp932)
    144     60    Fukusyoku (服色, cp932)
    204     84    Reserved/Padding (全角スペース)
    288     3     Futan (負担重量, ×10)
    291     3     FutanBefore (変更前負担重量)
    294     1     Blinker
    295     1     Reserved
    296     5     KisyuCode (騎手コード)
    301     5     KisyuCodeBefore
    306     8     KisyuRyakusyo (騎手名略称, cp932)
    314     8     KisyuRyakusyoBefore (cp932)
    322     1     MinaraiCD
    323     1     MinaraiCDBefore
    324     3     BaTaijyu (馬体重)
    327     1     ZogenFugo (増減符号)
    328     3     ZogenSa (増減差)
    331     1     IJyoCD (異常区分)
    332     2     NyusenJyuni (入線順位)
    334     2     KakuteiJyuni (確定着順)
    336     1     DochakuKubun
    337     1     DochakuTosu
    338     4     Time (走破タイム, 1/10秒)
    342     3     ChakusaCD (着差コード)
    345     3     ChakusaCDP
    348     3     ChakusaCDPP
    351     2     Jyuni1c (1角順位)
    353     2     Jyuni2c (2角順位)
    355     2     Jyuni3c (3角順位)
    357     2     Jyuni4c (4角順位)
    359     4     Odds (単勝オッズ, ×10)
    363     2     Ninki (人気)
    365     6     Honsyokin (本賞金)
    371     6     Fukasyokin (付加賞金)
    377     3     Reserved3
    380     3     Reserved4
    383     3     HaronTimeL4 (後4Fタイム)
    386     3     HaronTimeL3 (後3Fタイム)
    389     4     Unknown (TARGET拡張?)
    393     10    ChakuUma1_KettoNum
    403     36    ChakuUma1_Bamei (cp932)
    439     10    ChakuUma2_KettoNum
    449     36    ChakuUma2_Bamei (cp932)
    485     10    ChakuUma3_KettoNum
    495     36    ChakuUma3_Bamei (cp932)
    531     4     TimeDiff (タイム差)
    535     20    TARGET拡張データ
    -----------------------------------------------
    """
    if len(rec) != SE_RECORD_SIZE:
        return None

    def ascii_field(start, length):
        return rec[start:start+length].decode('ascii', errors='replace').strip()

    def cp932_field(start, length):
        return rec[start:start+length].decode('cp932', errors='replace').rstrip('　').rstrip()

    def int_field(start, length, default=None):
        val = ascii_field(start, length).lstrip('0') or '0'
        try:
            return int(val)
        except ValueError:
            return default

    def float_field_div10(start, length, default=None):
        val = ascii_field(start, length)
        try:
            return int(val) / 10.0
        except ValueError:
            return default

    r = {}
    r['race_date'] = ascii_field(3, 8)  # MakeDate使わず、レース日を使う
    # 実際のレース日は11:19
    r['race_date'] = ascii_field(11, 8)
    r['place_code'] = ascii_field(19, 2)
    r['place_name'] = PLACE_CODES.get(r['place_code'], r['place_code'])
    r['kai'] = int_field(21, 2)
    r['day'] = int_field(23, 2)
    r['race_num'] = int_field(25, 2)
    r['wakuban'] = int_field(27, 1)
    r['umaban'] = int_field(28, 2)
    r['ketto_num'] = ascii_field(30, 10)
    r['horse_name'] = cp932_field(40, 36)
    r['uma_kigo_cd'] = ascii_field(76, 2)
    r['sex_cd'] = ascii_field(78, 1)
    r['sex'] = SEX_CODES.get(r['sex_cd'], r['sex_cd'])
    r['hinsyu_cd'] = ascii_field(79, 1)
    r['keiro_cd'] = ascii_field(80, 2)
    r['barei'] = int_field(82, 2)
    r['tozai_cd'] = ascii_field(84, 1)
    r['chokyosi_code'] = ascii_field(85, 5)
    r['chokyosi_name'] = cp932_field(90, 8)
    r['banusi_code'] = ascii_field(98, 6)
    r['banusi_name'] = cp932_field(104, 40)

    r['futan'] = float_field_div10(288, 3)  # 斤量 (×10)
    r['blinker'] = ascii_field(294, 1)
    r['kisyu_code'] = ascii_field(296, 5)
    r['kisyu_name'] = cp932_field(306, 8)

    r['bataijyu'] = int_field(324, 3)  # 馬体重
    zogen_fugo = ascii_field(327, 1)
    zogen_sa = int_field(328, 3, 0)
    if zogen_fugo == '-':
        r['zogen_sa'] = -zogen_sa
    elif zogen_fugo == '+':
        r['zogen_sa'] = zogen_sa
    else:
        r['zogen_sa'] = 0

    r['ijyo_cd'] = ascii_field(331, 1)
    r['nyusen_jyuni'] = int_field(332, 2)
    r['kakutei_jyuni'] = int_field(334, 2)
    r['time'] = float_field_div10(338, 4)  # 走破タイム (1/10秒)
    r['chakusa_cd'] = ascii_field(342, 3)

    r['jyuni_1c'] = int_field(351, 2)
    r['jyuni_2c'] = int_field(353, 2)
    r['jyuni_3c'] = int_field(355, 2)
    r['jyuni_4c'] = int_field(357, 2)

    r['odds'] = float_field_div10(359, 4)  # 単勝オッズ (×10)
    r['ninki'] = int_field(363, 2)

    r['honsyokin'] = int_field(365, 6)
    r['fukasyokin'] = int_field(371, 6)

    r['haron_time_l4'] = float_field_div10(383, 3)
    r['haron_time_l3'] = float_field_div10(386, 3)

    r['time_diff'] = ascii_field(531, 4)

    # レースIDを生成 (場+回+日+R で一意)
    r['race_id'] = f"{r['race_date']}_{r['place_code']}_{r['kai']:02d}_{r['day']:02d}_{r['race_num']:02d}"

    return r


RA_RECORD_SIZE = 1272


def parse_ra_record(rec):
    """
    RAレコード(1272バイト)をパースして辞書を返す

    主要フィールド:
    -----------------------------------------------
    0:2     RecordSpec ('RA')
    2:3     DataKubun
    3:11    MakeDate
    11:19   RaceDate
    19:21   JyoCD (場コード)
    21:23   Kaession
    23:25   Nichiji
    25:27   RaceNum
    27:29   Heads (頭数)
    32:152  Hondai (レース名, cp932 120bytes)
    614     GradeCD ('A'=G1,'B'=G2,'C'=G3,'L'=リステッド,'E'=OP特別)
    697:701 Kyori (距離 4桁)
    705:709 TrackCD (TARGET拡張: 2400=芝, 1700/1800=ダート)
    713:717 Honsyokin1 (1着賞金, 万円)
    -----------------------------------------------
    """
    if len(rec) != RA_RECORD_SIZE:
        return None

    def ascii_field(start, length):
        return rec[start:start+length].decode('ascii', errors='replace').strip()

    def cp932_field(start, length):
        return rec[start:start+length].decode('cp932', errors='replace').rstrip('　').rstrip()

    def int_field(start, length, default=None):
        val = ascii_field(start, length).lstrip('0') or '0'
        try:
            return int(val)
        except ValueError:
            return default

    r = {}
    r['race_date'] = ascii_field(11, 8)
    r['place_code'] = ascii_field(19, 2)
    r['place_name'] = PLACE_CODES.get(r['place_code'], r['place_code'])
    r['kai'] = int_field(21, 2)
    r['day'] = int_field(23, 2)
    r['race_num'] = int_field(25, 2)
    r['race_name'] = cp932_field(32, 120)

    # グレード
    grade_raw = ascii_field(614, 1)
    grade_map = {'A': 'G1', 'B': 'G2', 'C': 'G3', 'L': 'Listed', 'E': 'OP'}
    r['grade_cd'] = grade_raw
    r['grade'] = grade_map.get(grade_raw, '')

    # 距離
    r['kyori'] = int_field(697, 4)

    # トラック (芝/ダート判定)
    # TARGET独自TrackCD:
    #   芝: 2400(内回り), 1800(外回り), 2000(直線) 等
    #   ダート: 1700
    #   障害: 5200, 5400 等
    track_raw = ascii_field(705, 5)
    r['track_cd_raw'] = track_raw
    track_num = track_raw.rstrip('AB ')
    try:
        track_int = int(track_num)
    except ValueError:
        track_int = 0

    if track_int == 1700:
        r['surface'] = 'ダート'
    elif track_int >= 5000:
        r['surface'] = '障害'
    else:
        r['surface'] = '芝'

    # 1着賞金
    r['prize_1st'] = int_field(713, 4)

    # レースID
    r['race_id'] = f"{r['race_date']}_{r['place_code']}_{r['kai']:02d}_{r['day']:02d}_{r['race_num']:02d}"

    return r


def load_sr_file(filepath):
    """SRファイル(レース情報バイナリ)を読み込んでDataFrameに変換"""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        data = f.read()

    num_records = len(data) // RA_RECORD_SIZE
    records = []
    for i in range(num_records):
        rec = data[i * RA_RECORD_SIZE:(i + 1) * RA_RECORD_SIZE]
        parsed = parse_ra_record(rec)
        if parsed:
            records.append(parsed)

    return pd.DataFrame(records)


def load_su_file(filepath):
    """SUファイル(馬毎成績バイナリ)を読み込んでDataFrameに変換"""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        data = f.read()

    num_records = len(data) // SE_RECORD_SIZE
    if len(data) % SE_RECORD_SIZE != 0:
        print(f"Warning: file size {len(data)} is not a multiple of {SE_RECORD_SIZE}")

    records = []
    for i in range(num_records):
        rec = data[i * SE_RECORD_SIZE:(i + 1) * SE_RECORD_SIZE]
        parsed = parse_se_record(rec)
        if parsed:
            records.append(parsed)

    return pd.DataFrame(records)


def load_all_data(se_data_dir='C:/SE_DATA', years=range(2015, 2026)):
    """全場のSE+RAデータを読み込む"""
    return load_hanshin_data(se_data_dir, years, place_code=None)


def load_hanshin_data(se_data_dir='C:/SE_DATA', years=range(2015, 2026), place_code='09'):
    """
    指定場のSE(馬毎成績)とRA(レース情報)を読み込み結合して返す

    Parameters
    ----------
    se_data_dir : str
        SE_DATAのルートディレクトリ
    years : range
        対象年
    place_code : str
        場コード (09=阪神)

    Returns
    -------
    pd.DataFrame
        SE+RA結合済みデータ
    """
    all_se = []
    all_ra = []

    for year in years:
        year_dir = Path(se_data_dir) / str(year)
        if not year_dir.exists():
            continue

        # SUファイル（馬毎成績）
        for su_file in sorted(year_dir.glob('SU*.DAT')):
            if place_code is not None:
                code_char = place_code[-1]
                if code_char not in su_file.stem[-2:]:
                    continue
            df = load_su_file(su_file)
            if len(df) > 0:
                if place_code is not None:
                    df = df[df['place_code'] == place_code].copy()
                if len(df) > 0:
                    all_se.append(df)

        # SRファイル（レース情報）
        for sr_file in sorted(year_dir.glob('SR*.DAT')):
            if place_code is not None:
                code_char = place_code[-1]
                if code_char not in sr_file.stem[-2:]:
                    continue
            df = load_sr_file(sr_file)
            if len(df) > 0:
                if place_code is not None:
                    df = df[df['place_code'] == place_code].copy()
                if len(df) > 0:
                    all_ra.append(df)

    if not all_se:
        return pd.DataFrame()

    se_df = pd.concat(all_se, ignore_index=True)
    se_df['date'] = pd.to_datetime(se_df['race_date'], format='%Y%m%d', errors='coerce')

    # 出走取消・除外を除外（kakutei_jyuni=0）
    before = len(se_df)
    se_df = se_df[(se_df['kakutei_jyuni'] > 0) | (se_df['kakutei_jyuni'].isna())].copy()
    removed = before - len(se_df)
    if removed > 0:
        print(f"  Removed {removed} scratched/excluded entries")

    # 障害レースを除外
    if 'surface' in se_df.columns:
        before2 = len(se_df)
        se_df = se_df[se_df['surface'] != '障害'].copy()
        removed2 = before2 - len(se_df)
        if removed2 > 0:
            print(f"  Removed {removed2} steeplechase entries")

    if all_ra:
        ra_df = pd.concat(all_ra, ignore_index=True)
        ra_df = ra_df.drop_duplicates(subset='race_id')
        # SE に RA のレース条件を結合
        ra_cols = ['race_id', 'race_name', 'grade_cd', 'grade', 'kyori', 'surface',
                   'track_cd_raw', 'prize_1st']
        se_df = se_df.merge(ra_df[ra_cols], on='race_id', how='left', suffixes=('', '_ra'))

    return se_df


def filter_open_class(df):
    """
    オープンクラス以上をフィルタする
    GradeCDまたは1着賞金ベースで判定
    """
    if 'grade_cd' in df.columns:
        # GradeCD: A=G1, B=G2, C=G3, L=Listed, E=OP
        open_grades = {'A', 'B', 'C', 'L', 'E'}
        mask = df['grade_cd'].isin(open_grades)
        # grade_cdがない(RA未結合)場合は賞金で補完
        if 'prize_1st' in df.columns:
            mask = mask | (df['prize_1st'] >= 1900)
        return df[mask].copy()
    else:
        # RAデータなし: 賞金で推定
        race_max_prize = df.groupby('race_id')['honsyokin'].max()
        open_race_ids = race_max_prize[race_max_prize >= 1900].index
        return df[df['race_id'].isin(open_race_ids)].copy()


if __name__ == '__main__':
    import sys

    print("Loading Hanshin data from SE_DATA + SR_DATA (2015-2025)...")
    df = load_hanshin_data()

    if len(df) == 0:
        print("No data found!")
        sys.exit(1)

    print(f"\nTotal records: {len(df)}")
    print(f"Unique races: {df['race_id'].nunique()}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # RA結合結果の確認
    if 'surface' in df.columns:
        print(f"\nSurface distribution:")
        print(df['surface'].value_counts())
        print(f"\nGrade distribution:")
        print(df['grade'].value_counts())

    # オープン以上フィルタ
    df_open = filter_open_class(df)
    print(f"\nOpen class and above: {len(df_open)} records, {df_open['race_id'].nunique()} races")

    print(f"\nSample open class data:")
    cols = ['date', 'race_name', 'grade', 'surface', 'kyori', 'horse_name',
            'kakutei_jyuni', 'odds', 'kisyu_name', 'futan', 'time']
    cols = [c for c in cols if c in df_open.columns]
    print(df_open[cols].head(20).to_string())
