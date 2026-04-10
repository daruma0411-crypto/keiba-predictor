"""
TARGET frontier JV CSV Parser
TARGETからエクスポートされたCSV（cp932）をパースしてDataFrameに変換する
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


def parse_time(time_str):
    """走破タイムをパース（例: '2.16.6' → 136.6秒, '1088' → 108.8秒）"""
    if pd.isna(time_str) or str(time_str).strip() == '':
        return np.nan
    time_str = str(time_str).strip()
    # '2.16.6' 形式 (分.秒.10分の1秒)
    m = re.match(r'(\d+)\.(\d+)\.(\d+)', time_str)
    if m:
        minutes, seconds, tenths = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return minutes * 60 + seconds + tenths * 0.1
    # '1088' 形式 (10分の1秒単位)
    m = re.match(r'^(\d{3,4})$', time_str)
    if m:
        return int(m.group(1)) / 10.0
    return np.nan


def parse_finish(finish_str):
    """着順をパース（全角数字→半角数字、取消/除外等→NaN）"""
    if pd.isna(finish_str):
        return np.nan
    finish_str = str(finish_str).strip()
    # 全角数字→半角
    finish_str = finish_str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    try:
        return int(finish_str)
    except ValueError:
        return np.nan


def parse_weight(weight_str):
    """馬体重をパース"""
    if pd.isna(weight_str):
        return np.nan
    weight_str = str(weight_str).strip()
    try:
        return int(weight_str)
    except ValueError:
        return np.nan


def parse_weight_change(change_str):
    """馬体重増減をパース（例: '+8' → 8, '-4' → -4）"""
    if pd.isna(change_str):
        return np.nan
    change_str = str(change_str).strip()
    try:
        return int(change_str)
    except ValueError:
        return np.nan


def parse_odds(odds_str):
    """オッズをパース"""
    if pd.isna(odds_str):
        return np.nan
    odds_str = str(odds_str).strip()
    try:
        return float(odds_str)
    except ValueError:
        return np.nan


def parse_jockey_weight(jw_str):
    """斤量をパース（例: ' 57 ' → 57.0）"""
    if pd.isna(jw_str):
        return np.nan
    jw_str = str(jw_str).strip()
    try:
        return float(jw_str)
    except ValueError:
        return np.nan


def load_target_csv(filepath, encoding='cp932'):
    """
    TARGETのCSVを読み込んでDataFrameに変換する

    Parameters
    ----------
    filepath : str or Path
        CSVファイルパス
    encoding : str
        文字エンコーディング（デフォルト: cp932）

    Returns
    -------
    pd.DataFrame
        パース済みのDataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # CSVを読み込み（ヘッダー付き）
    df = pd.read_csv(filepath, encoding=encoding, dtype=str, na_values=['', ' '])

    # カラム名の前後空白を除去
    df.columns = df.columns.str.strip()

    # 先頭のMカラム（マーク列）があれば除去
    if df.columns[0] == 'Ｍ':
        df = df.drop(columns=['Ｍ'])

    # 末尾の矢印カラムを除去
    for col in ['→', '←']:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def clean_race_data(df):
    """
    DataFrameを整形して型変換する

    Parameters
    ----------
    df : pd.DataFrame
        load_target_csvで読み込んだ生データ

    Returns
    -------
    pd.DataFrame
        整形済みDataFrame
    """
    result = pd.DataFrame()

    # === レース情報 ===
    if '日付(yyyy.mm.dd)' in df.columns:
        # "2024. 1.21" のようにスペースが入る場合がある
        date_str = df['日付(yyyy.mm.dd)'].str.strip().str.strip('"').str.replace(r'\s+', '', regex=True)
        result['date'] = pd.to_datetime(date_str, format='%Y.%m.%d', errors='coerce')
    elif '日付S' in df.columns:
        date_str = df['日付S'].str.strip().str.replace(r'\s+', '', regex=True)
        result['date'] = pd.to_datetime(date_str, format='%Y.%m.%d', errors='coerce')
    elif '日付' in df.columns:
        result['date'] = pd.to_datetime(df['日付'].str.strip(), errors='coerce')

    result['venue'] = df.get('場所', df.get('開催', '')).astype(str).str.strip()
    result['race_num'] = pd.to_numeric(df.get('Ｒ', np.nan), errors='coerce')
    result['race_name'] = df.get('レース名', '').astype(str).str.strip()
    result['class_name'] = df.get('クラス名', '').astype(str).str.strip()

    # レースID（一意識別子）
    if 'レースID(新/馬番無)' in df.columns:
        result['race_id'] = df['レースID(新/馬番無)'].astype(str).str.strip()
    elif 'レースID(新)' in df.columns:
        result['race_id'] = df['レースID(新)'].astype(str).str.strip()

    # === コース条件 ===
    result['surface'] = df.get('芝・ダ', df.get('芝・ダート', '')).astype(str).str.strip()
    result['distance'] = pd.to_numeric(df.get('距離', np.nan), errors='coerce')
    result['track_condition'] = df.get('馬場状態', '').astype(str).str.strip()
    result['weather'] = df.get('天気', df.get('天候', '')).astype(str).str.strip()
    result['num_runners'] = pd.to_numeric(df.get('頭数', df.get('出走頭数', np.nan)), errors='coerce')

    # === 馬情報 ===
    result['horse_name'] = df.get('馬名', '').astype(str).str.strip()
    if '血統登録番号' in df.columns:
        result['horse_id'] = df['血統登録番号'].astype(str).str.strip()
    result['sex'] = df.get('性別', '').astype(str).str.strip()
    result['age'] = pd.to_numeric(df.get('年齢', np.nan), errors='coerce')
    result['post_position'] = pd.to_numeric(df.get('枠番', np.nan), errors='coerce')
    result['horse_number'] = pd.to_numeric(df.get('馬番', np.nan), errors='coerce')
    result['career'] = pd.to_numeric(df.get('キャリア', np.nan), errors='coerce')

    # 斤量
    result['jockey_weight'] = df.get('斤量', '').apply(parse_jockey_weight)

    # 馬体重
    result['weight'] = df.get('馬体重', '').apply(parse_weight)
    result['weight_change'] = df.get('馬体重増減', '').apply(parse_weight_change)

    # === 騎手・調教師 ===
    # 騎手カラムが2つある場合（略称とフル）、最初のものを使用
    jockey_cols = [c for c in df.columns if c == '騎手']
    if jockey_cols:
        result['jockey'] = df[jockey_cols[0]].astype(str).str.strip()
    if '騎手コード' in df.columns:
        result['jockey_id'] = df['騎手コード'].astype(str).str.strip()

    trainer_cols = [c for c in df.columns if c == '調教師']
    if trainer_cols:
        result['trainer'] = df[trainer_cols[0]].astype(str).str.strip()
    if '調教師コード' in df.columns:
        result['trainer_id'] = df['調教師コード'].astype(str).str.strip()

    # === 血統 ===
    result['sire'] = df.get('種牡馬', '').astype(str).str.strip()
    result['broodmare_sire'] = df.get('母父馬', '').astype(str).str.strip()

    # === 成績 ===
    result['finish'] = df.get('着順', '').apply(parse_finish)
    result['popularity'] = pd.to_numeric(df.get('人気', np.nan), errors='coerce')
    result['win_odds'] = df.get('単勝オッズ', '').apply(parse_odds)

    # タイム
    time_col = df.get('走破タイム', df.get('タイム', pd.Series(dtype=str)))
    result['time'] = time_col.apply(parse_time)
    result['margin'] = df.get('着差', '').apply(parse_odds)

    # 上がり3F
    result['last_3f'] = pd.to_numeric(df.get('上り3F', np.nan), errors='coerce')
    result['last_3f_rank'] = pd.to_numeric(df.get('上り3F順', df.get('上り3F順位', np.nan)), errors='coerce')

    # 通過順位
    for i, col in enumerate(['1角', '2角', '3角', '4角'], 1):
        if col in df.columns:
            result[f'corner_{i}'] = pd.to_numeric(df[col], errors='coerce')

    # 脚質
    result['running_style'] = df.get('脚質', '').astype(str).str.strip()

    # PCI
    result['pci'] = pd.to_numeric(df.get('PCI', np.nan), errors='coerce')
    result['rpci'] = pd.to_numeric(df.get('RPCI', np.nan), errors='coerce')

    # Ave-3F
    result['ave_3f'] = pd.to_numeric(df.get('Ave-3F', np.nan), errors='coerce')

    # === 前走情報 ===
    if '前走日付' in df.columns:
        result['prev_date'] = pd.to_datetime(df['前走日付'].str.strip().str.strip('"'), errors='coerce')
    result['prev_venue'] = df.get('前走場所', '').astype(str).str.strip()
    result['prev_finish'] = df.get('前走着順', '').apply(parse_finish)
    result['prev_popularity'] = pd.to_numeric(df.get('前走人気', np.nan), errors='coerce')
    result['prev_odds'] = df.get('前走単勝オッズ', '').apply(parse_odds)
    result['prev_last_3f'] = pd.to_numeric(df.get('前走上り3F', np.nan), errors='coerce')
    result['prev_weight'] = df.get('前走馬体重', '').apply(parse_weight)
    result['prev_running_style'] = df.get('前走脚質', '').astype(str).str.strip()

    # 間隔（前走からの週数）
    result['interval_weeks'] = pd.to_numeric(df.get('間隔', np.nan), errors='coerce')

    return result


def load_and_clean(filepath, encoding='cp932'):
    """CSVを読み込んで整形まで一括で行う"""
    raw = load_target_csv(filepath, encoding)
    return clean_race_data(raw)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_parser.py <csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    df = load_and_clean(filepath)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDate range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"Unique races: {df['race_id'].nunique() if 'race_id' in df.columns else 'N/A'}")
    print(f"\nSample:")
    print(df.head())
