"""
TARGET frontier JV UM_DATA (馬マスタ) バイナリパーサー

UM_DATA/{year}/UM{year}{code}.DAT から血統情報・生産者情報を抽出する。

レコード仕様 (1609バイト固定長, cp932エンコーディング):
-----------------------------------------------
Offset  Size  Field
0       2     RecordSpec ('UM')
2       1     DataKubun
3       8     MakeDate (YYYYMMDD)
11      10    KettoNum (血統登録番号)
46      36    Bamei (馬名, cp932)
82      36    BameiKana (馬名カナ, cp932)
118     36    BameiEng (馬名英字, cp932)
--- 3代血統情報 (Ketto3Info): 14頭 × 46バイト = 644バイト ---
204     10    Ketto3Info[0].KettoNum  (父 / Sire)
214     36    Ketto3Info[0].Bamei
250     10    Ketto3Info[1].KettoNum  (母 / Dam)
260     36    Ketto3Info[1].Bamei
296     10    Ketto3Info[2].KettoNum  (父父 / SS)
306     36    Ketto3Info[2].Bamei
342     10    Ketto3Info[3].KettoNum  (父母 / SD)
352     36    Ketto3Info[3].Bamei
388     10    Ketto3Info[4].KettoNum  (母父 / BMS)
398     36    Ketto3Info[4].Bamei
434     10    Ketto3Info[5].KettoNum  (母母 / DD)
444     36    Ketto3Info[5].Bamei
...           (6-13: 3代目 SSS,SSD,SDS,SDD,DSS,DSD,DDS,DDD)
848     end of Ketto3Info (204 + 14*46 = 848)
--- 馬主・生産者情報 ---
848     6     BanushiCode (馬主コード)
854     28    BanushiName (馬主名, cp932)
882     6     BreederCode (生産者コード)
888     2     BreederCodeSub (生産者サブコード)
890     70    BreederName (生産者名, cp932)
960     20    SanchiName (産地名, cp932)
-----------------------------------------------
"""

import pandas as pd
from pathlib import Path


UM_RECORD_SIZE = 1609
UM_DATA_ROOT = Path('C:/UM_DATA')

# 3代血統ツリーのインデックス名
PEDIGREE_LABELS = [
    'sire', 'dam', 'ss', 'sd', 'bms', 'dd',
    'sss', 'ssd', 'sds', 'sdd', 'dss', 'dsd', 'dds', 'ddd',
]


def _ascii_field(rec, start, length):
    """ASCIIフィールドを読み取り、前後空白を除去"""
    return rec[start:start + length].decode('ascii', errors='replace').strip()


def _cp932_field(rec, start, length):
    """cp932フィールドを読み取り、全角/半角スペースを除去"""
    return rec[start:start + length].decode('cp932', errors='replace').rstrip('　').rstrip()


def parse_um_record(rec):
    """
    UMレコード(1609バイト)をパースして辞書を返す。

    Returns
    -------
    dict or None
        パース結果。レコードサイズ不正なら None。
    """
    if len(rec) != UM_RECORD_SIZE:
        return None

    spec = rec[0:2].decode('ascii', errors='replace')
    if spec != 'UM':
        return None

    r = {}
    r['ketto_num'] = _ascii_field(rec, 11, 10)
    r['horse_name'] = _cp932_field(rec, 46, 36)

    # --- 3代血統 (14頭) ---
    base = 204
    for i, label in enumerate(PEDIGREE_LABELS):
        offset = base + i * 46
        r[f'{label}_id'] = _ascii_field(rec, offset, 10)
        r[f'{label}_name'] = _cp932_field(rec, offset + 10, 36)

    # --- 馬主 ---
    r['banushi_code'] = _ascii_field(rec, 848, 6)
    r['banushi_name'] = _cp932_field(rec, 854, 28)

    # --- 生産者 ---
    r['breeder_code'] = _ascii_field(rec, 882, 6)
    r['breeder_name'] = _cp932_field(rec, 890, 70)

    # --- 産地 ---
    r['sanchi_name'] = _cp932_field(rec, 960, 20)

    return r


def load_um_file(filepath):
    """
    単一UMファイルを読み込んでレコードのリストを返す。

    Parameters
    ----------
    filepath : str or Path
        UMファイルのパス

    Returns
    -------
    list[dict]
        パース済みレコードのリスト
    """
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        data = f.read()

    num_records = len(data) // UM_RECORD_SIZE
    if len(data) % UM_RECORD_SIZE != 0:
        print(f"Warning: {filepath.name} size {len(data)} "
              f"is not a multiple of {UM_RECORD_SIZE}")

    records = []
    for i in range(num_records):
        rec = data[i * UM_RECORD_SIZE:(i + 1) * UM_RECORD_SIZE]
        parsed = parse_um_record(rec)
        if parsed:
            records.append(parsed)

    return records


def load_um_data(um_data_dir=None, years=None, columns=None):
    """
    全UM_DATAファイルを読み込み、血統・生産者情報のDataFrameを返す。

    Parameters
    ----------
    um_data_dir : str or Path, optional
        UM_DATAのルートディレクトリ。デフォルト 'C:/UM_DATA'。
    years : iterable of int, optional
        対象年の範囲。None なら全年を自動検出。
    columns : list of str, optional
        返すカラムを限定。None なら主要カラムのみ:
        ketto_num, horse_name, sire_id, sire_name, bms_id, bms_name,
        breeder_code, breeder_name

    Returns
    -------
    pd.DataFrame
        ketto_num をインデックスとする血統情報DataFrame。
        同一 ketto_num が複数ファイルに存在する場合は最新を保持。
    """
    root = Path(um_data_dir) if um_data_dir else UM_DATA_ROOT

    if years is not None:
        year_dirs = [root / str(y) for y in years]
        year_dirs = [d for d in year_dirs if d.exists()]
    else:
        year_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: d.name,
        )

    all_records = []
    for year_dir in year_dirs:
        for um_file in sorted(year_dir.glob('UM*.DAT')):
            records = load_um_file(um_file)
            all_records.extend(records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # 同一 ketto_num の重複を除去 (後に読んだ方=最新を保持)
    df = df.drop_duplicates(subset='ketto_num', keep='last')

    # デフォルトカラム選択
    default_cols = [
        'ketto_num', 'horse_name',
        'sire_id', 'sire_name',
        'bms_id', 'bms_name',
        'breeder_code', 'breeder_name',
    ]

    if columns == '__all__':
        use_cols = [c for c in df.columns if c != 'ketto_num']
        df = df.set_index('ketto_num')
        return df
    elif columns is not None:
        use_cols = [c for c in columns if c in df.columns]
    else:
        use_cols = [c for c in default_cols if c in df.columns]

    df = df[use_cols].copy()
    df = df.set_index('ketto_num')

    return df


def load_um_data_full(um_data_dir=None, years=None):
    """
    全フィールド(3代血統14頭+馬主+生産者+産地)を含む完全版DataFrameを返す。

    Parameters
    ----------
    um_data_dir : str or Path, optional
        UM_DATAのルートディレクトリ。
    years : iterable of int, optional
        対象年の範囲。

    Returns
    -------
    pd.DataFrame
        ketto_num をインデックスとする全フィールドDataFrame。
    """
    return load_um_data(um_data_dir=um_data_dir, years=years,
                        columns='__all__')


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Loading UM_DATA (horse master)...")
    df = load_um_data(years=range(2020, 2025))

    print(f"\nTotal horses: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(10).to_string())

    # 種牡馬別頭数
    print(f"\nTop 10 sires by offspring count:")
    sire_counts = df['sire_name'].value_counts().head(10)
    print(sire_counts.to_string())

    # 母父別頭数
    print(f"\nTop 10 BMS by offspring count:")
    bms_counts = df['bms_name'].value_counts().head(10)
    print(bms_counts.to_string())

    # 生産者別頭数
    print(f"\nTop 10 breeders by horse count:")
    breeder_counts = df['breeder_name'].value_counts().head(10)
    print(breeder_counts.to_string())
