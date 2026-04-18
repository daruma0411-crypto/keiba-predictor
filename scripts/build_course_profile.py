"""
特徴量キャッシュからコース別QMCプロファイルを自動生成する
脚質別成績・枠順バイアス・ペース傾向を統計的に算出
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import json

# ============================================================
# データ読み込み
# ============================================================
df = pd.read_pickle('data/features_v9b_2026.pkl')
df['date'] = pd.to_datetime(df['date'])

# 直近3年に限定
df = df[df['date'] >= '2023-01-01']

# コース直線距離マップ
STRAIGHT_MAP = {
    'nakayama': {1200: 310, 1600: 310, 1800: 310, 2000: 310, 2200: 310, 2500: 310},
    'hanshin':  {1200: 356, 1400: 356, 1600: 473, 1800: 356, 2000: 356, 2200: 356, 2400: 356},
    'tokyo':    {1400: 525, 1600: 525, 1800: 525, 2000: 525, 2400: 525},
    'kyoto':    {1200: 328, 1400: 328, 1600: 328, 1800: 398, 2000: 398, 2200: 398, 2400: 398},
    'fukushima':{1200: 292, 1800: 292, 2000: 292, 2600: 292},
    'niigata':  {1200: 359, 1400: 359, 1600: 359, 1800: 359, 2000: 359},
    'chukyo':   {1200: 413, 1400: 413, 1600: 413, 1800: 413, 2000: 413},
    'kokura':   {1200: 293, 1700: 293, 1800: 293, 2000: 293, 2600: 293},
    'sapporo':  {1200: 266, 1500: 266, 1800: 266, 2000: 266, 2600: 266},
    'hakodate': {1200: 262, 1800: 262, 2000: 262, 2600: 262},
}

# 場名マップ (race_id から逆引き)
def extract_venue_dist(race_id):
    """race_idからvenue_key, distanceを抽出"""
    parts = race_id.split('_')
    if len(parts) >= 3:
        return parts[1], None  # venue_key
    return None, None


def classify_style(avg_run_style):
    """脚質数値→カテゴリ"""
    if pd.isna(avg_run_style):
        return 'unknown'
    if avg_run_style <= 1.5:
        return 'nige'
    elif avg_run_style <= 2.3:
        return 'senkou'
    elif avg_run_style <= 3.0:
        return 'sashi'
    else:
        return 'oikomi'


def analyze_course(course_df, course_name, straight):
    """コースデータからQMCプロファイルを生成"""
    n_races = course_df['race_id'].nunique()
    n_horses = len(course_df)

    print(f'\n{"="*60}')
    print(f'  {course_name} (直線{straight}m)')
    print(f'  {n_races}レース, {n_horses}頭 (2023-2025)')
    print(f'{"="*60}')

    # 脚質分類
    course_df = course_df.copy()
    course_df['style'] = course_df['avg_run_style'].apply(classify_style)

    # --- 脚質別成績 ---
    print('\n  [脚質別成績]')
    style_stats = {}
    for style in ['nige', 'senkou', 'sashi', 'oikomi']:
        s = course_df[course_df['style'] == style]
        if len(s) == 0:
            style_stats[style] = {'win': 0, 'top3': 0, 'n': 0}
            continue
        win_rate = s['is_win'].mean()
        top3_rate = s['is_top3'].mean()
        style_stats[style] = {'win': win_rate, 'top3': top3_rate, 'n': len(s)}
        print(f'    {style:8s}: 勝率{win_rate:6.1%} 複勝率{top3_rate:6.1%} (n={len(s)})')

    # 全体平均複勝率
    avg_top3 = course_df['is_top3'].mean()

    # style_bonus: (脚質別複勝率 - 全体平均) を正規化
    style_bonus = {}
    for style in ['nige', 'senkou', 'sashi', 'oikomi']:
        if style_stats[style]['n'] > 10:
            diff = style_stats[style]['top3'] - avg_top3
            # スケーリング: 複勝率差10%あたり+0.05のボーナス
            bonus = round(diff * 0.5, 4)
            # クリップ
            bonus = max(-0.10, min(0.15, bonus))
        else:
            bonus = 0.0
        style_bonus[style] = bonus

    print(f'\n    → style_bonus: {style_bonus}')

    # --- 枠順バイアス ---
    print('\n  [枠順バイアス]')
    course_df['is_inner'] = course_df['wakuban'] <= 4

    # 内枠×先行
    inner_senkou = course_df[(course_df['is_inner']) & (course_df['style'].isin(['nige', 'senkou']))]
    outer_senkou = course_df[(~course_df['is_inner']) & (course_df['style'].isin(['nige', 'senkou']))]
    if len(inner_senkou) > 10 and len(outer_senkou) > 10:
        inner_s_top3 = inner_senkou['is_top3'].mean()
        outer_s_top3 = outer_senkou['is_top3'].mean()
        gate_inner_senkou = round((outer_s_top3 - inner_s_top3) * 0.3, 4)  # 負=内有利
        gate_inner_senkou = max(-0.04, min(0.02, gate_inner_senkou))
        print(f'    内枠先行 複勝率{inner_s_top3:.1%} vs 外枠先行 {outer_s_top3:.1%}')
    else:
        gate_inner_senkou = -0.01

    # 外枠×差し追込
    inner_sashi = course_df[(course_df['is_inner']) & (course_df['style'].isin(['sashi', 'oikomi']))]
    outer_sashi = course_df[(~course_df['is_inner']) & (course_df['style'].isin(['sashi', 'oikomi']))]
    if len(inner_sashi) > 10 and len(outer_sashi) > 10:
        inner_d_top3 = inner_sashi['is_top3'].mean()
        outer_d_top3 = outer_sashi['is_top3'].mean()
        gate_outer_sashi = round((inner_d_top3 - outer_d_top3) * 0.3, 4)  # 正=外不利
        gate_outer_sashi = max(-0.02, min(0.04, gate_outer_sashi))
        print(f'    内枠差し 複勝率{inner_d_top3:.1%} vs 外枠差し {outer_d_top3:.1%}')
    else:
        gate_outer_sashi = 0.01

    # 内枠×差しの包まれ
    if len(inner_sashi) > 10:
        inner_block_rate = (inner_sashi['finish'] > 5).mean()
        outer_block_rate = (outer_sashi['finish'] > 5).mean() if len(outer_sashi) > 10 else 0.5
        gate_inner_block = round((inner_block_rate - outer_block_rate) * 0.1, 4)
        gate_inner_block = max(0.0, min(0.02, gate_inner_block))
        print(f'    内枠差し掲示板外率{inner_block_rate:.1%} vs 外枠差し{outer_block_rate:.1%}')
    else:
        gate_inner_block = 0.005

    gate_bias = {
        'inner_senkou': gate_inner_senkou,
        'outer_sashi': gate_outer_sashi,
        'inner_block': gate_inner_block,
    }
    print(f'    → gate_bias: {gate_bias}')

    # --- ペース傾向 ---
    # 逃げ馬の数でペース推定
    race_nige_counts = course_df[course_df['style'] == 'nige'].groupby('race_id').size()
    avg_nige = race_nige_counts.mean() if len(race_nige_counts) > 0 else 1.5

    # pace_base_per_runner: 逃げ馬が多いほどペースが上がるコースほど高い
    # 短距離(1200)は高め、中距離(2000)は低め
    dist = course_df['kyori'].mode().values[0] if len(course_df) > 0 else 1600
    if dist <= 1200:
        pace_base = 0.30
    elif dist <= 1400:
        pace_base = 0.28
    elif dist <= 1600:
        pace_base = 0.25
    elif dist <= 2000:
        pace_base = 0.22
    else:
        pace_base = 0.20

    # pace_noise: 逃げ馬数のばらつきから推定
    pace_noise = round(race_nige_counts.std() * 0.08 + 0.10, 4) if len(race_nige_counts) > 5 else 0.15
    pace_noise = max(0.10, min(0.25, pace_noise))

    print(f'\n  [ペース]')
    print(f'    平均逃げ馬数: {avg_nige:.1f}頭/レース')
    print(f'    → pace_base_per_runner: {pace_base}')
    print(f'    → pace_noise: {pace_noise}')

    # --- trouble系 ---
    # 小回り(直線短い)ほどtrouble多い
    if straight <= 300:
        trouble_rate = 0.06
    elif straight <= 360:
        trouble_rate = 0.05
    else:
        trouble_rate = 0.04

    trouble_penalty = 0.15
    noise_scale = 0.02

    # --- プロファイル構築 ---
    profile = {
        'name': course_name,
        'straight': straight,
        'pace_base_per_runner': pace_base,
        'pace_noise': pace_noise,
        'style_bonus': style_bonus,
        'gate_bias': gate_bias,
        'trouble_rate': trouble_rate,
        'trouble_penalty': trouble_penalty,
        'noise_scale': noise_scale,
    }

    print(f'\n  [生成プロファイル]')
    print(json.dumps(profile, indent=2, ensure_ascii=False))

    return profile


# ============================================================
# 対象コースの分析
# ============================================================
# race_idにvenue情報がないため、特徴量から推定
# kyori + is_turf + race_idのvenue部分で絞る

targets = [
    ('nakayama', 2000, '中山芝2000m'),
    ('nakayama', 1200, '中山芝1200m'),
    ('hanshin', 1400, '阪神芝1400m内回り'),
    ('hanshin', 1600, '阪神芝1600m外回り'),
]

profiles = {}

for venue_key, dist, name in targets:
    straight = STRAIGHT_MAP.get(venue_key, {}).get(dist, 350)

    # フィルタリング: 芝 + 距離 + 場名(race_idに含まれる)
    mask = (
        (df['is_turf'] == 1) &
        (df['kyori'] == dist) &
        (df['race_id'].str.contains(f'_{venue_key}_', na=False)) &
        (df['finish'] > 0)
    )

    # race_idにvenue_keyが含まれない場合、別のアプローチ
    course_df = df[mask]

    if len(course_df) < 50:
        # race_idフォーマットが違う可能性、距離のみで
        print(f'\n  WARNING: {name} - race_id match少 ({len(course_df)}行)')
        print(f'  race_id sample: {df["race_id"].head(3).tolist()}')
        # 場コードで代替マッチ
        venue_codes = {
            'nakayama': '06', 'hanshin': '09',
            'tokyo': '05', 'kyoto': '08',
        }
        code = venue_codes.get(venue_key)
        if code:
            mask2 = (
                (df['is_turf'] == 1) &
                (df['kyori'] == dist) &
                (df['race_id'].str.contains(f'_{code}_', na=False)) &
                (df['finish'] > 0)
            )
            course_df2 = df[mask2]
            if len(course_df2) > len(course_df):
                course_df = course_df2

    if len(course_df) < 50:
        print(f'\n  SKIP: {name} - データ不足 ({len(course_df)}行)')
        continue

    profile = analyze_course(course_df, name, straight)

    key = f'{venue_key}_turf_{dist}'
    profiles[key] = profile

# ============================================================
# Python コード出力
# ============================================================
print('\n\n' + '='*60)
print('qmc_courses.py に追加するコード:')
print('='*60)
for key, p in profiles.items():
    print(f"""
    '{key}': {{
        'name': '{p["name"]}',
        'straight': {p["straight"]},
        'pace_base_per_runner': {p["pace_base_per_runner"]},
        'pace_noise': {p["pace_noise"]},
        'style_bonus': {{
            'nige':    {p["style_bonus"]["nige"]:+.4f},
            'senkou':  {p["style_bonus"]["senkou"]:+.4f},
            'sashi':   {p["style_bonus"]["sashi"]:+.4f},
            'oikomi':  {p["style_bonus"]["oikomi"]:+.4f},
        }},
        'gate_bias': {{
            'inner_senkou': {p["gate_bias"]["inner_senkou"]:.5f},
            'outer_sashi':  {p["gate_bias"]["outer_sashi"]:.5f},
            'inner_block':  {p["gate_bias"]["inner_block"]:.5f},
        }},
        'trouble_rate': {p["trouble_rate"]},
        'trouble_penalty': {p["trouble_penalty"]},
        'noise_scale': {p["noise_scale"]},
    }},""")
