"""
v9b 馬場バイアス Phase 2: 評価軸変更 + Subset 分析

Phase 1 (simulate_v9b_with_bias.py) の null 結果を受けて以下を追加検証:

評価軸 (Phase 1 の Top-5×Top-3 は baseline 95% で天井近く検出力不足):
  1. Top-1 ヒット率 (= 単勝当たり率)
  2. 単勝 Brier score (連続値、天井なし)
  3. オッズ加重 ROI (実用評価)

Subset 分析 (bias は (date×venue×surface) 単位 → 同一日同場の全馬同値で
fixed effect 化しやすい。バイアスが立っているレースに限定して効果を検出):
  - bias_active:  |time_diff| > 1.5  OR  frame_bias_score != 0  OR  fb_bias_score != 0
  - bias_flat:    上記の補集合 (and bias_matched)

家PC での Phase 1 修正を取り込み済:
  - place_code を race_id (YYYYMMDD_PP_KK_NN_RR) から派生
  - surface は is_turf から派生
  - finish 列は既存なので rename 不要
  - FEATURES_V9 は pkl 内に存在する列だけにフィルタ

実行:
  python scripts/simulate_v9b_phase2.py
"""
import io
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import numpy as np
import pandas as pd

from src.predictor import CAT_FEATURES, FEATURES_V9, Predictor
from src.binary_parser import PLACE_CODES

BIAS_FEATURES = [
    "time_diff",
    "frame_bias_score",
    "straight_bias_score",
    "fb_bias_score",
]

FRAME_MAP = {
    "超内": -2.0, "内": -1.0, "やや内": -0.5,
    "フラット": 0.0,
    "やや外": 0.5, "外": 1.0, "超外": 2.0,
}
STRAIGHT_MAP = {
    "内伸び": -1.0, "やや内伸び": -0.5,
    "フラット": 0.0,
    "やや外伸び": 0.5, "外伸び": 1.0,
}
FB_MAP = {
    "超前": -2.0, "前残り": -1.0, "前": -1.0,
    "展開次第": 0.0,
    "差し": 1.0, "差し有利": 1.0, "超差し": 2.0,
}


def text_to_score(text, mapping):
    if not text:
        return None
    found = []
    keywords = sorted(mapping.keys(), key=len, reverse=True)
    masked = text
    for kw in keywords:
        if kw in masked:
            found.append(mapping[kw])
            masked = masked.replace(kw, "#" * len(kw))
    if not found:
        return None
    return float(np.mean(found))


def load_bias(path: Path, kind: str = "結果") -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if kind and r.get("kind") != kind:
                continue
            rows.append({
                "date": r["date"],
                "venue": r["venue"],
                "surface": r["surface"],
                "time_diff": r["time_diff"],
                "frame_bias_score": text_to_score(r["frame_bias"], FRAME_MAP),
                "straight_bias_score": text_to_score(r["straight_bias"], STRAIGHT_MAP),
                "fb_bias_score": text_to_score(r["fb_bias"], FB_MAP),
            })
    return pd.DataFrame(rows)


def derive_join_keys(feat: pd.DataFrame) -> pd.DataFrame:
    """
    家PC Phase 1 で発見済の派生ロジック:
      - race_id 形式: YYYYMMDD_PP_KK_NN_RR
      - place_code は race_id 第2要素
      - surface は is_turf (1=芝, 0=ダート) から
    """
    f = feat.copy()
    if "race_id" in f.columns:
        parts = f["race_id"].astype(str).str.split("_", expand=True)
        f["_date_str"] = parts[0].str[:4] + "-" + parts[0].str[4:6] + "-" + parts[0].str[6:8]
        f["_place_code"] = parts[1].str.zfill(2)
    else:
        f["_date_str"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
        f["_place_code"] = f.get("place_code", "").astype(str).str.zfill(2)
    f["_venue"] = f["_place_code"].map(PLACE_CODES)
    if "surface" in f.columns and f["surface"].notna().any():
        f["_surface"] = f["surface"]
    elif "is_turf" in f.columns:
        f["_surface"] = f["is_turf"].map({1: "芝", 0: "ダート"})
    else:
        f["_surface"] = None
    return f


def join_bias(feat: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    f = derive_join_keys(feat)
    merged = f.merge(
        bias,
        left_on=["_date_str", "_venue", "_surface"],
        right_on=["date", "venue", "surface"],
        how="left",
        suffixes=("", "_bias"),
    )
    merged["_bias_matched"] = merged["time_diff"].notna()
    return merged


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    d = pd.to_datetime(df["_date_str"])
    return df[(d >= s) & (d <= e)].copy()


# ============================================================
# 評価関数
# ============================================================

def evaluate_top5_top3(df):
    """既存の馬券内カバー率 (top-5 推奨に top-3 着内が含まれる率)"""
    df = df.copy()
    df["_rank"] = df.groupby("race_id")["mu"].rank(ascending=True, method="first")
    top5 = df[df["_rank"] <= 5]
    return top5.groupby("race_id")["finish"].apply(lambda x: (x <= 3).any()).mean()


def evaluate_top1(df):
    """各レースで mu 最小 (=最強推奨) の馬が実際 1着だった率"""
    idx = df.groupby("race_id")["mu"].idxmin()
    top1 = df.loc[idx]
    return (top1["finish"] == 1).mean()


def evaluate_brier(df, temperature=0.1):
    """
    単勝確率を softmax(-mu/T) で計算し、1着フラグとの Brier score を返す。
    低いほど良い。
    """
    df = df.copy()
    df["_y"] = (df["finish"] == 1).astype(float)
    def _softmax_neg_mu(g):
        x = -g["mu"].values / temperature
        x = x - x.max()
        e = np.exp(x)
        return pd.Series(e / e.sum(), index=g.index)
    df["_p"] = df.groupby("race_id", group_keys=False).apply(_softmax_neg_mu)
    return ((df["_p"] - df["_y"]) ** 2).mean()


def evaluate_roi(df, odds_col="odds"):
    """各レース mu 最小の馬に 100yen 単勝。ROI = (払戻合計 - 賭金合計) / 賭金合計"""
    if odds_col not in df.columns:
        return None
    idx = df.groupby("race_id")["mu"].idxmin()
    top1 = df.loc[idx].copy()
    top1["_payout"] = np.where(top1["finish"] == 1, top1[odds_col].astype(float) * 100, 0.0)
    n = len(top1)
    if n == 0:
        return None
    total_payout = top1["_payout"].sum()
    total_stake = n * 100.0
    return (total_payout - total_stake) / total_stake


def report_block(name, df_a, df_b):
    """A と B 両方で全評価を出力。"""
    rows = []
    for label, df in [("A (baseline)", df_a), ("B (with bias)", df_b)]:
        rows.append({
            "model": label,
            "n_races": df["race_id"].nunique(),
            "top5_top3": evaluate_top5_top3(df),
            "top1_hit": evaluate_top1(df),
            "brier": evaluate_brier(df),
            "roi": evaluate_roi(df),
        })
    out = pd.DataFrame(rows)
    print(f"\n[{name}]")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))
    if len(out) == 2:
        d = out.iloc[1] - out.iloc[0]
        print(f"  Delta: top5_top3 {d['top5_top3']:+.4f} | "
              f"top1 {d['top1_hit']:+.4f} | "
              f"brier {d['brier']:+.4f} | "
              f"roi {d['roi']:+.4f}" if pd.notna(d["roi"]) else "")


def main():
    feat_path = ROOT / "data" / "features_v9b_2026.pkl"
    bias_path = ROOT / "data" / "track_bias_parsed.jsonl"

    if not feat_path.exists():
        print(f"[err] {feat_path} not found")
        return 1
    if not bias_path.exists():
        print(f"[err] {bias_path} not found")
        return 1

    print("[1/6] Loading...")
    feat = pd.read_pickle(feat_path)
    bias = load_bias(bias_path, kind="結果")
    print(f"  feat: {len(feat)} rows, bias: {len(bias)} rows")

    print("[2/6] Joining...")
    merged = join_bias(feat, bias)
    n_match = merged["_bias_matched"].sum()
    print(f"  matched {n_match} / {len(merged)} ({100*n_match/len(merged):.1f}%)")

    print("[3/6] Train/test split (turf only)...")
    merged = merged[merged["_surface"] == "芝"].copy()
    train_df = filter_period(merged, "2019-05-01", "2024-12-31")
    test_df = filter_period(merged, "2025-01-01", "2026-12-31")
    print(f"  train: {len(train_df)}  test: {len(test_df)}")

    available = [c for c in FEATURES_V9 if c in train_df.columns]
    missing = set(FEATURES_V9) - set(available)
    if missing:
        print(f"  [warn] FEATURES_V9 missing in pkl: {sorted(missing)}")
    print(f"  using {len(available)} / {len(FEATURES_V9)} numeric features")

    print("[4/6] Training A (baseline)...")
    pred_a = Predictor(numeric_features=available, cat_features=CAT_FEATURES)
    pred_a.train(train_df, ep=50, lr=0.003, seed=42)
    preds_a = pred_a.predict(test_df)

    print("[5/6] Training B (with bias)...")
    train_b = train_df.copy()
    test_b = test_df.copy()
    for c in BIAS_FEATURES:
        med = train_b[c].median()
        train_b[c] = train_b[c].fillna(med)
        test_b[c] = test_b[c].fillna(med)
    pred_b = Predictor(numeric_features=available + BIAS_FEATURES, cat_features=CAT_FEATURES)
    pred_b.train(train_b, ep=50, lr=0.003, seed=42)
    preds_b = pred_b.predict(test_b)

    print("[6/6] Evaluating...")
    # Attach metadata back to predictions
    df_a = preds_a.copy()
    df_a["race_id"] = test_df["race_id"].values
    df_a["finish"] = test_df["finish"].values
    df_a["_bias_matched"] = test_df["_bias_matched"].values
    df_a["time_diff"] = test_df["time_diff"].values
    df_a["frame_bias_score"] = test_df["frame_bias_score"].values
    df_a["fb_bias_score"] = test_df["fb_bias_score"].values
    if "odds" not in df_a.columns and "odds" in test_df.columns:
        df_a["odds"] = test_df["odds"].values

    df_b = preds_b.copy()
    df_b["race_id"] = test_b["race_id"].values
    df_b["finish"] = test_b["finish"].values
    df_b["_bias_matched"] = test_b["_bias_matched"].values
    df_b["time_diff"] = test_b["time_diff"].values
    df_b["frame_bias_score"] = test_b["frame_bias_score"].values
    df_b["fb_bias_score"] = test_b["fb_bias_score"].values
    if "odds" not in df_b.columns and "odds" in test_b.columns:
        df_b["odds"] = test_b["odds"].values

    # Subset masks (race-level)
    def bias_active_races(d):
        race_keys = (
            (d["time_diff"].abs() > 1.5)
            | (d["frame_bias_score"].fillna(0) != 0)
            | (d["fb_bias_score"].fillna(0) != 0)
        )
        active_ids = d.loc[race_keys, "race_id"].unique()
        return d[d["race_id"].isin(active_ids)].copy(), d[~d["race_id"].isin(active_ids)].copy()

    print("\n" + "=" * 60)
    print("OVERALL (test set, turf, 2025-01〜2026-12)")
    print("=" * 60)
    report_block("OVERALL", df_a, df_b)

    print("\n" + "=" * 60)
    print("BIAS-MATCHED ONLY (bias data 取得日 のみ)")
    print("=" * 60)
    df_a_m = df_a[df_a["_bias_matched"]]
    df_b_m = df_b[df_b["_bias_matched"]]
    report_block("BIAS-MATCHED", df_a_m, df_b_m)

    print("\n" + "=" * 60)
    print("BIAS-ACTIVE ONLY (|time_diff|>1.5 or non-flat 枠/前後)")
    print("=" * 60)
    a_active, a_flat = bias_active_races(df_a_m)
    b_active, b_flat = bias_active_races(df_b_m)
    report_block("BIAS-ACTIVE", a_active, b_active)
    print(f"  (bias-active races: {a_active['race_id'].nunique()})")

    print("\n" + "=" * 60)
    print("BIAS-FLAT ONLY (バイアスがほぼ立っていないレース)")
    print("=" * 60)
    report_block("BIAS-FLAT", a_flat, b_flat)
    print(f"  (bias-flat races: {a_flat['race_id'].nunique()})")

    print("\n" + "=" * 60)
    print("採用判定: いずれかの指標で B が A を有意に上回るか確認")
    print("  - top1_hit:   +0.01 (1pt) 以上で採用候補")
    print("  - brier:      -0.001 以下 (より小さい) で採用候補")
    print("  - roi:        +0.05 (5pt) 以上で採用候補")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
