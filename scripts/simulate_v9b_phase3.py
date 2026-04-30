"""
Phase 3a: 学習データ着順補正アプローチの軽量検証

A: 既存 pkl で finish そのまま学習
B: bias_correction_coefficients.json で finish を補正してから学習
   effective_finish = clip(finish - corr_inner_outer - corr_front_back, 1, heads)
   bias 期間外 (2019/5以前) は補正なし (delta=0) で混在学習。

注: target (finish) のみ補正。pkl 内の EMA 系は元のまま (= 説明変数は変えない)。
    効果が出れば Phase 3b でフル pkl 再ビルドへ。

評価: Phase 2 と同じ (Top1 / Brier / ROI) + subset 分析。

実行:
  py -3.13 scripts/bias_calibration.py     # 先に係数推定
  py -3.13 scripts/simulate_v9b_phase3.py
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

FRAME_MAP = {
    "超内": -2.0, "内": -1.0, "やや内": -0.5,
    "フラット": 0.0,
    "やや外": 0.5, "外": 1.0, "超外": 2.0,
}
FB_MAP = {
    "超前": -2.0, "前残り": -1.0, "前": -1.0,
    "展開次第": 0.0,
    "差し": 1.0, "差し有利": 1.0, "超差し": 2.0,
}
STRAIGHT_MAP = {
    "内伸び": -1.0, "やや内伸び": -0.5,
    "フラット": 0.0,
    "やや外伸び": 0.5, "外伸び": 1.0,
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


def load_bias(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "結果":
                continue
            rows.append({
                "date": r["date"],
                "venue": r["venue"],
                "surface": r["surface"],
                "time_diff": r["time_diff"],
                "frame_bias_score": text_to_score(r["frame_bias"], FRAME_MAP),
                "fb_bias_score": text_to_score(r["fb_bias"], FB_MAP),
            })
    return pd.DataFrame(rows)


def derive_join(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    parts = f["race_id"].astype(str).str.split("_", expand=True)
    f["_date_str"] = parts[0].str[:4] + "-" + parts[0].str[4:6] + "-" + parts[0].str[6:8]
    f["_place_code"] = parts[1].str.zfill(2)
    f["_venue"] = f["_place_code"].map(PLACE_CODES)
    f["_surface"] = np.where(f["is_turf"] == 1, "芝", "ダート")
    return f


def join_bias(feat: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    f = derive_join(feat)
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


def bin_wakuban(w):
    if pd.isna(w):
        return "unknown"
    w = int(w)
    if w <= 2:
        return "inner"
    if w <= 5:
        return "middle"
    return "outer"


def bin_frame(s):
    if pd.isna(s):
        return "unknown"
    if s <= -1.0:
        return "inner_fav"
    if s >= 1.0:
        return "outer_fav"
    return "flat"


def bin_jyuni(j):
    if pd.isna(j):
        return "unknown"
    if j <= 3.0:
        return "front"
    if j <= 7.0:
        return "middle"
    return "back"


def bin_fb(s):
    if pd.isna(s):
        return "unknown"
    if s <= -1.0:
        return "front_fav"
    if s >= 1.0:
        return "back_fav"
    return "flat"


def apply_correction(df: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    d = df.copy()
    waku = d["wakuban"].map(bin_wakuban)
    frame = d["frame_bias_score"].map(bin_frame)
    jyuni = d["avg_jyuni_4c"].map(bin_jyuni)
    fb = d["fb_bias_score"].map(bin_fb)

    io_table = coeffs.get("wakuban_x_frame", {})
    fb_table = coeffs.get("jyuni_x_fb", {})

    def lookup(table, key):
        v = table.get(key)
        return v["correction"] if v else 0.0

    delta_io = pd.Series(
        [lookup(io_table, f"{w}|{fr}") for w, fr in zip(waku, frame)],
        index=d.index,
    )
    delta_fb = pd.Series(
        [lookup(fb_table, f"{j}|{f}") for j, f in zip(jyuni, fb)],
        index=d.index,
    )
    # bias 期間外は _bias_matched=False → delta=0 にする (lookup でも 0 になるが念のため明示)
    delta_io = delta_io.where(d["_bias_matched"], 0.0)
    delta_fb = delta_fb.where(d["_bias_matched"], 0.0)

    eff = d["finish"] - delta_io - delta_fb
    heads = d.get("heads", pd.Series(18, index=d.index)).fillna(18)
    eff = eff.clip(lower=1.0, upper=heads)
    d["effective_finish"] = eff
    d["_delta_io"] = delta_io
    d["_delta_fb"] = delta_fb
    return d


# ============================================================
# Evaluation (Phase 2 と同等)
# ============================================================

def evaluate_top5_top3(df):
    df = df.copy()
    df["_rank"] = df.groupby("race_id")["mu"].rank(ascending=True, method="first")
    top5 = df[df["_rank"] <= 5]
    return top5.groupby("race_id")["finish"].apply(lambda x: (x <= 3).any()).mean()


def evaluate_top1(df):
    idx = df.groupby("race_id")["mu"].idxmin()
    top1 = df.loc[idx]
    return (top1["finish"] == 1).mean()


def evaluate_brier(df, temperature=0.1):
    df = df.copy()
    df["_y"] = (df["finish"] == 1).astype(float)
    def _smx(g):
        x = -g["mu"].values / temperature
        x = x - x.max()
        e = np.exp(x)
        return pd.Series(e / e.sum(), index=g.index)
    df["_p"] = df.groupby("race_id", group_keys=False).apply(_smx)
    return ((df["_p"] - df["_y"]) ** 2).mean()


def evaluate_roi(df, odds_col="odds"):
    if odds_col not in df.columns:
        return None
    idx = df.groupby("race_id")["mu"].idxmin()
    top1 = df.loc[idx].copy()
    top1["_payout"] = np.where(top1["finish"] == 1, top1[odds_col].astype(float) * 100, 0.0)
    n = len(top1)
    if n == 0:
        return None
    return (top1["_payout"].sum() - n * 100.0) / (n * 100.0)


def report_block(name, df_a, df_b):
    rows = []
    for label, df in [("A (raw target)", df_a), ("B (corrected target)", df_b)]:
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
        num_cols = ["top5_top3", "top1_hit", "brier", "roi"]
        d = out[num_cols].iloc[1] - out[num_cols].iloc[0]
        roi_str = f"{d['roi']:+.4f}" if pd.notna(d["roi"]) else "—"
        print(f"  Delta: top5_top3 {d['top5_top3']:+.4f} | "
              f"top1 {d['top1_hit']:+.4f} | "
              f"brier {d['brier']:+.4f} | "
              f"roi {roi_str}")


def main():
    feat_path = ROOT / "data" / "features_v9b_2026.pkl"
    bias_path = ROOT / "data" / "track_bias_parsed.jsonl"
    coeff_path = ROOT / "data" / "bias_correction_coefficients.json"

    for p in (feat_path, bias_path, coeff_path):
        if not p.exists():
            print(f"[err] {p} not found")
            if p == coeff_path:
                print("  → 先に scripts/bias_calibration.py を実行")
            return 1

    print("[1/7] Loading...")
    feat = pd.read_pickle(feat_path)
    bias = load_bias(bias_path)
    coeffs = json.loads(coeff_path.read_text(encoding="utf-8"))
    print(f"  feat: {len(feat)} rows, bias: {len(bias)} rows")
    print(f"  coeffs: {len(coeffs.get('wakuban_x_frame', {}))} io cells, "
          f"{len(coeffs.get('jyuni_x_fb', {}))} fb cells")

    print("[2/7] Joining...")
    merged = join_bias(feat, bias)
    n_match = merged["_bias_matched"].sum()
    print(f"  matched {n_match} / {len(merged)} ({100*n_match/len(merged):.1f}%)")

    print("[3/7] Train/test split (turf only)...")
    merged = merged[merged["_surface"] == "芝"].copy()
    train_df = filter_period(merged, "2019-05-01", "2024-12-31")
    test_df = filter_period(merged, "2025-01-01", "2026-12-31")
    print(f"  train: {len(train_df)}  test: {len(test_df)}")

    available = [c for c in FEATURES_V9 if c in train_df.columns]
    missing = sorted(set(FEATURES_V9) - set(available))
    if missing:
        print(f"  [warn] FEATURES_V9 missing: {missing}")
    print(f"  using {len(available)} / {len(FEATURES_V9)} numeric features")

    print("[4/7] Applying correction to train target...")
    train_corr = apply_correction(train_df, coeffs)
    print(f"  matched train rows: {train_corr['_bias_matched'].sum()} / {len(train_corr)}")
    print(f"  delta_io: mean={train_corr['_delta_io'].mean():+.3f} "
          f"std={train_corr['_delta_io'].std():.3f}")
    print(f"  delta_fb: mean={train_corr['_delta_fb'].mean():+.3f} "
          f"std={train_corr['_delta_fb'].std():.3f}")
    print(f"  finish vs effective_finish: corr={train_corr[['finish','effective_finish']].corr().iloc[0,1]:.4f}")

    print("[5/7] Training A (raw target)...")
    pred_a = Predictor(numeric_features=available, cat_features=CAT_FEATURES)
    pred_a.train(train_df, ep=50, lr=0.003, seed=42)
    preds_a = pred_a.predict(test_df)

    print("[6/7] Training B (corrected target)...")
    train_b = train_corr.copy()
    # Predictor.train は 'finish' 列をターゲットに使う
    train_b["finish"] = train_b["effective_finish"]
    pred_b = Predictor(numeric_features=available, cat_features=CAT_FEATURES)
    pred_b.train(train_b, ep=50, lr=0.003, seed=42)
    preds_b = pred_b.predict(test_df)

    print("[7/7] Evaluating...")
    df_a = preds_a.copy()
    df_a["race_id"] = test_df["race_id"].values
    df_a["finish"] = test_df["finish"].values
    df_a["_bias_matched"] = test_df["_bias_matched"].values
    df_a["time_diff"] = test_df["time_diff"].values
    df_a["frame_bias_score"] = test_df["frame_bias_score"].values
    df_a["fb_bias_score"] = test_df["fb_bias_score"].values
    if "odds" in test_df.columns:
        df_a["odds"] = test_df["odds"].values

    df_b = preds_b.copy()
    df_b["race_id"] = test_df["race_id"].values
    df_b["finish"] = test_df["finish"].values
    df_b["_bias_matched"] = test_df["_bias_matched"].values
    df_b["time_diff"] = test_df["time_diff"].values
    df_b["frame_bias_score"] = test_df["frame_bias_score"].values
    df_b["fb_bias_score"] = test_df["fb_bias_score"].values
    if "odds" in test_df.columns:
        df_b["odds"] = test_df["odds"].values

    print("\n" + "=" * 60)
    print("OVERALL (test set, turf, 2025-01〜2026-12)")
    print("=" * 60)
    report_block("OVERALL", df_a, df_b)

    print("\n" + "=" * 60)
    print("BIAS-MATCHED ONLY (test 中で bias data 取得日 のみ)")
    print("=" * 60)
    df_a_m = df_a[df_a["_bias_matched"]]
    df_b_m = df_b[df_b["_bias_matched"]]
    report_block("BIAS-MATCHED", df_a_m, df_b_m)

    def split_active(d):
        race_keys = (
            (d["time_diff"].abs() > 1.5)
            | (d["frame_bias_score"].fillna(0) != 0)
            | (d["fb_bias_score"].fillna(0) != 0)
        )
        active_ids = d.loc[race_keys, "race_id"].unique()
        return d[d["race_id"].isin(active_ids)].copy(), d[~d["race_id"].isin(active_ids)].copy()

    a_active, a_flat = split_active(df_a_m)
    b_active, b_flat = split_active(df_b_m)
    print("\n" + "=" * 60)
    print("BIAS-ACTIVE ONLY")
    print("=" * 60)
    report_block("BIAS-ACTIVE", a_active, b_active)
    print(f"  (active races: {a_active['race_id'].nunique()})")

    print("\n" + "=" * 60)
    print("BIAS-FLAT ONLY")
    print("=" * 60)
    report_block("BIAS-FLAT", a_flat, b_flat)
    print(f"  (flat races: {a_flat['race_id'].nunique()})")

    print("\n" + "=" * 60)
    print("採用判定 (Phase 3 issue):")
    print("  - top1_hit:   B - A ≥ +0.01 で採用候補")
    print("  - roi:        B - A ≥ +0.05 で採用候補")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
