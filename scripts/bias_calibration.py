"""
Phase 3 Step 1: 馬場バイアス補正係数の推定

設計:
  各セルで:
    residual_i = finish_i - horse_avg_finish_i
    correction(cell) = mean(residual_i)
  「同馬の平均着順と比べてこのセル条件下で何着ずれたか」が補正量。

セル定義:
  (a) wakuban_band × frame_bias_band  3×3=9 (内外補正)
  (b) avg_jyuni_4c_band × fb_bias_band 3×3=9 (前後補正、当該レース jyuni_4c
      は pkl に無いため過去平均 avg_jyuni_4c をプロキシとして使用)

入力:
  data/features_v9b_2026.pkl
  data/track_bias_parsed.jsonl

出力:
  data/bias_correction_coefficients.json

実行:
  py -3.13 scripts/bias_calibration.py
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

from src.binary_parser import PLACE_CODES

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
    f = feat.copy()
    parts = f["race_id"].astype(str).str.split("_", expand=True)
    f["_date_str"] = parts[0].str[:4] + "-" + parts[0].str[4:6] + "-" + parts[0].str[6:8]
    f["_place_code"] = parts[1].str.zfill(2)
    f["_venue"] = f["_place_code"].map(PLACE_CODES)
    f["_surface"] = np.where(f["is_turf"] == 1, "芝", "ダート")
    return f


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


def main():
    feat_path = ROOT / "data" / "features_v9b_2026.pkl"
    bias_path = ROOT / "data" / "track_bias_parsed.jsonl"
    out_path = ROOT / "data" / "bias_correction_coefficients.json"

    if not feat_path.exists():
        print(f"[err] {feat_path} not found")
        return 1
    if not bias_path.exists():
        print(f"[err] {bias_path} not found")
        return 1

    print("[1/5] Loading...")
    feat = pd.read_pickle(feat_path)
    bias = load_bias(bias_path, kind="結果")
    print(f"  feat: {len(feat)} rows, bias: {len(bias)} rows")

    print("[2/5] Joining (date×venue×surface)...")
    feat = derive_join_keys(feat)
    merged = feat.merge(
        bias,
        left_on=["_date_str", "_venue", "_surface"],
        right_on=["date", "venue", "surface"],
        how="left",
        suffixes=("", "_bias"),
    )
    merged["_matched"] = merged["time_diff"].notna()
    n_match = merged["_matched"].sum()
    print(f"  matched {n_match} / {len(merged)} ({100*n_match/len(merged):.1f}%)")

    # 芝のみ + matched のみで係数推定
    cal = merged[(merged["_surface"] == "芝") & merged["_matched"]].copy()
    print(f"  calibration sample (turf, matched): {len(cal)} rows")

    print("[3/5] Computing horse_avg_finish per ketto_num (1-pass)...")
    horse_avg = cal.groupby("ketto_num")["finish"].mean().rename("horse_avg_finish")
    cal = cal.merge(horse_avg, on="ketto_num", how="left")
    cal["_residual"] = cal["finish"] - cal["horse_avg_finish"]
    n_horse = cal["ketto_num"].nunique()
    print(f"  unique horses: {n_horse}, mean residual: {cal['_residual'].mean():.4f} (should ~0)")

    print("[4/5] Estimating cell corrections...")
    cal["_waku_band"] = cal["wakuban"].map(bin_wakuban)
    cal["_frame_band"] = cal["frame_bias_score"].map(bin_frame)
    cal["_jyuni_band"] = cal["avg_jyuni_4c"].map(bin_jyuni)
    cal["_fb_band"] = cal["fb_bias_score"].map(bin_fb)

    inner_outer = (
        cal[cal["_waku_band"] != "unknown"]
        .groupby(["_waku_band", "_frame_band"])["_residual"]
        .agg(["mean", "count"])
        .reset_index()
    )
    front_back = (
        cal[cal["_jyuni_band"] != "unknown"]
        .groupby(["_jyuni_band", "_fb_band"])["_residual"]
        .agg(["mean", "count"])
        .reset_index()
    )

    print("\n[inner/outer × frame_bias] correction (residual mean):")
    pivot_io = inner_outer.pivot(index="_waku_band", columns="_frame_band", values="mean")
    print(pivot_io.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n  cell counts:")
    pivot_io_n = inner_outer.pivot(index="_waku_band", columns="_frame_band", values="count")
    print(pivot_io_n.to_string())

    print("\n[front/back × fb_bias] correction (residual mean):")
    pivot_fb = front_back.pivot(index="_jyuni_band", columns="_fb_band", values="mean")
    print(pivot_fb.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n  cell counts:")
    pivot_fb_n = front_back.pivot(index="_jyuni_band", columns="_fb_band", values="count")
    print(pivot_fb_n.to_string())

    print("\n[5/5] Saving JSON...")
    coeffs = {
        "meta": {
            "source": str(feat_path.name),
            "bias": str(bias_path.name),
            "n_calibration_rows": int(len(cal)),
            "n_unique_horses": int(n_horse),
            "note": "avg_jyuni_4c は当該レース jyuni_4c のプロキシ (pkl に当該レース値が無いため)",
        },
        "wakuban_x_frame": {
            f"{r['_waku_band']}|{r['_frame_band']}": {
                "correction": float(r["mean"]),
                "n": int(r["count"]),
            }
            for _, r in inner_outer.iterrows()
        },
        "jyuni_x_fb": {
            f"{r['_jyuni_band']}|{r['_fb_band']}": {
                "correction": float(r["mean"]),
                "n": int(r["count"]),
            }
            for _, r in front_back.iterrows()
        },
    }
    out_path.write_text(json.dumps(coeffs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
