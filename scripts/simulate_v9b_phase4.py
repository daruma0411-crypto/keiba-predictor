"""
Phase 4-α: Layer 2 QMC への馬場バイアス統合（兆候判定）

A: 既存 qmc_sim (bias なし)
B: bias データで profile を動的修正してから qmc_sim

bias × QMC パラメータ対応 (issue #4):
  fb_bias_score      → style_bonus.{nige,senkou,sashi,oikomi}
  frame_bias_score   → gate_bias.{inner_senkou,outer_sashi}
  straight_bias_score→ gate_bias.outer_sashi 微調整
  time_diff          → pace_noise, noise_scale (高速馬場で変動↓)

評価: Top1 ヒット率 / ROI （Phase 2 と同じ、Brier は除外）
Subset: OVERALL / BIAS-MATCHED / BIAS-ACTIVE

実行:
  py -3.13 scripts/simulate_v9b_phase4.py [--strength 0.05] [--n 10000]
"""
import argparse
import copy
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
from src.qmc_courses import COURSE_PROFILES, qmc_sim

# ============================================================
# bias text → score 変換
# ============================================================
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
                "straight_bias_score": text_to_score(r["straight_bias"], STRAIGHT_MAP),
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


# ============================================================
# 場名 (kanji) → venue_key
# ============================================================
VENUE_KEY = {
    "札幌": "sapporo", "函館": "hakodate", "福島": "fukushima",
    "新潟": "niigata", "東京": "tokyo", "中山": "nakayama",
    "中京": "chukyo", "京都": "kyoto", "阪神": "hanshin", "小倉": "kokura",
}

# (venue_key, distance) → COURSE_PROFILES key
COURSE_KEY_MAP = {
    ("nakayama", 1200): "nakayama_turf_1200",
    ("nakayama", 1600): "nakayama_turf_1600",
    ("nakayama", 2000): "nakayama_turf_2000",
    ("fukushima", 1200): "fukushima_turf_1200",
    ("fukushima", 2000): "fukushima_turf_2000",
    ("fukushima", 2600): "fukushima_turf_2600",
    ("hanshin", 1400): "hanshin_turf_1400",
    ("hanshin", 1600): "hanshin_turf_1600_outer",
    ("hanshin", 2000): "hanshin_turf_2000",
    ("chukyo", 1400): "chukyo_turf_1400",
}


def detect_course_key(venue: str, distance: int, is_turf: bool) -> str | None:
    if not is_turf:
        return None
    vk = VENUE_KEY.get(venue)
    if not vk:
        return None
    return COURSE_KEY_MAP.get((vk, int(distance)))


# ============================================================
# bias → QMC profile への変換
# ============================================================
def apply_bias_to_profile(base: dict, bias: dict, strength: float = 0.05) -> dict:
    """
    bias を QMC profile に注入する。
      fb_bias_score: 前(-)で逃げ先行↑/差し追込↓
      frame_bias_score: 内有利(-)で内枠先行↑/外枠追込↓
      straight_bias_score: 外伸び(+)で外枠差し届く
      time_diff: 高速(+)→ノイズ縮小（バイアス強化）
    """
    if bias is None:
        return base

    def _safe(v):
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if np.isnan(f) else f

    fb = _safe(bias.get("fb_bias_score"))
    fr = _safe(bias.get("frame_bias_score"))
    st = _safe(bias.get("straight_bias_score"))
    td = _safe(bias.get("time_diff"))

    p = copy.deepcopy(base)

    # 前後 → style_bonus
    p["style_bonus"]["nige"]   += -fb * strength * 1.0
    p["style_bonus"]["senkou"] += -fb * strength * 0.6
    p["style_bonus"]["sashi"]  += +fb * strength * 0.6
    p["style_bonus"]["oikomi"] += +fb * strength * 1.0

    # 内/外 → gate_bias
    p["gate_bias"]["inner_senkou"] += -fr * strength
    p["gate_bias"]["outer_sashi"]  += -fr * strength
    # 外伸び → outer_sashi 微調整 (st が正→外差し届く→不利度↓)
    p["gate_bias"]["outer_sashi"]  += -st * strength * 0.5

    # 時計差 → ノイズ縮小 (高速馬場で変動↓)
    factor = max(0.6, 1.0 - td * strength * 0.05)
    p["pace_noise"]  *= factor
    p["noise_scale"] *= factor

    return p


def register_temp_profile(base_key: str, bias: dict, strength: float) -> str:
    """COURSE_PROFILES に一時的なエントリを登録し、キーを返す。"""
    base = COURSE_PROFILES[base_key]
    new_prof = apply_bias_to_profile(base, bias, strength=strength)
    temp_key = f"__phase4_tmp__{base_key}"
    COURSE_PROFILES[temp_key] = new_prof
    return temp_key


# ============================================================
# 評価
# ============================================================
def evaluate_top1(df):
    return (df["pick_finish"] == 1).mean()


def evaluate_roi(df):
    payout = np.where(df["pick_finish"] == 1, df["pick_odds"].astype(float) * 100, 0.0)
    n = len(df)
    return (payout.sum() - n * 100.0) / (n * 100.0)


def report(label: str, summary_df: pd.DataFrame):
    print(f"\n[{label}]  n_races={len(summary_df)}")
    a = summary_df[["a_pick_finish", "a_pick_odds"]].rename(columns={"a_pick_finish": "pick_finish", "a_pick_odds": "pick_odds"})
    b = summary_df[["b_pick_finish", "b_pick_odds"]].rename(columns={"b_pick_finish": "pick_finish", "b_pick_odds": "pick_odds"})
    top1_a = evaluate_top1(a)
    top1_b = evaluate_top1(b)
    roi_a = evaluate_roi(a)
    roi_b = evaluate_roi(b)
    same_pick = (summary_df["a_pick_umaban"] == summary_df["b_pick_umaban"]).mean()
    print(f"  A: top1={top1_a:.4f} roi={roi_a:+.4f}")
    print(f"  B: top1={top1_b:.4f} roi={roi_b:+.4f}")
    print(f"  Δ: top1={top1_b-top1_a:+.4f} ({(top1_b-top1_a)*100:+.2f}pt)  "
          f"roi={roi_b-roi_a:+.4f} ({(roi_b-roi_a)*100:+.2f}pt)")
    print(f"  same_pick (A と B が同じ馬を選ぶ率): {same_pick:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", type=float, default=0.05, help="bias 注入強度 (default 0.05)")
    ap.add_argument("--n", type=int, default=10000, help="QMC samples per race (default 10000)")
    args = ap.parse_args()

    feat_path = ROOT / "data" / "features_v9b_2026.pkl"
    bias_path = ROOT / "data" / "track_bias_parsed.jsonl"
    for p in (feat_path, bias_path):
        if not p.exists():
            print(f"[err] {p} not found")
            return 1

    print(f"[settings] strength={args.strength}  qmc_n={args.n}")
    print("[1/6] Loading...")
    feat = pd.read_pickle(feat_path)
    bias = load_bias(bias_path)
    print(f"  feat: {len(feat)} rows, bias: {len(bias)} rows")

    print("[2/6] Joining + train/test split...")
    merged = join_bias(feat, bias)
    merged = merged[merged["_surface"] == "芝"].copy()
    train_df = filter_period(merged, "2019-05-01", "2024-12-31")
    test_df = filter_period(merged, "2025-01-01", "2026-12-31")
    print(f"  train: {len(train_df)}  test: {len(test_df)}")

    available = [c for c in FEATURES_V9 if c in train_df.columns]
    missing = sorted(set(FEATURES_V9) - set(available))
    if missing:
        print(f"  [warn] FEATURES_V9 missing: {missing}")

    print("[3/6] Training Predictor (single shared model)...")
    pred = Predictor(numeric_features=available, cat_features=CAT_FEATURES)
    pred.train(train_df, ep=50, lr=0.003, seed=42)
    preds_full = pred.predict(test_df)
    preds_full["race_id"] = test_df["race_id"].values
    preds_full["finish"] = test_df["finish"].values
    preds_full["wakuban"] = test_df["wakuban"].values
    preds_full["avg_run_style"] = test_df["avg_run_style"].values
    preds_full["_bias_matched"] = test_df["_bias_matched"].values

    # bias columns に test_df の値をマッピング
    for col in ["time_diff", "frame_bias_score", "fb_bias_score", "straight_bias_score"]:
        if col in test_df.columns:
            preds_full[col] = test_df[col].values
        else:
            preds_full[col] = np.nan

    # course key を test_df から導出
    test_meta = test_df[["race_id", "_venue", "kyori", "is_turf"]].drop_duplicates(subset=["race_id"]).copy()
    test_meta["course_key"] = test_meta.apply(
        lambda r: detect_course_key(r["_venue"], r["kyori"], bool(r["is_turf"])),
        axis=1,
    )
    n_total = len(test_meta)
    n_with_profile = test_meta["course_key"].notna().sum()
    print(f"  test races: {n_total}, course-profile defined: {n_with_profile} ({100*n_with_profile/n_total:.1f}%)")
    test_meta_use = test_meta[test_meta["course_key"].notna()].copy()

    print("[4/6] Running QMC per race (A: bias=None / B: bias=actual)...")
    rows = []
    skipped = 0
    for idx, meta in enumerate(test_meta_use.itertuples(index=False)):
        rid = meta.race_id
        course_key = meta.course_key
        race_preds = preds_full[preds_full["race_id"] == rid].copy()
        if len(race_preds) < 4:
            skipped += 1
            continue

        rf = race_preds[["wakuban", "avg_run_style"]].reset_index(drop=True)
        ps = race_preds[["mu", "sigma", "horse_name", "umaban", "odds"]].reset_index(drop=True)

        # A: no bias
        try:
            mc_a = qmc_sim(ps, race_features=rf, course=course_key, n=args.n)
        except Exception as e:
            skipped += 1
            continue

        # B: with bias (matched 行のみ。未マッチなら A と同じ profile)
        bias_dict = None
        if race_preds["_bias_matched"].any():
            r0 = race_preds.iloc[0]
            bias_dict = {
                "time_diff": r0["time_diff"],
                "frame_bias_score": r0["frame_bias_score"],
                "fb_bias_score": r0["fb_bias_score"],
                "straight_bias_score": r0["straight_bias_score"],
            }

        if bias_dict is not None:
            tmp_key = register_temp_profile(course_key, bias_dict, args.strength)
            try:
                mc_b = qmc_sim(ps, race_features=rf, course=tmp_key, n=args.n)
            finally:
                COURSE_PROFILES.pop(tmp_key, None)
        else:
            mc_b = mc_a

        # それぞれ最良 (expected_rank 最小) を pick
        a_pick = mc_a.iloc[0]
        b_pick = mc_b.iloc[0]
        a_finish = race_preds.loc[race_preds["umaban"] == a_pick["umaban"], "finish"]
        b_finish = race_preds.loc[race_preds["umaban"] == b_pick["umaban"], "finish"]
        if len(a_finish) == 0 or len(b_finish) == 0:
            skipped += 1
            continue

        rows.append({
            "race_id": rid,
            "course_key": course_key,
            "_bias_matched": bool(bias_dict is not None),
            "a_pick_umaban": int(a_pick["umaban"]),
            "a_pick_odds": float(a_pick["odds"]) if pd.notna(a_pick["odds"]) else 0.0,
            "a_pick_finish": int(a_finish.iloc[0]),
            "b_pick_umaban": int(b_pick["umaban"]),
            "b_pick_odds": float(b_pick["odds"]) if pd.notna(b_pick["odds"]) else 0.0,
            "b_pick_finish": int(b_finish.iloc[0]),
            "time_diff": bias_dict["time_diff"] if bias_dict else np.nan,
            "frame_bias_score": bias_dict["frame_bias_score"] if bias_dict else np.nan,
            "fb_bias_score": bias_dict["fb_bias_score"] if bias_dict else np.nan,
        })

        if (idx + 1) % 200 == 0:
            print(f"  ... {idx+1}/{len(test_meta_use)} races (skipped={skipped})")

    summary = pd.DataFrame(rows)
    print(f"  total evaluated: {len(summary)}, skipped: {skipped}")

    print("[5/6] Subset analysis...")
    print("\n" + "=" * 60)
    print("OVERALL (course-profile defined)")
    print("=" * 60)
    report("OVERALL", summary)

    matched = summary[summary["_bias_matched"]]
    print("\n" + "=" * 60)
    print("BIAS-MATCHED ONLY")
    print("=" * 60)
    report("BIAS-MATCHED", matched)

    active_mask = (
        matched["time_diff"].abs() > 1.5
    ) | (matched["frame_bias_score"].fillna(0) != 0) | (matched["fb_bias_score"].fillna(0) != 0)
    active = matched[active_mask]
    print("\n" + "=" * 60)
    print("BIAS-ACTIVE ONLY")
    print("=" * 60)
    report("BIAS-ACTIVE", active)

    flat = matched[~active_mask]
    print("\n" + "=" * 60)
    print("BIAS-FLAT ONLY")
    print("=" * 60)
    report("BIAS-FLAT", flat)

    print("\n" + "=" * 60)
    print(f"[6/6] 兆候判定 (Phase 4-α, strength={args.strength}):")
    print("  - top1 ≥ +0.5pt 兆候 / ≥ +1pt 採用候補")
    print("  - roi  ≥ +2pt 兆候 / ≥ +5pt 採用候補")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
