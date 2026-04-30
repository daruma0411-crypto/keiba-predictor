"""
Phase 4-β: Optuna による strength パラメータ独立最適化

Phase 4-α では一括 strength スカラ。 4-β では fb / fr / st / td を独立化。
NN 学習 + A 側 QMC は1回キャッシュ。各 trial は B 側のみ再計算。

目的関数: BIAS-ACTIVE subset の ROI を最大化。

実行:
  py -3.13 scripts/simulate_v9b_phase4_beta.py [--n-trials 100] [--qmc-n 10000]
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
import optuna

from src.predictor import CAT_FEATURES, FEATURES_V9, Predictor
from src.binary_parser import PLACE_CODES
from src.qmc_courses import COURSE_PROFILES, qmc_sim

# ============================================================
# bias text → score (Phase 4-α と同じ)
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


VENUE_KEY = {
    "札幌": "sapporo", "函館": "hakodate", "福島": "fukushima",
    "新潟": "niigata", "東京": "tokyo", "中山": "nakayama",
    "中京": "chukyo", "京都": "kyoto", "阪神": "hanshin", "小倉": "kokura",
}
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


def detect_course_key(venue, distance, is_turf):
    if not is_turf:
        return None
    vk = VENUE_KEY.get(venue)
    if not vk:
        return None
    return COURSE_KEY_MAP.get((vk, int(distance)))


def _safe(v):
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(f) else f


def apply_bias_to_profile(base, bias, s_fb, s_fr, s_st, s_td):
    """4成分で独立 strength 注入。"""
    if bias is None:
        return base
    fb = _safe(bias.get("fb_bias_score"))
    fr = _safe(bias.get("frame_bias_score"))
    st = _safe(bias.get("straight_bias_score"))
    td = _safe(bias.get("time_diff"))

    p = copy.deepcopy(base)

    # 前後
    p["style_bonus"]["nige"]   += -fb * s_fb * 1.0
    p["style_bonus"]["senkou"] += -fb * s_fb * 0.6
    p["style_bonus"]["sashi"]  += +fb * s_fb * 0.6
    p["style_bonus"]["oikomi"] += +fb * s_fb * 1.0

    # 内/外
    p["gate_bias"]["inner_senkou"] += -fr * s_fr
    p["gate_bias"]["outer_sashi"]  += -fr * s_fr
    p["gate_bias"]["outer_sashi"]  += -st * s_st * 0.5

    # 時計差
    factor = max(0.6, 1.0 - td * s_td * 0.05)
    p["pace_noise"]  *= factor
    p["noise_scale"] *= factor

    return p


# ============================================================
# 評価
# ============================================================
def compute_roi(picks_df):
    """picks_df: columns [pick_finish, pick_odds]"""
    payout = np.where(picks_df["pick_finish"] == 1, picks_df["pick_odds"].astype(float) * 100, 0.0)
    n = len(picks_df)
    if n == 0:
        return 0.0
    return float((payout.sum() - n * 100.0) / (n * 100.0))


def compute_top1(picks_df):
    return float((picks_df["pick_finish"] == 1).mean()) if len(picks_df) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--qmc-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", choices=["bias_active_roi", "overall_roi"], default="bias_active_roi")
    args = ap.parse_args()

    feat_path = ROOT / "data" / "features_v9b_2026.pkl"
    bias_path = ROOT / "data" / "track_bias_parsed.jsonl"
    out_path = ROOT / "data" / "phase4_beta_optuna_result.json"

    print(f"[settings] n_trials={args.n_trials}  qmc_n={args.qmc_n}  objective={args.objective}")

    print("[1/5] Load + train NN + predict (one-time)...")
    feat = pd.read_pickle(feat_path)
    bias = load_bias(bias_path)
    merged = join_bias(feat, bias)
    merged = merged[merged["_surface"] == "芝"].copy()
    train_df = filter_period(merged, "2019-05-01", "2024-12-31")
    test_df = filter_period(merged, "2025-01-01", "2026-12-31")

    available = [c for c in FEATURES_V9 if c in train_df.columns]
    pred = Predictor(numeric_features=available, cat_features=CAT_FEATURES)
    pred.train(train_df, ep=50, lr=0.003, seed=42)
    preds_full = pred.predict(test_df)
    preds_full["race_id"] = test_df["race_id"].values
    preds_full["finish"] = test_df["finish"].values
    preds_full["wakuban"] = test_df["wakuban"].values
    preds_full["avg_run_style"] = test_df["avg_run_style"].values
    preds_full["_bias_matched"] = test_df["_bias_matched"].values
    for col in ["time_diff", "frame_bias_score", "fb_bias_score", "straight_bias_score"]:
        preds_full[col] = test_df[col].values if col in test_df.columns else np.nan

    test_meta = test_df[["race_id", "_venue", "kyori", "is_turf"]].drop_duplicates(subset=["race_id"]).copy()
    test_meta["course_key"] = test_meta.apply(
        lambda r: detect_course_key(r["_venue"], r["kyori"], bool(r["is_turf"])), axis=1
    )
    test_meta_use = test_meta[test_meta["course_key"].notna()].copy()
    print(f"  evaluable races: {len(test_meta_use)}")

    print("[2/5] Pre-compute per-race data (ps, rf, bias_dict)...")
    race_records = []
    for meta in test_meta_use.itertuples(index=False):
        rid = meta.race_id
        course_key = meta.course_key
        race_preds = preds_full[preds_full["race_id"] == rid]
        if len(race_preds) < 4:
            continue
        rf = race_preds[["wakuban", "avg_run_style"]].reset_index(drop=True)
        ps = race_preds[["mu", "sigma", "horse_name", "umaban", "odds"]].reset_index(drop=True)
        bias_dict = None
        is_active = False
        if race_preds["_bias_matched"].any():
            r0 = race_preds.iloc[0]
            bias_dict = {
                "time_diff": r0["time_diff"],
                "frame_bias_score": r0["frame_bias_score"],
                "fb_bias_score": r0["fb_bias_score"],
                "straight_bias_score": r0["straight_bias_score"],
            }
            td_v = _safe(bias_dict["time_diff"])
            fr_v = _safe(bias_dict["frame_bias_score"])
            fb_v = _safe(bias_dict["fb_bias_score"])
            is_active = (abs(td_v) > 1.5) or (fr_v != 0) or (fb_v != 0)

        # finish/odds 辞書 (umaban → finish, odds)
        finish_lookup = dict(zip(race_preds["umaban"].astype(int), race_preds["finish"].astype(int)))
        odds_lookup = dict(zip(race_preds["umaban"].astype(int), race_preds["odds"].astype(float)))
        race_records.append({
            "race_id": rid,
            "course_key": course_key,
            "ps": ps,
            "rf": rf,
            "bias": bias_dict,
            "is_matched": bias_dict is not None,
            "is_active": is_active,
            "finish_lookup": finish_lookup,
            "odds_lookup": odds_lookup,
        })
    print(f"  per-race records: {len(race_records)}")

    print("[3/5] Compute A baseline (bias=None, one-time)...")
    a_picks = []
    for rec in race_records:
        mc_a = qmc_sim(rec["ps"], race_features=rec["rf"], course=rec["course_key"], n=args.qmc_n)
        a_pick = mc_a.iloc[0]
        umb = int(a_pick["umaban"])
        a_picks.append({
            "race_id": rec["race_id"],
            "is_matched": rec["is_matched"],
            "is_active": rec["is_active"],
            "pick_umaban": umb,
            "pick_finish": rec["finish_lookup"][umb],
            "pick_odds": rec["odds_lookup"].get(umb, 0.0),
        })
    a_df = pd.DataFrame(a_picks)
    a_overall_roi = compute_roi(a_df)
    a_active_roi = compute_roi(a_df[a_df["is_active"]])
    a_overall_top1 = compute_top1(a_df)
    a_active_top1 = compute_top1(a_df[a_df["is_active"]])
    print(f"  A: OVERALL  roi={a_overall_roi:+.4f} top1={a_overall_top1:.4f}")
    print(f"     ACTIVE   roi={a_active_roi:+.4f} top1={a_active_top1:.4f} (n={a_df['is_active'].sum()})")

    print("[4/5] Optuna optimization...")

    def evaluate_b(s_fb, s_fr, s_st, s_td):
        b_picks = []
        for rec in race_records:
            if rec["bias"] is None:
                # B = A (no bias to apply)
                pick_uma = a_picks[len(b_picks)]["pick_umaban"]
                b_picks.append({
                    "is_active": False,
                    "pick_finish": rec["finish_lookup"][pick_uma],
                    "pick_odds": rec["odds_lookup"].get(pick_uma, 0.0),
                })
                continue
            base = COURSE_PROFILES[rec["course_key"]]
            new_prof = apply_bias_to_profile(base, rec["bias"], s_fb, s_fr, s_st, s_td)
            tmp_key = f"__beta_tmp__{rec['course_key']}"
            COURSE_PROFILES[tmp_key] = new_prof
            try:
                mc_b = qmc_sim(rec["ps"], race_features=rec["rf"], course=tmp_key, n=args.qmc_n)
            finally:
                COURSE_PROFILES.pop(tmp_key, None)
            umb = int(mc_b.iloc[0]["umaban"])
            b_picks.append({
                "is_active": rec["is_active"],
                "pick_finish": rec["finish_lookup"][umb],
                "pick_odds": rec["odds_lookup"].get(umb, 0.0),
            })
        return pd.DataFrame(b_picks)

    def objective(trial):
        s_fb = trial.suggest_float("s_fb", 0.05, 0.30)
        s_fr = trial.suggest_float("s_fr", 0.05, 0.30)
        s_st = trial.suggest_float("s_st", 0.0, 0.20)
        s_td = trial.suggest_float("s_td", 0.0, 0.20)
        b_df = evaluate_b(s_fb, s_fr, s_st, s_td)
        if args.objective == "bias_active_roi":
            roi = compute_roi(b_df[b_df["is_active"]])
        else:
            roi = compute_roi(b_df)
        # Optuna maximizes
        return roi

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for i in range(args.n_trials):
        trial = study.ask()
        roi = objective(trial)
        study.tell(trial, roi)
        delta = roi - (a_active_roi if args.objective == "bias_active_roi" else a_overall_roi)
        if (i + 1) % 5 == 0 or i < 5:
            print(f"  trial {i+1:3d}/{args.n_trials}: roi={roi:+.4f} (Δ vs A {delta*100:+.2f}pt)  "
                  f"best={study.best_value:+.4f}  params={trial.params}")

    print("\n[5/5] Best result:")
    bp = study.best_params
    bv = study.best_value
    a_ref = a_active_roi if args.objective == "bias_active_roi" else a_overall_roi
    print(f"  best ROI: {bv:+.4f}")
    print(f"  Δ vs A:    {(bv-a_ref)*100:+.2f}pt")
    print(f"  best params: {bp}")

    # 採用判定 + フル subset 評価
    print("\n[verify] Full subset評価 with best params...")
    b_df = evaluate_b(bp["s_fb"], bp["s_fr"], bp["s_st"], bp["s_td"])
    b_overall_roi = compute_roi(b_df)
    b_active_roi = compute_roi(b_df[b_df["is_active"]])
    b_overall_top1 = compute_top1(b_df)
    b_active_top1 = compute_top1(b_df[b_df["is_active"]])
    print("\n" + "=" * 60)
    print("Verification with best params")
    print("=" * 60)
    print(f"  OVERALL      A: roi={a_overall_roi:+.4f} top1={a_overall_top1:.4f}")
    print(f"               B: roi={b_overall_roi:+.4f} top1={b_overall_top1:.4f}")
    print(f"               Δ: roi={(b_overall_roi-a_overall_roi)*100:+.2f}pt  top1={(b_overall_top1-a_overall_top1)*100:+.2f}pt")
    print(f"  BIAS-ACTIVE  A: roi={a_active_roi:+.4f} top1={a_active_top1:.4f}")
    print(f"               B: roi={b_active_roi:+.4f} top1={b_active_top1:.4f}")
    print(f"               Δ: roi={(b_active_roi-a_active_roi)*100:+.2f}pt  top1={(b_active_top1-a_active_top1)*100:+.2f}pt")

    # 保存
    out_path.write_text(json.dumps({
        "n_trials": args.n_trials,
        "qmc_n": args.qmc_n,
        "objective": args.objective,
        "best_params": bp,
        "best_objective": bv,
        "a_baseline": {
            "overall_roi": a_overall_roi,
            "active_roi": a_active_roi,
            "overall_top1": a_overall_top1,
            "active_top1": a_active_top1,
        },
        "b_best": {
            "overall_roi": b_overall_roi,
            "active_roi": b_active_roi,
            "overall_top1": b_overall_top1,
            "active_top1": b_active_top1,
        },
        "delta_pt": {
            "overall_roi": (b_overall_roi - a_overall_roi) * 100,
            "active_roi": (b_active_roi - a_active_roi) * 100,
        },
        "trials_top10": [
            {"params": t.params, "value": t.value}
            for t in sorted(study.trials, key=lambda x: -x.value)[:10]
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
