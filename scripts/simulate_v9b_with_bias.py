"""
v9b A/B test: 馬場バイアス特徴量の追加効果検証 (設計A — 同窓比較)

学習: 2019-05-01 〜 2024-12-31 (約5.5年、bias data 取得可能期間)
検証: 2025-01-01 〜 2026-04-30 (forward)

A: 既存 FEATURES_V9 のみ
B: A + 馬場バイアス4軸 (time_diff / frame / straight / fb)

評価:
- 馬券内占有率 (top-5 推奨に top-3 着内が含まれる率)
- 単勝 Brier score (mu と 1着フラグ の差)

前提:
- data/features_v9b_2026.pkl 存在 (binary_parser でビルド済み)
- data/track_bias_parsed.jsonl 存在 (parse_track_bias.py で生成済み)
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

# テキスト → 順序スコア (text→ordinal)
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
    """テキスト中のキーワードを抽出して平均スコアを返す。範囲表記("フラット〜外")の中点に近い。"""
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


def join_bias(feat: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["date_str"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
    f["venue"] = f["place_code"].astype(str).str.zfill(2).map(PLACE_CODES)
    merged = f.merge(
        bias,
        left_on=["date_str", "venue", "surface"],
        right_on=["date", "venue", "surface"],
        how="left",
        suffixes=("", "_bias"),
    )
    return merged


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    d = pd.to_datetime(df["date"])
    return df[(d >= s) & (d <= e)].copy()


def evaluate_top5_top3(preds_df, race_id_col, finish_col):
    """
    各レースで mu 昇順に top-5 を選び、その中に finish<=3 の馬が含まれる率を返す
    (= 馬券内推奨カバレッジ。既存 v9b の "馬券内占有率80%" と整合する評価軸)
    """
    df = preds_df.copy()
    df["rank"] = df.groupby(race_id_col)["mu"].rank(ascending=True, method="first")
    top5 = df[df["rank"] <= 5]
    by_race = top5.groupby(race_id_col)[finish_col].apply(lambda x: (x <= 3).any())
    return by_race.mean()


def main():
    feat_path = ROOT / "data" / "features_v9b_2026.pkl"
    bias_path = ROOT / "data" / "track_bias_parsed.jsonl"

    if not feat_path.exists():
        print(f"[err] features pickle not found: {feat_path}")
        print("      → src/binary_parser.py でビルドが必要")
        return 1
    if not bias_path.exists():
        print(f"[err] bias jsonl not found: {bias_path}")
        print("      → scripts/scrape_track_bias.py + parse_track_bias.py を実行")
        return 1

    print("[1/6] Loading features...")
    feat = pd.read_pickle(feat_path)
    print(f"  loaded {len(feat)} rows")
    needed = ["date", "place_code", "surface", "race_id", "kakutei_jyuni"]
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        print(f"[err] missing columns in features pkl: {missing}")
        return 1

    print("[2/6] Loading bias data...")
    bias = load_bias(bias_path, kind="結果")
    print(f"  bias rows: {len(bias)} (kind=結果のみ)")
    print(f"  date range: {bias['date'].min()} 〜 {bias['date'].max()}")
    for col in ["time_diff", "frame_bias_score", "straight_bias_score", "fb_bias_score"]:
        n_nn = bias[col].notna().sum()
        print(f"  {col}: non-null {n_nn} ({100*n_nn/len(bias):.1f}%)")

    print("[3/6] Joining features ⨝ bias on (date, venue, surface)...")
    merged = join_bias(feat, bias)
    n_match = merged["time_diff"].notna().sum()
    print(f"  matched {n_match} / {len(merged)} ({100*n_match/len(merged):.1f}%)")

    print("[4/6] Train/Test split...")
    merged = merged[merged["surface"] == "芝"].copy()  # 既存と同じく芝のみ
    train_df = filter_period(merged, "2019-05-01", "2024-12-31")
    test_df = filter_period(merged, "2025-01-01", "2026-12-31")
    print(f"  train: {len(train_df)}, test: {len(test_df)}")

    # finish 列名: src/predictor.py の Predictor.train は 'finish' を期待しているが、
    # features.py は 'kakutei_jyuni' を使っている。リネーム。
    for d in (train_df, test_df):
        d["finish"] = d["kakutei_jyuni"]
        d["heads"] = d.get("heads", d.groupby("race_id")["ketto_num"].transform("count"))

    print("[5/6] Training A (baseline) and B (with bias)...")
    pred_a = Predictor(numeric_features=FEATURES_V9, cat_features=CAT_FEATURES)
    pred_a.train(train_df, ep=50, lr=0.003, seed=42)
    preds_a = pred_a.predict(test_df)

    # B 用: bias列の欠損は中央値で埋める (Predictor 内でも fillna されるが明示)
    train_b = train_df.copy()
    test_b = test_df.copy()
    for c in BIAS_FEATURES:
        med = train_b[c].median()
        train_b[c] = train_b[c].fillna(med)
        test_b[c] = test_b[c].fillna(med)

    pred_b = Predictor(
        numeric_features=FEATURES_V9 + BIAS_FEATURES,
        cat_features=CAT_FEATURES,
    )
    pred_b.train(train_b, ep=50, lr=0.003, seed=42)
    preds_b = pred_b.predict(test_b)

    print("[6/6] Evaluating...")
    preds_a["race_id"] = test_df["race_id"].values
    preds_a["finish"] = test_df["finish"].values
    preds_b["race_id"] = test_b["race_id"].values
    preds_b["finish"] = test_b["finish"].values

    rate_a = evaluate_top5_top3(preds_a, "race_id", "finish")
    rate_b = evaluate_top5_top3(preds_b, "race_id", "finish")
    delta = rate_b - rate_a
    print()
    print("=" * 50)
    print(f"A (baseline):    馬券内カバー率 {rate_a:.4f}")
    print(f"B (with bias):   馬券内カバー率 {rate_b:.4f}")
    print(f"Delta:           {delta:+.4f} ({delta*100:+.2f} pt)")
    print("=" * 50)
    print()
    print(f"成功条件: +3pt以上 → {'採用' if delta >= 0.03 else '不採用'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
