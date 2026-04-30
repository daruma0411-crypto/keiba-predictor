"""
馬場バイアスを QMC コース profile に注入するヘルパー。

Phase 4-β (2026-04-30) で確定した最適 strength を default として使用。
詳細は data/phase4_beta_optuna_result.json と issue #4 参照。

使い方:
  from src.qmc_bias import lookup_bias, apply_bias_to_profile
  from src.qmc_courses import COURSE_PROFILES

  bias = lookup_bias('2026-05-04', '東京', '芝')
  if bias is not None:
      modified = apply_bias_to_profile(COURSE_PROFILES['tokyo_turf_2000'], bias)
      # qmc_sim 呼ぶ前に COURSE_PROFILES に temp key で登録するなど
"""
import copy
import json
from pathlib import Path

import numpy as np


# Phase 4-β Optuna 最適値 (2026-04-30、bias_active_roi 目的、100 trials)
# A vs B BIAS-ACTIVE: roi +8.74pt / OVERALL: roi +6.18pt
DEFAULT_STRENGTHS = {
    "s_fb": 0.2537,   # 前後 (最重要)
    "s_fr": 0.0681,   # 内/外
    "s_st": 0.1484,   # 直線伸び
    "s_td": 0.0117,   # 時計差 (ほぼ無効)
}


# bias text → 順序スコア
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


def _text_to_score(text, mapping):
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


def _safe(v):
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(f) else f


def lookup_bias(date_str: str, venue: str, surface: str,
                jsonl_path: str = "data/track_bias_parsed.jsonl",
                prefer_kind: str = "予想") -> dict | None:
    """
    指定 date×venue×surface の bias を返す。
    prefer_kind="予想" を優先（事前予測）、無ければ "結果"。

    Returns dict with keys: time_diff, frame_bias_score, fb_bias_score, straight_bias_score
    or None if no match.
    """
    p = Path(jsonl_path)
    if not p.exists():
        return None

    candidates = {"予想": None, "結果": None}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("date") != date_str:
                continue
            if r.get("venue") != venue:
                continue
            if r.get("surface") != surface:
                continue
            kind = r.get("kind")
            if kind in candidates and candidates[kind] is None:
                candidates[kind] = r

    chosen = candidates.get(prefer_kind) or candidates.get("結果") or candidates.get("予想")
    if chosen is None:
        return None

    return {
        "kind": chosen.get("kind"),
        "time_diff": chosen.get("time_diff"),
        "frame_bias_score": _text_to_score(chosen.get("frame_bias"), FRAME_MAP),
        "fb_bias_score": _text_to_score(chosen.get("fb_bias"), FB_MAP),
        "straight_bias_score": _text_to_score(chosen.get("straight_bias"), STRAIGHT_MAP),
        "raw_frame_bias": chosen.get("frame_bias"),
        "raw_fb_bias": chosen.get("fb_bias"),
        "raw_straight_bias": chosen.get("straight_bias"),
    }


def apply_bias_to_profile(base: dict, bias: dict, strengths: dict = None) -> dict:
    """
    bias で QMC profile を動的修正。
      fb_bias_score → style_bonus.{nige,senkou,sashi,oikomi}
      frame_bias_score → gate_bias.{inner_senkou,outer_sashi}
      straight_bias_score → gate_bias.outer_sashi 微調整
      time_diff → pace_noise / noise_scale (高速馬場で変動↓)
    """
    if bias is None:
        return base
    s = dict(DEFAULT_STRENGTHS)
    if strengths:
        s.update(strengths)

    fb = _safe(bias.get("fb_bias_score"))
    fr = _safe(bias.get("frame_bias_score"))
    st = _safe(bias.get("straight_bias_score"))
    td = _safe(bias.get("time_diff"))

    p = copy.deepcopy(base)

    # 前後
    p["style_bonus"]["nige"]   += -fb * s["s_fb"] * 1.0
    p["style_bonus"]["senkou"] += -fb * s["s_fb"] * 0.6
    p["style_bonus"]["sashi"]  += +fb * s["s_fb"] * 0.6
    p["style_bonus"]["oikomi"] += +fb * s["s_fb"] * 1.0

    # 内/外
    p["gate_bias"]["inner_senkou"] += -fr * s["s_fr"]
    p["gate_bias"]["outer_sashi"]  += -fr * s["s_fr"]

    # 直線伸び (st が正→外伸び→outer_sashi 不利度↓)
    p["gate_bias"]["outer_sashi"]  += -st * s["s_st"] * 0.5

    # 時計差
    factor = max(0.6, 1.0 - td * s["s_td"] * 0.05)
    p["pace_noise"]  *= factor
    p["noise_scale"] *= factor

    return p


def format_bias_summary(bias: dict) -> str:
    """ユーザー向けに bias の中身を1行で要約。"""
    if bias is None:
        return "(no bias data)"
    kind = bias.get("kind", "?")
    parts = [f"kind={kind}"]
    if bias.get("time_diff") is not None:
        parts.append(f"time_diff={bias['time_diff']:+.1f}")
    if bias.get("frame_bias_score") is not None:
        parts.append(f"frame={bias['frame_bias_score']:+.2f}")
    if bias.get("fb_bias_score") is not None:
        parts.append(f"fb={bias['fb_bias_score']:+.2f}")
    if bias.get("straight_bias_score") is not None:
        parts.append(f"straight={bias['straight_bias_score']:+.2f}")
    return "  ".join(parts)
