#!/usr/bin/env python3
"""
Parse data/track_bias_raw.jsonl into structured (date, venue, surface) rows.

Output: data/track_bias_parsed.jsonl
  1行 = 1 (date, venue, surface)

Strategy:
- Title → (year, month, days, venues, kind)
- Body内で各 venue の【解説】section を特定
- ・芝 / ・ダート で surface 分割
- 4軸 statement を regex 抽出 (時計/枠/直線の伸び/前後)
- 時計だけ数値化、他は raw text 保持（後段で categorical 化）

Usage:
  python scripts/parse_track_bias.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "track_bias_raw.jsonl"
OUTPUT = ROOT / "data" / "track_bias_parsed.jsonl"
PARSE_LOG = ROOT / "data" / "track_bias_parse.log"

VENUES_ALL = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]


def normalize_minus(s: str) -> str:
    return (s.replace("－", "-")
             .replace("−", "-")
             .replace("—", "-")
             .replace("―", "-")
             .replace("ー", "-"))


def parse_title(title: str):
    """例: '2026,4,25～26 トラックバイアス結果 (東京競馬場　京都競馬場　福島競馬場)'"""
    title_n = normalize_minus(title)
    m = re.search(r"(\d{4})[,，](\d{1,2})[,，](\d{1,2})(?:[~～-](\d{1,2}))?", title_n)
    if not m:
        return None
    year, month, day1, day2 = m.groups()
    days = [int(day1)]
    if day2:
        days.append(int(day2))
    venues = re.findall(r"(" + "|".join(VENUES_ALL) + r")競馬場", title)
    seen = []
    for v in venues:
        if v not in seen:
            seen.append(v)
    kind = "結果" if "バイアス結果" in title else ("予想" if "バイアス予想" in title else "unknown")
    return {
        "year": int(year),
        "month": int(month),
        "days": days,
        "venues": seen,
        "kind": kind,
    }


def find_venue_explanation_section(body: str, venue: str) -> str:
    """
    venue の【解説】 section を返す。
    body 内には venue が複数回出る（目次・詳細本文）。
    詳細本文側は ・芝 / ・ダート / 【解説】 を全部含むので、それでスコアリング。
    """
    venue_full = venue + "競馬場"
    candidates = []
    for m in re.finditer(re.escape(venue_full), body):
        pos = m.end()
        window = body[pos:pos + 6000]
        # 次の venue で切る
        next_venue_pos = len(window)
        for other in VENUES_ALL:
            if other == venue:
                continue
            m2 = re.search(re.escape(other + "競馬場"), window[10:])
            if m2:
                next_venue_pos = min(next_venue_pos, 10 + m2.start())
        section = window[:next_venue_pos]
        score = 0
        if "【解説】" in section:
            score += 1
        if "・芝" in section:
            score += 2
        if "・ダート" in section:
            score += 2
        candidates.append((score, pos, section))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: (-x[0], x[1]))
    score, _, section = candidates[0]
    if score == 0:
        return ""
    return section


def split_explanation_by_surface(section: str):
    """
    section から ・芝 / ・ダート の subsection を抽出。
    後ろの境界は 【今週末へ向けて】 / 【次開催へ向けて】 / 次のvenue。
    """
    end_marker = re.search(r"【今週末へ向けて】|【次開催へ向けて】", section)
    end_pos = end_marker.start() if end_marker else len(section)
    explain_start = section.find("【解説】")
    if explain_start == -1:
        return {}
    explain_text = section[explain_start:end_pos]

    result = {}
    m_shiba = re.search(r"・芝(?:\s|\n|$)", explain_text)
    m_dirt = re.search(r"・ダート(?:\s|\n|$)", explain_text)
    if m_shiba:
        end_shiba = m_dirt.start() if m_dirt else len(explain_text)
        result["芝"] = explain_text[m_shiba.end():end_shiba].strip()
    if m_dirt:
        result["ダート"] = explain_text[m_dirt.end():].strip()
    return result


def extract_axes(surface_text: str):
    """4軸の statement を抽出 (時計/枠/直線の伸び/前後)。"""
    text = surface_text.replace("\n", " ")
    axes = {"time": None, "frame": None, "straight": None, "fb": None}

    m = re.search(r"時計[はが、,]?\s*([^。]+?)。", text)
    if m:
        axes["time"] = m.group(1).strip()

    m_combined = re.search(r"枠と直線の伸び[はが、,]?\s*([^。]+?)。", text)
    if m_combined:
        axes["frame"] = m_combined.group(1).strip()
        axes["straight"] = m_combined.group(1).strip()
    else:
        m = re.search(r"(?<!直線の伸び)枠[はが、,]?\s*([^。]+?)。", text)
        if m:
            axes["frame"] = m.group(1).strip()
        m = re.search(r"直線の伸び[はが、,]?\s*([^。]+?)。", text)
        if m:
            axes["straight"] = m.group(1).strip()

    m = re.search(r"前後[はが、,]?\s*([^。]+?)。", text)
    if m:
        axes["fb"] = m.group(1).strip()

    return axes


def parse_time_values(axis_time: str, days_count: int):
    """
    時計 statement から数値を抽出。
    返り値: {'sat': float|None, 'sun': float|None}
    """
    if not axis_time:
        return {"sat": None, "sun": None}
    text = normalize_minus(axis_time)
    matches = re.findall(r"([\-\+±]?\d+(?:\.\d+)?)秒", text)
    values = []
    for s in matches:
        s = s.replace("±", "0")
        try:
            values.append(float(s))
        except ValueError:
            continue

    if not values:
        return {"sat": None, "sun": None}

    has_sat = "土曜" in text
    has_sun = "日曜" in text

    if len(values) >= 2 and has_sat and has_sun:
        # 範囲表記 (例: "-2.0秒～-1.5秒") を平均に圧縮しないため最初の2つを採用
        return {"sat": values[0], "sun": values[1]}
    elif len(values) >= 2:
        # "-2.4〜-2.0秒位" のようなレンジ → 平均
        avg = sum(values[:2]) / 2.0
        if days_count == 2:
            return {"sat": avg, "sun": avg}
        return {"sat": avg, "sun": None}
    elif len(values) == 1:
        if days_count == 2:
            return {"sat": values[0], "sun": values[0]}
        if has_sun and not has_sat:
            return {"sat": None, "sun": values[0]}
        if has_sat and not has_sun:
            return {"sat": values[0], "sun": None}
        return {"sat": values[0], "sun": values[0]}
    return {"sat": None, "sun": None}


def split_axis_by_day(axis_text: str):
    """
    軸テキストを土曜/日曜別に分解。
    例: '土曜日は展開次第、日曜日は展開次第〜差し' → {'sat': '展開次第', 'sun': '展開次第〜差し'}
        '2日間ともフラット' → {'sat': 'フラット', 'sun': 'フラット'}
    """
    if not axis_text:
        return {"sat": None, "sun": None}

    if "2日間とも" in axis_text or "両日とも" in axis_text:
        m = re.search(r"(?:2日間とも|両日とも)\s*(.+?)$", axis_text)
        v = m.group(1).strip() if m else axis_text.strip()
        return {"sat": v, "sun": v}

    sat = None
    sun = None
    m_sat = re.search(r"土曜日?\s*[はが、,]?\s*([^、,。]+?)(?=[、,。]|$|日曜)", axis_text)
    m_sun = re.search(r"日曜日?\s*[はが、,]?\s*([^、,。]+?)(?=[、,。]|$|土曜)", axis_text)
    if m_sat:
        sat = m_sat.group(1).strip()
    if m_sun:
        sun = m_sun.group(1).strip()

    if sat is None and sun is None:
        # 土日区別なし: 全体を両日に
        return {"sat": axis_text.strip(), "sun": axis_text.strip()}
    return {"sat": sat, "sun": sun}


def make_date(year, month, day):
    return f"{year:04d}-{month:02d}-{day:02d}"


def process_article(rec, log):
    title = rec.get("title", "")
    body = rec.get("body", "")
    url = rec.get("url", "")
    parsed_title = parse_title(title)
    if not parsed_title:
        log.write(f"[skip:title] {url}\n")
        return []
    if parsed_title["kind"] == "unknown":
        log.write(f"[skip:kind] {url} | {title}\n")
        return []
    if not parsed_title["venues"]:
        log.write(f"[skip:venue] {url}\n")
        return []

    rows = []
    days = parsed_title["days"]

    for venue in parsed_title["venues"]:
        section = find_venue_explanation_section(body, venue)
        if not section:
            log.write(f"[no-section] {venue} {url}\n")
            continue
        surfaces = split_explanation_by_surface(section)
        if not surfaces:
            log.write(f"[no-surface] {venue} {url}\n")
            continue

        for surface, surface_text in surfaces.items():
            axes_raw = extract_axes(surface_text)
            time_vals = parse_time_values(axes_raw["time"] or "", len(days))
            frame_d = split_axis_by_day(axes_raw["frame"] or "")
            straight_d = split_axis_by_day(axes_raw["straight"] or "")
            fb_d = split_axis_by_day(axes_raw["fb"] or "")

            day_to_label = {}
            if len(days) == 2:
                day_to_label[days[0]] = "sat"
                day_to_label[days[1]] = "sun"
            else:
                if "土曜" in title:
                    day_to_label[days[0]] = "sat"
                elif "日曜" in title:
                    day_to_label[days[0]] = "sun"
                else:
                    day_to_label[days[0]] = "sat"

            for day, lab in day_to_label.items():
                row = {
                    "date": make_date(parsed_title["year"], parsed_title["month"], day),
                    "venue": venue,
                    "surface": surface,
                    "kind": parsed_title["kind"],
                    "time_diff": time_vals.get(lab),
                    "frame_bias": frame_d.get(lab),
                    "straight_bias": straight_d.get(lab),
                    "fb_bias": fb_d.get(lab),
                    "raw_axis_time": axes_raw["time"],
                    "raw_axis_frame": axes_raw["frame"],
                    "raw_axis_straight": axes_raw["straight"],
                    "raw_axis_fb": axes_raw["fb"],
                    "src_url": url,
                }
                rows.append(row)
    return rows


def main():
    if not INPUT.exists():
        print(f"[err] input not found: {INPUT}", file=sys.stderr)
        return 1

    n_in = 0
    n_out = 0
    n_skip = 0
    with INPUT.open("r", encoding="utf-8") as f, \
         OUTPUT.open("w", encoding="utf-8") as out, \
         PARSE_LOG.open("w", encoding="utf-8") as log:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue
            rows = process_article(rec, log)
            if not rows:
                n_skip += 1
                continue
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"[done] articles in: {n_in}, rows out: {n_out}, articles skipped: {n_skip}")
    print(f"  output: {OUTPUT}")
    print(f"  log:    {PARSE_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
