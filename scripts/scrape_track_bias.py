#!/usr/bin/env python3
"""
Scrape track bias articles from bloodline-trackbias.work
Span: 2019/5 〜 2026/4 (sitemap_periodical.xml で月別URL取得)

Output: data/track_bias_raw.jsonl
  1行 = 1記事 = {url, lastmod, title, body}

Design:
- 生テキスト保存。4軸抽出は parse_track_bias.py で後段
- Resumable: 既に取得済みURLはスキップ
- Rate limit: 2秒/リクエスト

Usage:
  python scripts/scrape_track_bias.py
"""
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "track_bias_raw.jsonl"
SITEMAP_URL = "https://www.bloodline-trackbias.work/sitemap_periodical.xml?year={year}&month={month}"
USER_AGENT = "Mozilla/5.0 (compatible; keiba-predictor-research/1.0)"
SLEEP = 2.0


def fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_urls_for_month(year: int, month: int):
    url = SITEMAP_URL.format(year=year, month=month)
    try:
        xml = fetch(url)
    except HTTPError as e:
        if e.code == 404:
            return []
        raise
    xml = re.sub(r'\sxmlns="[^"]+"', "", xml, count=1)
    root = ET.fromstring(xml)
    items = []
    for url_node in root.findall("url"):
        loc = url_node.find("loc")
        lastmod = url_node.find("lastmod")
        if loc is not None and loc.text:
            items.append({
                "url": loc.text,
                "lastmod": lastmod.text if lastmod is not None else None,
            })
    return items


def extract_text(html: str):
    title_match = re.search(r"<title>([^<]+)</title>", html)
    title = title_match.group(1).strip() if title_match else ""

    body_match = re.search(
        r'<div class="entry-content[^"]*"[^>]*>(.*?)<(?:footer|div class="entry-footer)',
        html,
        re.DOTALL,
    )
    body_html = body_match.group(1) if body_match else html

    body_html = re.sub(r"<script[^>]*>.*?</script>", "", body_html, flags=re.DOTALL)
    body_html = re.sub(r"<style[^>]*>.*?</style>", "", body_html, flags=re.DOTALL)
    body_html = re.sub(r"<img[^>]+alt=\"([^\"]*)\"[^>]*>", r" [IMG:\1] ", body_html)
    body_html = re.sub(r"<img[^>]*>", " [IMG] ", body_html)
    body = re.sub(r"<[^>]+>", "\n", body_html)
    body = re.sub(r"&nbsp;", " ", body)
    body = re.sub(r"&amp;", "&", body)
    body = re.sub(r"&lt;", "<", body)
    body = re.sub(r"&gt;", ">", body)
    body = re.sub(r"\n\s*\n", "\n\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    return title, body.strip()


def load_existing_urls():
    if not OUT_PATH.exists():
        return set()
    seen = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(rec["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def iter_months(start=(2019, 5), end=(2026, 4)):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = load_existing_urls()
    print(f"[init] already have {len(seen)} URLs", flush=True)

    all_urls = []
    for year, month in iter_months():
        try:
            items = get_urls_for_month(year, month)
            print(f"[sitemap {year}-{month:02d}] {len(items)} urls", flush=True)
            all_urls.extend(items)
        except Exception as e:
            print(f"[sitemap {year}-{month:02d}] ERROR: {e}", flush=True)
        time.sleep(SLEEP)

    unfetched = [u for u in all_urls if u["url"] not in seen]
    print(f"[urls] total {len(all_urls)}, unfetched {len(unfetched)}", flush=True)

    fetched = 0
    failed = 0
    with OUT_PATH.open("a", encoding="utf-8") as out:
        for item in unfetched:
            try:
                html = fetch(item["url"])
                title, body = extract_text(html)
                rec = {
                    "url": item["url"],
                    "lastmod": item["lastmod"],
                    "title": title,
                    "body": body,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                fetched += 1
                if fetched % 10 == 0:
                    print(f"[fetch] {fetched} ok, {failed} fail", flush=True)
            except Exception as e:
                print(f"[fetch] FAIL {item['url']}: {e}", flush=True)
                failed += 1
            time.sleep(SLEEP)

    print(f"[done] fetched {fetched}, failed {failed}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
