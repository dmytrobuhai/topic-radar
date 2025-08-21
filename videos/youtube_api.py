# youtube_parser.py
"""
YouTube search with date window, PRF expansion, lexical scoring, and ranking.

- Multi-strategy retrieval (phrase, AND-like tokens) within [now-days, now]
- Pseudo-Relevance Feedback (PRF) from title+description
- Confidence = title_coverage*w1 + desc_coverage*w2 + PRF_title*w3 + PRF_desc*w4 + recency*w5
- Dedupe, sort, JSON-friendly output, CLI

Env:
  setx YT_API_KEY "your key"


Usage:
  python -m videos.youtube_api "computer vision industrial automation" --days 10 --max 10 --debug
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparse

from core.parsers_helper import (
    tokens, compile_word_patterns, coverage, recency_score,
    prf_terms_from_pairs, compute_confidence, pass_threshold,
    http_get_json, DEFAULT_STOPWORDS
)
import config as cfg

# ---------------- Config ----------------
YT_KEY = cfg.YT_KEY
YT_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
HEADERS = {"User-Agent": "topic-radar/1.0 (youtube client)"}
TIMEOUT = 30
PER_PAGE = 30
SLEEP = 0.6

# Scoring weights
W_TITLE_BASE = 0.60
W_DESC_BASE  = 0.25
W_TITLE_PRF  = 0.10
W_DESC_PRF   = 0.03
W_RECENCY    = 0.02

# PRF params
PRF_TOP_DOCS  = 40
PRF_TOP_TERMS = 10
PRF_MIN_LEN   = 3
PRF_MAX_LEN   = 24

STOPWORDS = DEFAULT_STOPWORDS

# ---------------- Time helpers ----------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _to_rfc3339(dttm: dt.datetime) -> str:
    return dttm.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _date_window(days: int) -> Tuple[str, str]:
    end_dt = _now_utc()
    start_dt = (end_dt - dt.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return _to_rfc3339(start_dt), _to_rfc3339(end_dt)

# ---------------- HTTP ----------------
def _yt_get(url: str, params: Dict[str, Any], debug: bool=False) -> Dict[str, Any]:
    if not YT_KEY:
        raise RuntimeError("Missing YT_API_KEY env var")
    qp = {"key": YT_KEY, **params}
    return http_get_json(url, params=qp, headers=HEADERS, timeout=TIMEOUT, debug=debug)

# ---------------- Normalization ----------------
def _normalize_search_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if it.get("id", {}).get("kind") != "youtube#video":
        return None
    vid = it["id"].get("videoId")
    sn = it.get("snippet") or {}
    title = sn.get("title")
    desc = sn.get("description")
    published_iso = None
    if sn.get("publishedAt"):
        try:
            published_iso = dateparse.parse(sn["publishedAt"]).strftime("%Y-%m-%d")
        except Exception:
            published_iso = sn["publishedAt"]
    channel = sn.get("channelTitle")
    channel_id = sn.get("channelId")
    url = f"https://www.youtube.com/watch?v={vid}" if vid else None
    return {
        "source": "YouTube",
        "video_id": vid,
        "title": title,
        "description": desc,
        "published": published_iso,
        "channel": channel,
        "channel_id": channel_id,
        "url": url,
        "stats": {},
    }

def _attach_stats(items: List[Dict[str, Any]], debug: bool=False) -> None:
    ids = [it["video_id"] for it in items if it.get("video_id")]
    if not ids:
        return
    for i in range(0, len(ids), 30):
        chunk = ids[i:i+30]
        data = _yt_get(YT_VIDEOS, {
            "part": "statistics,contentDetails",
            "id": ",".join(chunk),
            "maxResults": 30
        }, debug=debug)
        by_id = {x["id"]: x for x in (data.get("items") or [])}
        for it in items:
            vid = it.get("video_id")
            if not vid or vid not in by_id:
                continue
            st = by_id[vid].get("statistics", {})
            it["stats"] = {
                "viewCount": int(st.get("viewCount", 0)) if st.get("viewCount") else None,
                "likeCount": int(st.get("likeCount", 0)) if st.get("likeCount") else None,
                "commentCount": int(st.get("commentCount", 0)) if st.get("commentCount") else None,
            }
        if debug:
            print(f"[DEBUG] stats attached for {len(chunk)} ids")
        time.sleep(SLEEP)

# ---------------- Retrieval ----------------
def build_initial_queries(topic: str) -> List[str]:
    toks = tokens(topic, stopwords=STOPWORDS) or ["universal", "search"]
    q_and = " ".join(toks)
    q_phrase = f"\"{' '.join(toks)}\""
    out: List[str] = []
    for q in (q_phrase, q_and):
        if q not in out:
            out.append(q)
    return out

def fetch_query(q: str, days: int, want: int, debug: bool=False) -> List[Dict[str, Any]]:
    after, before = _date_window(days)
    collected: List[Dict[str, Any]] = []
    page_token = None
    while len(collected) < want:
        params = {
            "part": "snippet",
            "type": "video",
            "q": q,
            "maxResults": min(PER_PAGE, want - len(collected)),
            "order": "date",
            "publishedAfter": after,
            "publishedBefore": before,
            "safeSearch": "none",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _yt_get(YT_SEARCH, params, debug=debug)
        items = data.get("items") or []
        for raw in items:
            norm = _normalize_search_item(raw)
            if norm:
                collected.append(norm)
                if len(collected) >= want:
                    break
        page_token = data.get("nextPageToken")
        if not page_token or len(collected) >= want:
            break
        time.sleep(SLEEP)
    _attach_stats(collected, debug=debug)
    return collected

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool=False) -> List[Dict[str, Any]]:
    qs = build_initial_queries(topic)
    if debug:
        print("[DEBUG] YouTube queries:")
        for q in qs: print("   ", q)
    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(60, min(PER_PAGE, pool_target))
    for qi, q in enumerate(qs, 1):
        chunk = fetch_query(q, days=days, want=per_try, debug=debug)
        for v in chunk:
            key = v.get("video_id") or v.get("url")
            if key and key not in seen:
                seen.add(key); pool.append(v)
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------------- Orchestrator ----------------
def youtube_search(topic: str, days: int = 14, max_results: int = 30, debug: bool=False) -> List[Dict[str, Any]]:
    base_tokens = tokens(topic, stopwords=STOPWORDS)
    base_pats = compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 150)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)

    prf = prf_terms_from_pairs(
        ((v.get("title"), v.get("description")) for v in pool),
        base_tokens,
        top_docs=min(PRF_TOP_DOCS, len(pool)),
        top_terms=PRF_TOP_TERMS,
        min_len=PRF_MIN_LEN,
        max_len=PRF_MAX_LEN,
        stopwords=STOPWORDS
    )
    prf = [t for t in prf if t not in base_tokens]
    if debug:
        print("[DEBUG] PRF terms:", prf)
    prf_pats = compile_word_patterns(prf) if prf else []

    weights = dict(
        W_TITLE_BASE=W_TITLE_BASE, W_ABS_BASE=W_DESC_BASE,
        W_TITLE_PRF=W_TITLE_PRF,  W_ABS_PRF=W_DESC_PRF, W_RECENCY=W_RECENCY
    )

    def _engagement(x: Dict[str, Any]) -> float:
        vc = (x.get("stats") or {}).get("viewCount") or 0
        return math.log1p(vc) / 15.0

    scored: List[Dict[str, Any]] = []
    for v in pool:
        tb, _ = coverage(v.get("title") or "", base_pats)
        db, _ = coverage(v.get("description") or "", base_pats)
        tp, _ = coverage(v.get("title") or "", prf_pats) if prf_pats else (0.0, [])
        dp, _ = coverage(v.get("description") or "", prf_pats) if prf_pats else (0.0, [])
        rec = recency_score(v.get("published"), half_life_days=21)
        conf = compute_confidence(
            title_base_cov=tb, abs_base_cov=db, title_prf_cov=tp, abs_prf_cov=dp,
            recency=rec, weights=weights, scale_100=True
        )
        v["confidence"] = conf
        if pass_threshold(conf, threshold=20.0):
            scored.append(v)

    scored.sort(
        key=lambda x: (x.get("confidence", 0.0), x.get("published") or "", _engagement(x)),
        reverse=True
    )
    return scored[:max_results]

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YouTube search with PRF and lexical scoring")
    ap.add_argument("topic", type=str)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    results = youtube_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(results)}")
    for i, v in enumerate(results[:10], 1):
        conf = v.get("confidence", 0.0)
        date_show = v.get("published") or ""
        ch = v.get("channel") or ""
        views = (v.get("stats") or {}).get("viewCount")
        print(f"{i:>2}. {v.get('title','')[:120]}  [conf={conf:.1f}]  [{date_show}]  ({ch})  views={views}")
        print(f"    url: {v.get('url')}")
    with open("youtube_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: youtube_sample.json")
