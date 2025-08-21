"""
Aggregate tech/news RSS and rank by relevance+recency.

Sources (RSS):
  - Microsoft Research Blog
  - Hugging Face Blog
  - NVIDIA Developer Blog
  - OpenCV (News)
  - Towards Data Science
  - embedded.com


Usage:
  python -m news.news_parser "computer vision industrial automation" --days 30 --max 10 --debug
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparse

from core.parsers_helper import (
    tokens, compile_word_patterns, coverage, recency_score,
    prf_terms_from_pairs, compute_confidence, pass_threshold,
    http_get_text, DEFAULT_STOPWORDS
)

# ---------------- Config ----------------
import config as cfg
EMAIL = cfg.EMAIL
HEADERS = {"User-Agent": f"topic-radar/1.0 (news rss client; contact:{EMAIL})"}
TIMEOUT = 25
SLEEP = 0.5

FEEDS = [
    ("Microsoft Research Blog", "https://www.microsoft.com/en-us/research/blog/feed/"),
    ("Hugging Face Blog",       "https://huggingface.co/blog/feed.xml"),
    ("NVIDIA Developer Blog",   "https://developer.nvidia.com/blog/feed/"),
    ("OpenCV News",             "https://opencv.org/feed/"),
    ("Towards Data Science",    "https://towardsdatascience.com/feed"),
    ("embedded.com",            "https://www.embedded.com/feed/"),
]

# Scoring weights
W_TITLE_BASE = 0.60
W_SUMM_BASE  = 0.25
W_TITLE_PRF  = 0.10
W_SUMM_PRF   = 0.03
W_RECENCY    = 0.02

# PRF params
PRF_TOP_DOCS  = 40
PRF_TOP_TERMS = 10
PRF_MIN_LEN   = 3
PRF_MAX_LEN   = 24

STOPWORDS = DEFAULT_STOPWORDS

# ---------------- Fetch & parse ----------------
def http_get_feed(url: str, debug: bool=False) -> feedparser.FeedParserDict:
    raw = http_get_text(url, headers=HEADERS, timeout=TIMEOUT, debug=debug)
    feed = feedparser.parse(raw)
    if getattr(feed, "bozo", 0) and debug:
        print("[DEBUG] feedparser bozo:", getattr(feed, "bozo_exception", ""))
    time.sleep(SLEEP)
    return feed

def parse_entry_date(entry: feedparser.FeedParserDict) -> Optional[str]:
    if getattr(entry, "published_parsed", None):
        d = dt.date(*entry.published_parsed[:3])
        return d.isoformat()
    if getattr(entry, "updated_parsed", None):
        d = dt.date(*entry.updated_parsed[:3])
        return d.isoformat()
    for key in ("published", "updated", "created", "date"):
        if getattr(entry, key, None):
            try:
                d = dateparse.parse(getattr(entry, key)).date()
                if d > dt.date.today():
                    d = d - dt.timedelta(days=1)
                return d.isoformat()
            except Exception:
                continue
    return None

def strip_html(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    try:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    except Exception:
        return text

def normalize_entry(source: str, e: feedparser.FeedParserDict) -> Dict[str, Any]:
    title = getattr(e, "title", None)
    link  = getattr(e, "link", None)
    summary = getattr(e, "summary", None)
    if not summary:
        try:
            content = getattr(e, "content", None)
            if content and isinstance(content, list) and content:
                summary = content[0].get("value")
        except Exception:
            pass
    summary = strip_html(summary)
    date_iso = parse_entry_date(e)
    return {
        "source": source,
        "title": title,
        "summary": summary,
        "url": link,
        "date": date_iso,
    }

# ---------------- Orchestrator ----------------
def news_search(topic: str, days: int = 14, max_results: int = 30, debug: bool=False) -> List[Dict[str, Any]]:
    base_tokens = tokens(topic, stopwords=STOPWORDS)
    base_pats = compile_word_patterns(base_tokens)

    items: List[Dict[str, Any]] = []
    for name, url in FEEDS:
        try:
            feed = http_get_feed(url, debug=debug)
            for e in feed.entries:
                items.append(normalize_entry(name, e))
            if debug:
                print(f"[DEBUG] {name}: +{len(feed.entries)} (pool={len(items)})")
        except Exception as ex:
            if debug:
                print(f"[DEBUG] fetch error {name}:", str(ex))

    cut = dt.date.today() - dt.timedelta(days=days)
    items = [it for it in items if (it.get("date") and dt.date.fromisoformat(it["date"]) >= cut)]

    # Deduplicate by URL or title
    seen = set(); uniq: List[Dict[str, Any]] = []
    for it in items:
        key = (it.get("url") or "").split("?")[0] or (it.get("title","").strip().lower())
        if key and key not in seen:
            seen.add(key); uniq.append(it)
    items = uniq

    prf = prf_terms_from_pairs(
        ((it.get("title"), it.get("summary")) for it in items),
        base_tokens,
        top_docs=min(PRF_TOP_DOCS, len(items)),
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
        W_TITLE_BASE=W_TITLE_BASE, W_ABS_BASE=W_SUMM_BASE,
        W_TITLE_PRF=W_TITLE_PRF,  W_ABS_PRF=W_SUMM_PRF, W_RECENCY=W_RECENCY
    )

    scored: List[Dict[str, Any]] = []
    for it in items:
        tb, _ = coverage(it.get("title") or "", base_pats)
        sb, _ = coverage(it.get("summary") or "", base_pats)
        tp, _ = coverage(it.get("title") or "", prf_pats) if prf_pats else (0.0, [])
        sp, _ = coverage(it.get("summary") or "", prf_pats) if prf_pats else (0.0, [])
        rec = recency_score(it.get("date"), half_life_days=14)
        conf = compute_confidence(
            title_base_cov=tb, abs_base_cov=sb, title_prf_cov=tp, abs_prf_cov=sp,
            recency=rec, weights=weights, scale_100=True
        )
        it["confidence"] = conf
        if pass_threshold(conf, threshold=25.0):
            scored.append(it)

    scored.sort(key=lambda x: (x.get("confidence", 0.0), x.get("date") or ""), reverse=True)
    return scored[:max_results]

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="News RSS parser with PRF and lexical scoring")
    ap.add_argument("topic", type=str)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    res = news_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(res)}")
    for i, p in enumerate(res[:10], 1):
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={p.get('confidence',0):.1f}]  [{p.get('date','')}]  ({p.get('source','')})")
        print(f"    url: {p.get('url')}")
    with open("news_sample.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        print("Saved: news_sample.json")
