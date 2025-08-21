# papers/hf_parser.py
"""
Hugging Face Papers parser.

Fixes:
- Robust date extraction: <date> tag, regex fallback, and page-level fallback.
- Query-first: crawl recent dailies with ?q=..., filter by window.
- Window-first: fetch month/week/day pages to cover window, filter by window.
- Merge + PRF + lexical scoring + recency; rank and return top-K.

Usage:
  python -m papers.hf_parser "computer vision industrial automation" --days 10 --max 10 --debug
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from dateutil import parser as dateparse

from core.parsers_helper import (
    tokens, compile_word_patterns, coverage, recency_score,
    prf_terms_from_pairs, compute_confidence, pass_threshold,
    http_get_text, DEFAULT_STOPWORDS, now_utc
)

# ---------- Config ----------
import config as cfg
EMAIL = cfg.EMAIL
BASE = "https://huggingface.co"
HEADERS = {"User-Agent": f"topic-radar/1.3 (HF Papers HTML crawler) {EMAIL}".strip()}
TIMEOUT = 30
SLEEP = 0.75
PER_PAGE_LIMIT = 300

# Scoring weights
W_TITLE_BASE = 0.55
W_ABS_BASE   = 0.20
W_TITLE_PRF  = 0.15
W_ABS_PRF    = 0.05
W_RECENCY    = 0.05

# PRF params
PRF_TOP_DOCS  = 30
PRF_TOP_TERMS = 12
PRF_MIN_LEN   = 3
PRF_MAX_LEN   = 24

STOPWORDS = DEFAULT_STOPWORDS

# ---------- Model ----------
@dataclass
class HFItem:
    source: str
    hf_url: Optional[str]
    hf_slug: Optional[str]
    title: Optional[str]
    summary: Optional[str]
    date: Optional[str]
    authors: Optional[str]
    links: Dict[str, Optional[str]]
    tags: List[str]
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- Date & parsing ----------
DATE_REGEX = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE
)

def _parse_card_date_text(txt: str) -> Optional[str]:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        d = dateparse.parse(txt, default=now_utc())
        dd = d.date()
        if dd > now_utc().date():
            dd = dt.date(dd.year - 1, dd.month, dd.day)
        return dd.isoformat()
    except Exception:
        return None

def _extract_date_from_card(art: BeautifulSoup) -> Optional[str]:
    date_el = art.select_one("date")
    if date_el:
        iso = _parse_card_date_text(date_el.get_text(strip=True))
        if iso:
            return iso
    txt = art.get_text(" ", strip=True)
    m = DATE_REGEX.search(txt)
    if m:
        iso = _parse_card_date_text(m.group(0))
        if iso:
            return iso
    return None

def _date_key(date_iso: Optional[str]) -> dt.date:
    """Make a sortable key from ISO date; None -> very old date."""
    try:
        return dt.date.fromisoformat((date_iso or "")[:10])
    except Exception:
        return dt.date(1970, 1, 1)

def _parse_cards(html: str, fallback_date: Optional[dt.date] = None) -> List[HFItem]:
    out: List[HFItem] = []
    soup = BeautifulSoup(html, "html.parser")
    arts = soup.select("article")
    if not arts:
        return out
    for art in arts[:PER_PAGE_LIMIT]:
        a_title = art.select_one("h3 a[href^='/papers/']")
        if not a_title:
            continue
        title = a_title.get_text(strip=True)
        href = a_title.get("href")
        slug = (href or "").split("/papers/")[-1] if href else None
        hf_url = f"{BASE}{href}" if href else None

        p = art.select_one("a[href^='/papers/'] p")
        summary = p.get_text(" ", strip=True) if p else None

        date_iso = _extract_date_from_card(art)
        if not date_iso and fallback_date is not None:
            date_iso = fallback_date.isoformat()

        parent_txt = art.get_text(" ", strip=True)
        m = re.search(r"(\d+\s*authors?)", parent_txt, re.IGNORECASE)
        authors = m.group(1) if m else None

        arxiv_id = slug if slug and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", slug) else None
        arxiv_abs = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
        pdf_url   = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None

        out.append(HFItem(
            source="HF Papers",
            hf_url=hf_url,
            hf_slug=slug,
            title=title,
            summary=summary,
            date=date_iso,
            authors=authors,
            links={"abs": hf_url, "arxiv": arxiv_abs, "pdf": pdf_url},
            tags=[],
        ))
    return out

# ---------- Fetch helpers ----------
def _get(url: str, debug: bool=False) -> str:
    if debug: print("[DEBUG] GET", url)
    return http_get_text(url, headers=HEADERS, timeout=TIMEOUT, debug=debug)

def _url_day(d: dt.date, q: Optional[str] = None) -> str:
    u = f"{BASE}/papers/date/{d.isoformat()}"
    return f"{u}?{urlencode({'q': q})}" if q else u

def _url_week(iso_year: int, iso_week: int) -> str:
    return f"{BASE}/papers/week/{iso_year}-W{iso_week:02d}"

def _url_month(year: int, month: int) -> str:
    return f"{BASE}/papers/month/{year}-{month:02d}"

def _daterange(days: int) -> List[dt.date]:
    today = now_utc().date()
    return [today - dt.timedelta(d) for d in range(days)]

def fetch_day(d: dt.date, q: Optional[str] = None, debug: bool=False) -> List[HFItem]:
    html = _get(_url_day(d, q=q), debug=debug); time.sleep(SLEEP)
    return _parse_cards(html, fallback_date=d)

def fetch_week(d: dt.date, debug: bool=False) -> List[HFItem]:
    iy, iw, _ = d.isocalendar()
    html = _get(_url_week(iy, iw), debug=debug); time.sleep(SLEEP)
    monday = dt.date.fromisocalendar(iy, iw, 1)
    return _parse_cards(html, fallback_date=monday)

def fetch_month(d: dt.date, debug: bool=False) -> List[HFItem]:
    html = _get(_url_month(d.year, d.month), debug=debug); time.sleep(SLEEP)
    fallback = dt.date(d.year, d.month, 15)
    return _parse_cards(html, fallback_date=fallback)

# ---------- Strategies ----------
def crawl_recent_query_pool(topic: str, days_back: int, pool_limit: int, debug: bool=False) -> List[HFItem]:
    q = " ".join(tokens(topic, stopwords=STOPWORDS)) or topic.strip()
    pool: List[HFItem] = []
    for d in _daterange(days_back):
        try:
            items = fetch_day(d, q=q, debug=debug)
        except Exception as e:
            if debug: print("[DEBUG] fetch_day(query) error", d, str(e))
            items = []
        pool.extend(items)
        if len(pool) >= pool_limit:
            break
    # unique by slug/arxiv/url/title
    seen = set(); uniq: List[HFItem] = []
    for it in pool:
        key = it.links.get("arxiv") or it.hf_slug or (it.title or "").strip().lower()
        if key and key not in seen:
            seen.add(key); uniq.append(it)
    return uniq[:pool_limit]

def _months_in_range(start: dt.date, end: dt.date) -> List[Tuple[int,int]]:
    y, m = start.year, start.month
    res = []
    while True:
        res.append((y, m))
        if y == end.year and m == end.month:
            break
        m += 1
        if m > 12:
            m = 1; y += 1
    return res

def _weeks_in_range(start: dt.date, end: dt.date) -> List[Tuple[int,int]]:
    seen = set(); res: List[Tuple[int,int]] = []
    cur = start
    while cur <= end:
        iy, iw, _ = cur.isocalendar()
        if (iy, iw) not in seen:
            seen.add((iy, iw)); res.append((iy, iw))
        cur += dt.timedelta(days=7)
    iy, iw, _ = end.isocalendar()
    if (iy, iw) not in seen:
        res.append((iy, iw))
    return res

def fetch_window(days: int, debug: bool=False) -> List[HFItem]:
    today = now_utc().date()
    start = today - dt.timedelta(days=days)
    pool: List[HFItem] = []
    if days > 31:
        for y, m in _months_in_range(start, today):
            try:
                items = fetch_month(dt.date(y, m, 15), debug=debug)
            except Exception as e:
                if debug: print("[DEBUG] fetch_month error", y, m, str(e)); items = []
            pool.extend(items)
    elif days > 7:
        for iy, iw in _weeks_in_range(start, today):
            try:
                monday = dt.date.fromisocalendar(iy, iw, 1)
                items = fetch_week(monday, debug=debug)
            except Exception as e:
                if debug: print("[DEBUG] fetch_week error", iy, iw, str(e)); items = []
            pool.extend(items)
    else:
        for d in _daterange(days):
            try:
                items = fetch_day(d, q=None, debug=debug)
            except Exception as e:
                if debug: print("[DEBUG] fetch_day error", d, str(e)); items = []
            pool.extend(items)
    # unique
    seen = set(); uniq: List[HFItem] = []
    for it in pool:
        key = it.links.get("arxiv") or it.hf_slug or (it.title or "").strip().lower()
        if key and key not in seen:
            seen.add(key); uniq.append(it)
    cutoff = now_utc().date() - dt.timedelta(days=days)
    return [it for it in uniq if (it.date and dt.date.fromisoformat(it.date) >= cutoff)]

# ---------- Orchestrator ----------
def hf_search(topic: str, days: int = 30, max_results: int = 50, debug: bool=False) -> List[Dict[str, Any]]:
    def _date_key(date_iso: Optional[str]) -> dt.date:
        try:
            return dt.date.fromisoformat((date_iso or "")[:10])
        except Exception:
            return dt.date(1970, 1, 1)

    base_tokens = tokens(topic, stopwords=STOPWORDS)
    base_pats = compile_word_patterns(base_tokens)

    q_pool = crawl_recent_query_pool(
        topic,
        days_back=max(days, 60),                 
        pool_limit=max(2*max_results, 100),
        debug=debug
    )
    w_pool = fetch_window(days=days, debug=debug)

    merged: List[HFItem] = []
    seen = set()
    for it in (q_pool + w_pool):
        key = it.links.get("arxiv") or it.hf_slug or (it.title or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(it)

    today = now_utc().date()
    cutoff = today - dt.timedelta(days=days)
    in_window: List[HFItem] = []
    for it in merged:
        if not it.date:
            continue  
        try:
            d = dt.date.fromisoformat(it.date[:10])
        except Exception:
            continue
        if cutoff <= d <= today:
            in_window.append(it)

    if debug:
        print(f"[DEBUG] merged={len(merged)}  in_window(with date)={len(in_window)}  window=[{cutoff}..{today}]")

    prf = prf_terms_from_pairs(
        ((it.title, it.summary) for it in in_window),
        base_tokens,
        top_docs=min(PRF_TOP_DOCS, len(in_window)),
        top_terms=PRF_TOP_TERMS,
        min_len=PRF_MIN_LEN,
        max_len=PRF_MAX_LEN,
        stopwords=STOPWORDS
    )
    prf = [t for t in prf if t not in base_tokens]
    prf_pats = compile_word_patterns(prf) if prf else []
    if debug:
        print("[DEBUG] PRF terms:", prf)

    weights = dict(
        W_TITLE_BASE=W_TITLE_BASE, W_ABS_BASE=W_ABS_BASE,
        W_TITLE_PRF=W_TITLE_PRF,  W_ABS_PRF=W_ABS_PRF, W_RECENCY=W_RECENCY
    )

    scored: List[HFItem] = []
    for it in in_window:
        tb, _ = coverage(it.title or "", base_pats)
        ab, _ = coverage(it.summary or "", base_pats)
        tp, _ = coverage(it.title or "", prf_pats) if prf_pats else (0.0, [])
        ap, _ = coverage(it.summary or "", prf_pats) if prf_pats else (0.0, [])
        rec = recency_score(it.date, half_life_days=60)
        conf = compute_confidence(
            title_base_cov=tb, abs_base_cov=ab, title_prf_cov=tp, abs_prf_cov=ap,
            recency=rec, weights=weights, scale_100=True
        )
        it.confidence = conf
        if pass_threshold(conf, threshold=25.0):
            scored.append(it)

    scored.sort(key=lambda x: (_date_key(x.date), float(x.confidence)), reverse=True)

    return [it.to_dict() for it in scored[:max_results]]


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HF Papers parser")
    ap.add_argument("topic", type=str)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    res = hf_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(res)}")
    for i, p in enumerate(res[:10], 1):
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={p.get('confidence',0):.1f}]  [{p.get('date','')}]")
        print(f"    HF:  {p['links'].get('abs')}")
        if p['links'].get('arxiv'):
            print(f"    abs: {p['links']['arxiv']}")
        if p['links'].get('pdf'):
            print(f"    pdf: {p['links']['pdf']}")
    with open("hf_sample.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        print("Saved: hf_sample.json")
