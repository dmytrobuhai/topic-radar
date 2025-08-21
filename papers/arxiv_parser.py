"""
Universal arXiv search.

Features:
- Advanced-like arXiv API queries with submittedDate window
- Multi-strategy retrieval (AND tokens => OR tokens => phrase in ti/abs)
- Pseudo-Relevance Feedback (PRF) expansion from top retrieved docs
- Lexical scoring: title coverage => abstract coverage; + recency prior
- Dedupe, rank by confidence, JSON export

Usage:
  python -m papers.arxiv_parser "computer vision industrial automation" --days 30 --max 10 --debug
"""

# papers/arxiv_parser.py
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
import feedparser

from core.parsers_helper import (
    tokens, compile_word_patterns, coverage, recency_score,
    prf_terms_from_pairs, compute_confidence, pass_threshold,
    now_utc, http_get_text, DEFAULT_STOPWORDS
)

# ---------- Config ----------
import config as cfg
EMAIL = cfg.EMAIL
HEADER = f"arxiv-parser/3.1 (mailto:{EMAIL})" if EMAIL else "arxiv-parser/3.1"
ARXIV_API = "https://export.arxiv.org/api/query"
PER_CALL = 100
SLEEP_BETWEEN_CALLS = 3

# Scoring weights (kept for app.py dynamic updates)
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

UTC = dt.timezone.utc

def _submitted_range(days: int) -> Tuple[str, str]:
    to_dt = now_utc()
    from_dt = to_dt - dt.timedelta(days=days)
    fmt = "%Y%m%d%H%M"
    return from_dt.strftime(fmt), to_dt.strftime(fmt)

# ---------- Query building ----------
def _and_field(tt: List[str], field: str = "all") -> str:
    return " AND ".join([f"{field}:{t}" for t in tt])

def _or_field(tt: List[str], field: str = "all") -> str:
    return " OR ".join([f"{field}:{t}" for t in tt])

def build_initial_queries(topic: str, days: int) -> List[str]:
    toks = tokens(topic, stopwords=STOPWORDS) or ["universal", "search"]
    frm, to = _submitted_range(days)
    date_clause = f"submittedDate:[{frm} TO {to}]"
    phrase = " ".join(toks)
    q1 = f"({_and_field(toks,'all')}) AND {date_clause}"
    q2 = f"({_or_field(toks,'all')}) AND {date_clause}"
    q3 = f'(ti:"{phrase}" OR abs:"{phrase}") AND {date_clause}'
    out: List[str] = []
    for q in (q1, q2, q3):
        if q not in out:
            out.append(q)
    return out

# ---------- HTTP ----------
def _fetch_raw(q: str, start: int, max_results: int, debug: bool) -> Optional[str]:
    params = {
        "search_query": q,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    }
    hdrs = {"User-Agent": HEADER}
    try:
        return http_get_text(ARXIV_API, params=params, headers=hdrs, debug=debug)
    except Exception as e:
        if debug:
            print("[DEBUG] arXiv GET error:", e)
        return None

# ---------- Parse & normalize ----------
def _entry_dates(e: feedparser.FeedParserDict) -> Tuple[Optional[str], Optional[str]]:
    pub, upd = None, None
    if getattr(e, "published", None):
        pub = str(e.published)[:10]
    elif getattr(e, "published_parsed", None):
        pub = dt.date(*e.published_parsed[:3]).isoformat()
    if getattr(e, "updated", None):
        upd = str(e.updated)[:10]
    elif getattr(e, "updated_parsed", None):
        upd = dt.date(*e.updated_parsed[:3]).isoformat()
    return pub, upd

def _abs_pdf_other(e: feedparser.FeedParserDict) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    abs_url = None; pdf_url = None; other = []
    for link in e.get("links", []) or []:
        href = link.get("href"); rel = link.get("rel"); typ = link.get("type"); ttl = link.get("title")
        if rel == "alternate" and href and "arxiv.org/abs/" in href:
            abs_url = href
        elif (ttl == "pdf" or typ == "application/pdf") and href:
            pdf_url = href
        else:
            other.append({"href": href, "rel": rel, "type": typ, "title": ttl})
    if not abs_url and e.get("link"):
        href = e.get("link")
        abs_url = href.replace("/pdf/", "/abs/").replace(".pdf", "") if "/pdf/" in href else href
    if not abs_url and pdf_url:
        abs_url = pdf_url.replace("/pdf/", "/abs/").replace(".pdf", "")
    return abs_url, pdf_url, other

def _authors(e: feedparser.FeedParserDict) -> List[Dict[str, Optional[str]]]:
    out = []
    for a in e.get("authors", []) or []:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        aff  = a.get("affiliation") if isinstance(a, dict) else getattr(a, "affiliation", None)
        if not aff:
            aff = a.get("arxiv_affiliation") if isinstance(a, dict) else getattr(a, "arxiv_affiliation", None)
        out.append({"name": name, "affiliation": aff})
    if not out and getattr(e, "author", None):
        out = [{"name": str(e.author), "affiliation": getattr(e, "arxiv_affiliation", None)}]
    return out

def _normalize(e: feedparser.FeedParserDict) -> Dict[str, Any]:
    abs_url, pdf_url, other_links = _abs_pdf_other(e)
    published_iso, updated_iso = _entry_dates(e)
    doi = getattr(e, "arxiv_doi", None) or getattr(e, "doi", None)
    doi_url = f"https://doi.org/{doi}" if doi else None
    abs_id_url = getattr(e, "id", None)
    arxiv_id = None; version = None
    if abs_id_url and "/abs/" in str(abs_id_url):
        arxiv_id = str(abs_id_url).split("/abs/")[-1]
        m = re.search(r"v(\d+)$", arxiv_id or "")
        if m: version = int(m.group(1))
    return {
        "source": "arXiv",
        "arxiv_id": arxiv_id,
        "version": version,
        "title": getattr(e, "title", None),
        "summary": getattr(e, "summary", None),
        "published": published_iso,
        "updated": updated_iso,
        "authors": _authors(e),
        "links": {"abs": abs_url, "pdf": pdf_url, "other": other_links},
        "doi": doi,
        "doi_url": doi_url,
    }

# ---------- Retrieval ----------
def fetch_query(q: str, want: int, debug: bool) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    start = 0
    while len(collected) < want:
        raw = _fetch_raw(q, start=start, max_results=min(PER_CALL, want - len(collected)), debug=debug)
        if not raw:
            break
        feed = feedparser.parse(raw)
        if getattr(feed, "bozo", 0) and debug:
            print("[DEBUG] feedparser bozo:", getattr(feed, "bozo_exception", ""))
        if not feed.entries:
            break
        for e in feed.entries:
            collected.append(_normalize(e))
            if len(collected) >= want:
                break
        try:
            per_page = int(getattr(feed.feed, "opensearch_itemsperpage", None) or
                           getattr(feed.feed, "opensearch_itemsPerPage", None) or
                           len(feed.entries))
            start_idx = int(getattr(feed.feed, "opensearch_startindex", None) or
                            getattr(feed.feed, "opensearch_startIndex", None) or start)
            start = start_idx + per_page
        except Exception:
            start += len(feed.entries)
        if len(collected) < want:
            time.sleep(SLEEP_BETWEEN_CALLS)
    return collected

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool) -> List[Dict[str, Any]]:
    qs = build_initial_queries(topic, days)
    if debug:
        print("[DEBUG] arXiv initial queries:")
        for q in qs: print("   ", q)
    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(50, min(PER_CALL, pool_target))
    for qi, q in enumerate(qs, 1):
        chunk = fetch_query(q, want=per_try, debug=debug)
        added = 0
        for r in chunk:
            key = r.get("arxiv_id") or (r.get("title", "").strip().lower())
            if key and key not in seen:
                seen.add(key); pool.append(r); added += 1
        if debug:
            print(f"[DEBUG] Q{qi} -> got={len(chunk)}  added_unique={added}  pool={len(pool)}")
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------- Orchestrator ----------
def arxiv_search(topic: str, days: int = 30, max_results: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
    base_tokens = tokens(topic, stopwords=STOPWORDS)
    base_pats = compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 100)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)

    # PRF expansion
    prf = prf_terms_from_pairs(
        ((e.get("title"), e.get("summary")) for e in pool),
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

    # Score
    prf_pats = compile_word_patterns(prf) if prf else []
    weights = dict(
        W_TITLE_BASE=W_TITLE_BASE, W_ABS_BASE=W_ABS_BASE,
        W_TITLE_PRF=W_TITLE_PRF,  W_ABS_PRF=W_ABS_PRF, W_RECENCY=W_RECENCY
    )
    scored: List[Dict[str, Any]] = []
    for e in pool:
        tb, _ = coverage(e.get("title") or "", base_pats)
        ab, _ = coverage(e.get("summary") or "", base_pats)
        tp, _ = coverage(e.get("title") or "", prf_pats) if prf_pats else (0.0, [])
        ap, _ = coverage(e.get("summary") or "", prf_pats) if prf_pats else (0.0, [])
        rec = recency_score(e.get("updated") or e.get("published"), half_life_days=60)
        conf = compute_confidence(
            title_base_cov=tb, abs_base_cov=ab, title_prf_cov=tp, abs_prf_cov=ap,
            recency=rec, weights=weights, scale_100=True
        )
        e["confidence"] = conf
        if pass_threshold(conf, threshold=25.0):
            scored.append(e)

    scored.sort(key=lambda x: (x.get("confidence", 0.0), x.get("updated") or "", x.get("published") or ""), reverse=True)
    return scored[:max_results]

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal arXiv search with PRF and lexical scoring")
    parser.add_argument("topic", type=str, help="Search topic")
    parser.add_argument("--days", type=int, default=30, help="Date window in days")
    parser.add_argument("--max", type=int, default=50, help="Max results")
    parser.add_argument("--debug", action="store_true", help="Debug logs")
    args = parser.parse_args()

    results = arxiv_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(results)}")
    for i, p in enumerate(results[:10], 1):
        conf = p.get("confidence", 0.0)
        date_show = p.get("updated") or p.get("published") or ""
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={conf:.1f}]  [{date_show}]")
        print(f"    abs: {(p.get('links') or {}).get('abs')}")
        if (p.get('links') or {}).get('pdf'):
            print(f"    pdf: {(p['links']).get('pdf')}")
    with open("arxiv_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: arxiv_sample.json")
