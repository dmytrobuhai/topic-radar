# crossref_parser.py
"""
Crossref 'works' search with date window, PRF expansion, lexical scoring, and ranking.

- Uses Crossref REST API (/works) with filters:
  from-pub-date / until-pub-date (primary), with fallback on created/indexed for recency scoring.
- Multi-strategy retrieval:
  * query.title (AND tokens)
  * query (OR tokens)
  * query.bibliographic (quoted phrase)
- PRF (pseudo-relevance feedback via simple TF-IDF) to expand vocabulary and re-query.
- Coverage-based scoring (title > abstract) + recency prior.
- Dedupe by DOI/URL/title; JSON output; CLI.

Usage:
  python -m papers.crossref_parser "computer vision industrial automation" --days 30 --max 10 --debug
"""

# papers/crossref_parser.py
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparse

from core.parsers_helper import (
    tokens, compile_word_patterns, coverage, recency_score,
    prf_terms_from_pairs, compute_confidence, pass_threshold,
    http_get_json, DEFAULT_STOPWORDS
)

# ---------------- Config ----------------
CROSSREF_API = "https://api.crossref.org/works"
import config as cfg
MAILTO = cfg.EMAIL
HEADER = f"topic-radar/1.0 (Crossref client; mailto:{MAILTO})" if MAILTO else "topic-radar/1.0"
TIMEOUT = 40
SLEEP = 0.75
PER_PAGE = 200
MAX_CURSOR_PAGES = 10

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

# ---------------- Time helpers ----------------
def _date_window(days: int) -> Tuple[str, str]:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    return start.isoformat(), end.isoformat()

def _first_date(parts: Optional[Dict[str, Any]]) -> Optional[str]:
    if not parts:
        return None
    try:
        dp = parts.get("date-parts") or []
        if not dp or not dp[0]:
            return None
        y = dp[0][0]
        m = dp[0][1] if len(dp[0]) > 1 else 1
        d = dp[0][2] if len(dp[0]) > 2 else 1
        return dt.date(int(y), int(m), int(d)).isoformat()
    except Exception:
        return None

# ---------------- HTTP ----------------
def _get(params: Dict[str, Any], debug: bool=False) -> Dict[str, Any]:
    if MAILTO:
        params = {**params, "mailto": MAILTO}
    return http_get_json(
        CROSSREF_API, params=params,
        headers={"User-Agent": HEADER}, timeout=TIMEOUT, debug=debug
    )

# ---------------- Normalization ----------------
def _best_title(item: Dict[str, Any]) -> Optional[str]:
    t = item.get("title") or []
    if isinstance(t, list) and t:
        return str(t[0]).strip() or None
    if isinstance(t, str):
        return t.strip() or None
    return None

def _best_pub_date(item: Dict[str, Any]) -> Optional[str]:
    for key in ("published-online", "published-print", "issued"):
        iso = _first_date(item.get(key))
        if iso:
            return iso
    for key in ("created", "indexed"):
        try:
            dtxt = item[key]["date-time"]
            return dateparse.parse(dtxt).date().isoformat()
        except Exception:
            continue
    return None

def _authors(item: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    out = []
    for a in item.get("author", []) or []:
        name = " ".join([a.get("given",""), a.get("family","")]).strip() or a.get("name")
        aff  = None
        affs = a.get("affiliation") or []
        if affs:
            aff = affs[0].get("name")
        out.append({"name": name, "affiliation": aff})
    return out

def _links(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    abs_url = item.get("URL")
    pdf_url = None
    for lnk in item.get("link", []) or []:
        if (lnk.get("content-type") == "application/pdf" or (lnk.get("URL","").endswith(".pdf"))):
            pdf_url = lnk.get("URL")
            break
    doi = item.get("DOI")
    doi_url = f"https://doi.org/{doi}" if doi else None
    return {"abs": abs_url, "pdf": pdf_url, "doi": doi_url}

def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    title = _best_title(item)
    abstract = item.get("abstract")
    if abstract:
        # strip rudimentary HTML
        abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
    pub = _best_pub_date(item)
    upd = None
    try:
        upd = dateparse.parse((item.get("indexed") or {}).get("date-time")).date().isoformat()
    except Exception:
        upd = pub
    links = _links(item)
    return {
        "source": "Crossref",
        "doi": item.get("DOI"),
        "title": title,
        "summary": abstract,         
        "abstract": abstract,         
        "published": pub,
        "updated": upd,
        "authors": _authors(item),
        "links": links,
    }

# ---------------- Query building ----------------
def build_initial_queries(topic: str) -> List[Dict[str, Any]]:
    toks = tokens(topic, stopwords=STOPWORDS) or ["universal", "search"]
    phrase = " ".join(toks)
    return [
        {"query.title": " ".join(toks)},          # AND-like
        {"query": " OR ".join(toks)},             # OR-like
        {"query.bibliographic": f"\"{phrase}\""}  # phrase
    ]

def _filter_for_window(days: int) -> str:
    frm, to = _date_window(days)
    return f"from-pub-date:{frm},until-pub-date:{to}"

# ---------------- Retrieval ----------------
def fetch_query(params_core: Dict[str, Any], days: int, want: int, debug: bool) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    cursor = "*"
    pages = 0
    while len(collected) < want and pages < MAX_CURSOR_PAGES:
        rows = min(PER_PAGE, want - len(collected))
        params = {
            "rows": rows,
            "cursor": cursor,
            "filter": _filter_for_window(days),
            **params_core
        }
        data = _get(params, debug=debug)
        msg = data.get("message") or {}
        items = msg.get("items") or []
        if not items:
            break
        for it in items:
            collected.append(_normalize(it))
            if len(collected) >= want:
                break
        cursor = msg.get("next-cursor")
        pages += 1
        if len(collected) < want:
            time.sleep(SLEEP)
    return collected

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool) -> List[Dict[str, Any]]:
    qs = build_initial_queries(topic)
    if debug:
        print("[DEBUG] Crossref initial queries:")
        for q in qs: print("   ", q)
    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(60, min(PER_PAGE, pool_target))
    for qi, core in enumerate(qs, 1):
        chunk = fetch_query(core, days=days, want=per_try, debug=debug)
        for r in chunk:
            key = r.get("doi") or (r.get("links") or {}).get("abs") or (r.get("title","").strip().lower())
            if key not in seen and key:
                seen.add(key); pool.append(r)
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------------- Orchestrator ----------------
def crossref_search(topic: str, days: int = 30, max_results: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
    base_tokens = tokens(topic, stopwords=STOPWORDS)
    base_pats = compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 150)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)

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
        rec = recency_score(e.get("published") or e.get("updated"), half_life_days=60)
        conf = compute_confidence(
            title_base_cov=tb, abs_base_cov=ab, title_prf_cov=tp, abs_prf_cov=ap,
            recency=rec, weights=weights, scale_100=True
        )
        e["confidence"] = conf
        if pass_threshold(conf, threshold=25.0):
            scored.append(e)

    scored.sort(key=lambda x: (x.get("confidence", 0.0), x.get("published") or "", x.get("updated") or ""), reverse=True)
    return scored[:max_results]

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossref works search with PRF and lexical scoring")
    parser.add_argument("topic", type=str)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    results = crossref_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(results)}")
    for i, p in enumerate(results[:10], 1):
        conf = p.get("confidence", 0.0)
        date_show = p.get("published") or p.get("updated") or ""
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={conf:.1f}]  [{date_show}]")
        print(f"    abs: {(p.get('links') or {}).get('abs')}")
        if (p.get('links') or {}).get('pdf'):
            print(f"    pdf: {(p['links']).get('pdf')}")
        if (p.get('links') or {}).get('doi'):
            print(f"    doi: {(p['links']).get('doi')}")
    with open("crossref_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: crossref_sample.json")
