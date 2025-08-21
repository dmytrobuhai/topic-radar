"""
Universal OpenAlex (via PyAlex).

Features:
- Multi-strategy retrieval (AND tokens => OR tokens => phrase)
- Date window by publication date (last N days)
- Pseudo-Relevance Feedback (PRF) from top docs (TF-IDF)
- Coverage-based scoring (title > abstract) + recency prior
- Dedupe, rank, JSON export

Usage:
  python -m papers.openalex_parser "computer vision industrial automation" --days 30 --max 10 --debug
"""

# papers/openalex_parser.py
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List, Optional

from core.parsers_helper import (
    tokens, compile_word_patterns, coverage, recency_score,
    prf_terms_from_pairs, compute_confidence, pass_threshold,
    DEFAULT_STOPWORDS
)

# --- PyAlex config ---
import config as cfg
PYALEX_EMAIL = cfg.EMAIL
try:
    import pyalex
    if PYALEX_EMAIL:
        pyalex.config.email = PYALEX_EMAIL
    pyalex.config.max_retries = 2
    pyalex.config.retry_backoff_factor = 0.25
    pyalex.config.retry_http_codes = [429, 500, 503]
    HAVE_PYALEX = True
except Exception:
    HAVE_PYALEX = False

# ---------- Constants ----------
PER_PAGE = 200
SLEEP_BETWEEN_CALLS = 1

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

def _since_iso(days: int) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).date().isoformat()

# ---------- Query building ----------
def build_initial_queries(topic: str) -> List[str]:
    toks = tokens(topic, stopwords=STOPWORDS) or ["universal", "search"]
    q1 = " ".join(toks)
    q2 = " OR ".join(toks)
    q3 = f"\"{' '.join(toks)}\""
    out: List[str] = []
    for q in (q1, q2, q3):
        if q not in out:
            out.append(q)
    return out

# ---------- Normalize ----------
def _reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not inv_idx:
        return None
    try:
        max_pos = max(max(pos_list) for pos_list in inv_idx.values())
        arr = [None] * (max_pos + 1)
        for word, pos_list in inv_idx.items():
            for pos in pos_list:
                if 0 <= pos < len(arr) and arr[pos] is None:
                    arr[pos] = word
        tokens_seq = [t for t in arr if t]
        return " ".join(tokens_seq).strip() or None
    except Exception:
        return None

def _authors(work: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    out = []
    for a in (work.get("authorships") or []):
        name = (a.get("author") or {}).get("display_name")
        aff  = None
        insts = a.get("institutions") or []
        if insts:
            aff = insts[0].get("display_name")
        out.append({"name": name, "affiliation": aff})
    return out


def _to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        if len(s) >= 10:
            s = s[:10]
        dt.date.fromisoformat(s)
        return s
    except Exception:
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return d.date().isoformat()
        except Exception:
            return None
        
def _normalize(work: Dict[str, Any]) -> Dict[str, Any]:
    title = work.get("display_name") or work.get("title")
    abstract = work.get("abstract") or _reconstruct_abstract(work.get("abstract_inverted_index"))
    pub_date = _to_iso_date(work.get("publication_date") or work.get("from_publication_date"))
    upd_date = _to_iso_date(work.get("updated_date") or pub_date)
    loc = work.get("primary_location") or {}
    abs_url = loc.get("landing_page_url") or (loc.get("source") or {}).get("homepage_url") or work.get("id")
    pdf_url = loc.get("pdf_url")
    doi = work.get("doi")
    doi_url = f"https://doi.org/{doi}" if doi else None
    openalex_id = work.get("id")
    cited_by = work.get("cited_by_count")
    return {
        "source": "OpenAlex",
        "openalex_id": openalex_id,
        "title": title,
        "summary": abstract,
        "published": pub_date,
        "updated": upd_date,
        "authors": _authors(work),
        "links": {"abs": abs_url, "pdf": pdf_url, "openalex": openalex_id},
        "doi": doi,
        "doi_url": doi_url,
        "cited_by_count": cited_by,
    }

# ---------- Retrieval ----------
def fetch_query_openalex(q: str, since_iso: str, want: int, debug: bool) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not HAVE_PYALEX:
        if debug:
            print("[DEBUG] PyAlex missing.")
        return items
    base = (
        pyalex.Works()
        .search(q)
        .filter(from_publication_date=since_iso)
        .select([
            "id","doi","display_name","publication_date","updated_date",
            "primary_location","authorships","cited_by_count","abstract_inverted_index","biblio"
        ])
    )
    got = 0
    try:
        for page in base.paginate(per_page=min(PER_PAGE, max(25, want))):
            for w in page:
                items.append(_normalize(w))
                got += 1
                if got >= want:
                    break
            if got >= want:
                break
            time.sleep(SLEEP_BETWEEN_CALLS)
    except Exception as e:
        if debug:
            print("[DEBUG] OpenAlex fetch error:", str(e))
    return items

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool) -> List[Dict[str, Any]]:
    qs = build_initial_queries(topic)
    if debug:
        print("[DEBUG] OpenAlex queries:")
        for q in qs: print("   ", q)
    since = _since_iso(days)
    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(60, min(PER_PAGE, pool_target))
    for qi, q in enumerate(qs, 1):
        chunk = fetch_query_openalex(q, since_iso=since, want=per_try, debug=debug)
        for r in chunk:
            key = r.get("doi") or r.get("openalex_id") or (r.get("title","").strip().lower())
            if key and key not in seen:
                seen.add(key); pool.append(r)
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------- Orchestrator ----------
def openalex_search(topic: str, days: int = 30, max_results: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
    if not HAVE_PYALEX:
        raise RuntimeError("PyAlex is not installed or failed to import")

    base_tokens = tokens(topic, stopwords=STOPWORDS)
    base_pats = compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 120)
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
    parser = argparse.ArgumentParser(description="OpenAlex search with PRF and lexical scoring")
    parser.add_argument("topic", type=str)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    results = openalex_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(results)}")
    for i, p in enumerate(results[:10], 1):
        conf = p.get("confidence", 0.0)
        date_show = p.get("updated") or p.get("published") or ""
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={conf:.1f}]  [{date_show}]")
        print(f"    abs: {p['links'].get('abs')}")
        if p['links'].get('pdf'):
            print(f"    pdf: {p['links'].get('pdf')}")
        if p.get("doi_url"):
            print(f"    doi: {p['doi_url']}")
    with open("openalex_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: openalex_sample.json")
