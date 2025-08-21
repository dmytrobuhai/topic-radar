"""
Universal arXiv search.

Features:
- Advanced-like arXiv API queries with submittedDate window
- Multi-strategy retrieval (AND tokens => OR tokens => phrase in ti/abs)
- Pseudo-Relevance Feedback (PRF) expansion from top retrieved docs
- Lexical scoring: title coverage => abstract coverage; + recency prior
- Dedupe, rank by confidence, JSON export

Usage:
  python arxiv_parser.py "computer vision industrial automation" --days 30 --max 10 --debug
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import feedparser  

# ---------- Config ----------
EMAIL = os.getenv("EMAIL", "").strip()
HEADER = f"arxiv-parser/3.1 (mailto:{EMAIL})" 
ARXIV_API = "https://export.arxiv.org/api/query"
PER_CALL = 100
SLEEP_BETWEEN_CALLS = 3

# Scoring weights
W_TITLE_BASE = 0.55
W_ABS_BASE   = 0.20
W_TITLE_PRF  = 0.15
W_ABS_PRF    = 0.05
W_RECENCY    = 0.05

# PRF params
PRF_TOP_DOCS  = 30    # number of top docs to mine terms from
PRF_TOP_TERMS = 12    # number of expansion terms to keep
PRF_MIN_LEN   = 3     # min token length considered for PRF
PRF_MAX_LEN   = 24    # max token length considered for PRF

# Minimal stopword set
STOPWORDS = {
    "and", "or", "for", "the", "a", "an", "of", "in", "on", "to", "with", "by", "at",
    "from", "into", "using", "via", "based", "towards", "toward", "approach", "method",
    "study", "results", "analysis", "new", "novel"
}

UTC = dt.timezone.utc

def _now_utc() -> dt.datetime:
    """Return current UTC datetime."""
    return dt.datetime.now(UTC)

def _submitted_range(days: int) -> Tuple[str, str]:
    """Build arXiv submittedDate [FROM TO] window (YYYYMMDDHHMM UTC)."""
    to_dt = _now_utc()
    from_dt = to_dt - dt.timedelta(days=days)
    fmt = "%Y%m%d%H%M"
    return from_dt.strftime(fmt), to_dt.strftime(fmt)

def _recency_score(date_iso: Optional[str], half_life_days: int = 60) -> float:
    """Compute recency prior via exponential decay (0...1)."""
    if not date_iso:
        return 0.0
    try:
        d = dt.date.fromisoformat(date_iso[:10])
    except Exception:
        return 0.0
    days_old = (dt.date.today() - d).days
    return math.exp(-math.log(2) * max(days_old, 0) / max(half_life_days, 1))

# ---------- Text utils ----------
def _tokens(s: str) -> List[str]:
    """Tokenize, lowercase, strip stopwords/digits, enforce sane lengths, keep unique order."""
    raw = [t for t in re.split(r"\W+", (s or "").strip().lower()) if t]
    out, seen = [], set()
    for t in raw:
        if t in STOPWORDS: 
            continue
        if t.isdigit():
            continue
        if not (1 <= len(t) <= 32):
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _compile_word_patterns(tokens: List[str]) -> List[re.Pattern]:
    """Build case-insensitive word-boundary regex patterns for tokens."""
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tokens]

def _coverage(text: Optional[str], pats: List[re.Pattern]) -> Tuple[float, List[str]]:
    """Return coverage fraction and matched unique tokens in text."""
    if not text:
        return 0.0, []
    hits: List[str] = []
    for p in pats:
        if p.search(text):
            tok = p.pattern.replace(r"\b","")
            tok = re.sub(r"\\(.)", r"\1", tok).strip("^$")
            hits.append(tok.lower())
    uniq = list(dict.fromkeys(hits))
    denom = max(len(pats), 1)
    return len(uniq)/denom, uniq

# ---------- Query building ----------
def _and_field(tokens: List[str], field: str = "all") -> str:
    """Join tokens into 'field:t1 AND field:t2 ...'."""
    return " AND ".join([f"{field}:{t}" for t in tokens])

def _or_field(tokens: List[str], field: str = "all") -> str:
    """Join tokens into 'field:t1 OR field:t2 ...'."""
    return " OR ".join([f"{field}:{t}" for t in tokens])

def build_initial_queries(topic: str, days: int) -> List[str]:
    """Build three recall-first queries: AND tokens, OR tokens, phrase in ti/abs; all with date window."""
    toks = _tokens(topic) or ["universal", "search"]
    frm, to = _submitted_range(days)
    date_clause = f"submittedDate:[{frm} TO {to}]"
    q1 = f"({_and_field(toks,'all')}) AND {date_clause}"
    q2 = f"({_or_field(toks,'all')}) AND {date_clause}"
    phrase = " ".join(toks)
    q3 = f'(ti:"{phrase}" OR abs:"{phrase}") AND {date_clause}'
    out: List[str] = []
    for q in (q1, q2, q3):
        if q not in out:
            out.append(q)
    return out

def build_prf_query(base_tokens: List[str], prf_terms: List[str], days: int) -> Optional[str]:
    """Build one PRF-augmented query: (AND base) AND (OR PRF) AND date."""
    if not prf_terms:
        return None
    frm, to = _submitted_range(days)
    date_clause = f"submittedDate:[{frm} TO {to}]"
    core = _and_field(base_tokens, "all") if base_tokens else ""
    exp  = _or_field(prf_terms[:PRF_TOP_TERMS], "all")
    if core:
        return f"({core}) AND ({exp}) AND {date_clause}"
    return f"({exp}) AND {date_clause}"

# ---------- HTTP ----------
def _fetch_raw(q: str, start: int, max_results: int, debug: bool) -> Optional[str]:
    """Call arXiv Atom API for a query page and return raw XML text."""
    params = {
        "search_query": q,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    if debug:
        print("[DEBUG] arXiv request:", url)
    req = Request(url, headers={"User-Agent": HEADER})
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if debug:
                status = getattr(resp, "status", None) or getattr(resp, "code", None) or 200
                print("[DEBUG] arXiv response:", f"status={status}", f"bytes={len(raw)}")
            return raw
    except HTTPError as e:
        if debug: print("[DEBUG] HTTPError:", e.code, e.reason)
        return None
    except URLError as e:
        if debug: print("[DEBUG] URLError:", getattr(e, "reason", str(e)))
        return None

# ---------- Parse & normalize ----------
def _entry_dates(e: feedparser.FeedParserDict) -> Tuple[Optional[str], Optional[str]]:
    """Extract ISO dates from feed entry."""
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
    """Extract arXiv abs/pdf links and collect other link variants."""
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
    """Collect authors with optional affiliations from entry."""
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

def _categories(e: feedparser.FeedParserDict) -> Tuple[List[str], Optional[str]]:
    """Extract categories and primary_category from entry."""
    cats = [t.get("term") for t in (e.get("tags") or []) if t.get("term")]
    primary = None
    try:
        primary = e.tags[0]["term"] if e.get("tags") else None
    except Exception:
        pass
    if hasattr(e, "arxiv_primary_category"):
        try:
            primary = e.arxiv_primary_category.get("term", primary)
        except Exception:
            pass
    return cats, primary

def _first(e: feedparser.FeedParserDict, key: str) -> Optional[str]:
    """Return first/simple value for key if present."""
    val = getattr(e, key, None)
    if isinstance(val, list):
        return val[0] if val else None
    return val

def _normalize(e: feedparser.FeedParserDict) -> Dict[str, Any]:
    """Normalize a feed entry into a consistent paper dict."""
    abs_url, pdf_url, other_links = _abs_pdf_other(e)
    cats, primary = _categories(e)
    published_iso, updated_iso = _entry_dates(e)
    doi = _first(e, "arxiv_doi") or _first(e, "doi")
    doi_url = f"https://doi.org/{doi}" if doi else None
    journal_ref = _first(e, "arxiv_journal_ref") or _first(e, "journal_ref")
    report_no = _first(e, "arxiv_report_no") or _first(e, "arxiv_report_number") or _first(e, "report-no")
    authors = _authors(e)

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
        "authors": authors,
        "links": {"abs": abs_url, "pdf": pdf_url, "other": other_links},
        "categories": cats,
        "primary_category": primary,
        "doi": doi,
        "doi_url": doi_url,
        "journal_ref": journal_ref,
        "report_number": report_no,
    }

# ---------- Retrieval ----------
def fetch_query(q: str, want: int, debug: bool) -> List[Dict[str, Any]]:
    """Fetch up to "want" normalized entries for a single query."""
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
        total_str = getattr(feed.feed, "opensearch_totalresults", None) or getattr(feed.feed, "opensearch_totalResults", None)
        items_per_page = getattr(feed.feed, "opensearch_itemsperpage", None) or getattr(feed.feed, "opensearch_itemsPerPage", None)
        start_index = getattr(feed.feed, "opensearch_startindex", None) or getattr(feed.feed, "opensearch_startIndex", None)
        try:
            per_page = int(items_per_page) if items_per_page else len(feed.entries)
            start = (int(start_index) if start_index else start) + per_page
        except Exception:
            start += len(feed.entries)
        if len(collected) < want:
            time.sleep(SLEEP_BETWEEN_CALLS)
    return collected

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool) -> List[Dict[str, Any]]:
    """Run AND/OR/phrase queries, merge unique results until pool_target is reached."""
    qs = build_initial_queries(topic, days)
    if debug:
        print("[DEBUG] initial queries:")
        for q in qs: print("   ", q)
    seen_ids = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(50, min(PER_CALL, pool_target))
    for qi, q in enumerate(qs, 1):
        chunk = fetch_query(q, want=per_try, debug=debug)
        added = 0
        for r in chunk:
            key = r.get("arxiv_id") or (r.get("title", "").strip().lower())
            if key and key not in seen_ids:
                seen_ids.add(key)
                pool.append(r); added += 1
        if debug:
            print(f"[DEBUG] Q{qi} -> got={len(chunk)}  added_unique={added}  pool={len(pool)}")
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------- PRF (pseudo-relevance feedback) ----------
def _doc_tokens(entry: Dict[str, Any]) -> List[str]:
    """Tokenize title+summary for PRF term extraction."""
    text = f"{entry.get('title','')} {entry.get('summary','')}"
    return [t for t in _tokens(text) if PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN]

def prf_terms(candidates: List[Dict[str, Any]], base_tokens: List[str], k_docs: int = PRF_TOP_DOCS, top_terms: int = PRF_TOP_TERMS) -> List[str]:
    """Extract PRF terms via simple TF-IDF from top-k candidate docs."""
    docs = [_doc_tokens(e) for e in candidates[:k_docs]]
    if not docs:
        return []
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    N = len(docs)
    base_set = set(base_tokens)
    scores = defaultdict(float)
    for d in docs:
        tf = Counter(d)
        for t, f in tf.items():
            if t in base_set:
                continue
            if t.isdigit():
                continue
            if len(t) < PRF_MIN_LEN or len(t) > PRF_MAX_LEN:
                continue
            idf = math.log((N + 1) / (1 + df[t])) + 1.0
            scores[t] += (f / max(len(d), 1)) * idf
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:top_terms]]

# ---------- Scoring & selection ----------
def score_entry(entry: Dict[str, Any], base_pats: List[re.Pattern], prf_pats: List[re.Pattern]) -> Dict[str, Any]:
    """Compute coverage-based confidence and attach match breakdown."""
    title = entry.get("title") or ""
    abstract = entry.get("summary") or ""
    tb, th_b = _coverage(title, base_pats)
    ab, ah_b = _coverage(abstract, base_pats)
    tp, th_p = _coverage(title, prf_pats) if prf_pats else (0.0, [])
    ap, ah_p = _coverage(abstract, prf_pats) if prf_pats else (0.0, [])
    rec = _recency_score(entry.get("updated") or entry.get("published"))
    conf = W_TITLE_BASE*tb + W_ABS_BASE*ab + W_TITLE_PRF*tp + W_ABS_PRF*ap + W_RECENCY*rec
    entry["match"] = {
        "base": {"title_hits": th_b, "abstract_hits": ah_b, "title_coverage": round(tb,4), "abstract_coverage": round(ab,4)},
        "prf":  {"title_hits": th_p, "abstract_hits": ah_p, "title_coverage": round(tp,4), "abstract_coverage": round(ap,4)}
    }
    entry["confidence"] = round(conf, 6)
    return entry

def dedupe(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by arxiv_id (fallback: lowercased title)."""
    seen = set();
    out = []
    for e in entries:
        key = e.get("arxiv_id") or (e.get("title","").strip().lower())
        if key and key not in seen:
            seen.add(key); out.append(e)
    return out

# ---------- Orchestrator ----------
def arxiv_search(topic: str, days: int = 30, max_results: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
    """Run full pipeline: retrieve > PRF expand > retrieve > score > rank > top-K."""
    base_tokens = _tokens(topic)
    base_pats = _compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 100)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)
    if debug:
        print(f"[DEBUG] initial pool: {len(pool)}")

    prf = prf_terms(pool, base_tokens, k_docs=min(PRF_TOP_DOCS, len(pool)), top_terms=PRF_TOP_TERMS)
    prf = [t for t in prf if t not in base_tokens]
    if debug:
        print("[DEBUG] PRF terms:", prf)

    if prf:
        q_prf = build_prf_query(base_tokens, prf, days)
        if debug and q_prf:
            print("[DEBUG] PRF query:", q_prf)
        if q_prf:
            more = fetch_query(q_prf, want=pool_target, debug=debug)
            before = len(pool)
            pool.extend(more)
            pool = dedupe(pool)
            if debug:
                print(f"[DEBUG] PRF retrieve added: {len(pool) - before}, pool now: {len(pool)}")

    prf_pats = _compile_word_patterns(prf) if prf else []
    scored: List[Dict[str, Any]] = []
    for e in pool:
        e = score_entry(e, base_pats, prf_pats)
        ok = (
            e["match"]["base"]["title_hits"] or e["match"]["base"]["abstract_hits"] or
            e["match"]["prf"]["title_hits"]  or e["match"]["prf"]["abstract_hits"]
        )
        if ok:
            scored.append(e)

    scored.sort(key=lambda x: (x.get("confidence", 0.0), x.get("updated") or "", x.get("published") or ""), reverse=True)
    return scored[:max_results]

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal arXiv search with PRF and lexical scoring")
    parser.add_argument("topic", type=str, help="Search topic")
    parser.add_argument("--days", type=int, default=30, help="Date window in days (e.g., 7, 30, 180, 365)")
    parser.add_argument("--max", type=int, default=50, help="Max results to return after ranking")
    parser.add_argument("--debug", action="store_true", help="Print internal debug info")
    args = parser.parse_args()

    results = arxiv_search(
        args.topic,
        days=args.days,
        max_results=args.max,
        debug=args.debug
    )

    print(f"Found: {len(results)}")
    for i, p in enumerate(results[:10], 1):
        conf = p.get("confidence", 0.0)
        date_show = p.get("updated") or p.get("published") or ""
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={conf:.3f}]  [{date_show}]")
        print(f"    abs: {p['links'].get('abs')}")
        if p['links'].get('pdf'):
            print(f"    pdf: {p['links'].get('pdf')}")

    with open("arxiv_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: arxiv_sample.json")
