"""
Universal OpenAlex (via PyAlex).

Features:
- Multi-strategy retrieval (AND tokens => OR tokens => phrase)
- Date window by publication date (last N days)
- Pseudo-Relevance Feedback (PRF) from top docs (TF-IDF)
- Coverage-based scoring (title > abstract) + recency prior
- Dedupe, rank, JSON export

Usage:
  pip install pyalex python-dotenv
  set PYALEX_EMAIL=you@domain.com   # (Windows PowerShell)
  export PYALEX_EMAIL=you@domain.com # (macOS/Linux)
  python openalex_parser.py "computer vision industrial automation" --days 30 --max 10 --debug
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- PyAlex config ---
PYALEX_EMAIL = os.getenv("EMAIL", "").strip()

try:
    import pyalex
    if PYALEX_EMAIL:
        pyalex.config.email = PYALEX_EMAIL  # enter polite pool
    pyalex.config.max_retries = 2
    pyalex.config.retry_backoff_factor = 0.25
    pyalex.config.retry_http_codes = [429, 500, 503]
    HAVE_PYALEX = True
except Exception:
    HAVE_PYALEX = False

# ---------- Constants ----------
HEADER = f"openalex-parser/1.0 ({PYALEX_EMAIL or 'no-email'})"
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

# Minimal stopword set
STOPWORDS = {
    "and", "or", "for", "the", "a", "an", "of", "in", "on", "to", "with", "by", "at",
    "from", "into", "using", "via", "based", "towards", "toward", "method",
    "study", "results", "analysis", "new", "novel"
}

# ---------- Time helpers ----------
UTC = dt.timezone.utc

def _now_utc() -> dt.datetime:
    """Return current UTC datetime."""
    return dt.datetime.now(UTC)

def _since_iso(days: int) -> str:
    """Return ISO date YYYY-MM-DD for 'days' ago (UTC)."""
    return (_now_utc() - dt.timedelta(days=days)).date().isoformat()

def _recency_score(date_iso: Optional[str], half_life_days: int = 60) -> float:
    """Recency prior in [0..1]"""
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
    """Tokenize, lowercase, drop stopwords/digits, unique-preserving order."""
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
    """Return coverage fraction [0..1] and matched unique tokens in text."""
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

# ---------- Query building for OpenAlex ----------
def build_initial_queries(topic: str) -> List[str]:
    """Build three search strings for OpenAlex: AND, OR, and quoted phrase."""
    toks = _tokens(topic) or ["universal", "search"]
    q1 = " ".join(toks)
    q2 = " OR ".join(toks)
    phrase = " ".join(toks)
    q3 = f"\"{phrase}\""
    out: List[str] = []
    for q in (q1, q2, q3):
        if q not in out:
            out.append(q)
    return out

# ---------- OpenAlex helpers ----------
def _reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstruct abstract text from abstract_inverted_index."""
    if not inv_idx:
        return None
    try:
        max_pos = max(max(pos_list) for pos_list in inv_idx.values())
        arr = [None] * (max_pos + 1)
        for word, pos_list in inv_idx.items():
            for pos in pos_list:
                if 0 <= pos < len(arr) and arr[pos] is None:
                    arr[pos] = word
        tokens = [t if t is not None else "" for t in arr]
        text = " ".join([t for t in tokens if t != ""]).strip()
        return text or None
    except Exception:
        return None

def _authors(work: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    """Extract authors (names + first institution if available) from 'authorships'."""
    out = []
    for a in (work.get("authorships") or []):
        name = (a.get("author") or {}).get("display_name")
        aff  = None
        insts = a.get("institutions") or []
        if insts:
            aff = insts[0].get("display_name")
        out.append({"name": name, "affiliation": aff})
    return out

def _normalize(work: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize OpenAlex 'work' into a consistent dict."""
    title = work.get("display_name") or work.get("title")
    abstract = work.get("abstract")
    if not abstract:
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    pub_date = work.get("publication_date") or work.get("from_publication_date")
    upd_date = work.get("updated_date") or pub_date
    loc = work.get("primary_location") or {}
    abs_url = loc.get("landing_page_url") or (loc.get("source") or {}).get("homepage_url") or work.get("id")
    pdf_url = loc.get("pdf_url")
    biblio = work.get("biblio") or {}
    venue = biblio.get("venue") or ((loc.get("source") or {}).get("display_name"))
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
        "venue": venue,
        "authors": _authors(work),
        "links": {"abs": abs_url, "pdf": pdf_url, "openalex": openalex_id},
        "doi": doi,
        "doi_url": doi_url,
        "cited_by_count": cited_by,
    }

# ---------- Retrieval ----------
def fetch_query_openalex(q: str, since_iso: str, want: int, debug: bool) -> List[Dict[str, Any]]:
    """Fetch up to 'want' normalized works for an OpenAlex search string 'q' with date filter."""
    items: List[Dict[str, Any]] = []
    if not HAVE_PYALEX:
        if debug: print("[DEBUG] PyAlex not available. Install 'pyalex' and set PYALEX_EMAIL.")
        return items

    base = (
        pyalex.Works()
        .search(q)
        .filter(from_publication_date=since_iso)
        .select([
            "id",
            "doi",
            "display_name",
            "publication_date",
            "updated_date",
            "primary_location",
            "authorships",
            "cited_by_count",
            "abstract_inverted_index",
            "biblio"
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
    if debug:
        print(f"[DEBUG] OpenAlex '{q}' -> got={len(items)}")
    return items

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool) -> List[Dict[str, Any]]:
    """Run AND/OR/phrase OpenAlex searches, merge unique until pool_target."""
    qs = build_initial_queries(topic)
    if debug:
        print("[DEBUG] initial queries:")
        for q in qs: print("   ", q)

    since = _since_iso(days)
    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(60, min(PER_PAGE, pool_target))

    for qi, q in enumerate(qs, 1):
        chunk = fetch_query_openalex(q, since_iso=since, want=per_try, debug=debug)
        added = 0
        for r in chunk:
            key = r.get("doi") or r.get("openalex_id") or (r.get("title","").strip().lower())
            if key and key not in seen:
                seen.add(key); pool.append(r); added += 1
        if debug:
            print(f"[DEBUG] Q{qi}-> got={len(chunk)} added_unique={added} pool={len(pool)}")
        if len(pool) >= pool_target:
            return pool[:pool_target]
    return pool[:pool_target]

# ---------- PRF (pseudo-relevance feedback) ----------
def _doc_tokens(entry: Dict[str, Any]) -> List[str]:
    """Tokenize title+summary for PRF term extraction."""
    text = f"{entry.get('title','')} {entry.get('summary','')}"
    return [t for t in _tokens(text) if PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN]

def prf_terms(candidates: List[Dict[str, Any]], base_tokens: List[str],
              k_docs: int = PRF_TOP_DOCS, top_terms: int = PRF_TOP_TERMS) -> List[str]:
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
            if t in base_set or t.isdigit() or not (PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN):
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
    """Remove duplicates by DOI / OpenAlex ID / lowercased title."""
    seen = set(); out = []
    for e in entries:
        key = e.get("doi") or e.get("openalex_id") or (e.get("title","").strip().lower())
        if key and key not in seen:
            seen.add(key); out.append(e)
    return out

# ---------- Orchestrator ----------
def openalex_search(topic: str, days: int = 30, max_results: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
    """Run full pipeline on OpenAlex: retrieve > PRF expand > retrieve > score > rank > top-K."""
    if not HAVE_PYALEX:
        raise RuntimeError("PyAlex is not installed or failed to import")

    base_tokens = _tokens(topic)
    base_pats = _compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 120)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)
    if debug:
        print(f"[DEBUG] initial pool: {len(pool)}")

    prf = prf_terms(pool, base_tokens, k_docs=min(PRF_TOP_DOCS, len(pool)), top_terms=PRF_TOP_TERMS)
    prf = [t for t in prf if t not in base_tokens]
    if debug:
        print("[DEBUG] PRF terms:", prf)

    if prf:
        exp = " OR ".join(prf[:PRF_TOP_TERMS])
        core = " ".join(base_tokens)
        q_prf = f"{core} ({exp})" if core else exp
        more = fetch_query_openalex(q_prf, since_iso=_since_iso(days), want=pool_target, debug=debug)
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
    parser = argparse.ArgumentParser(description="Universal OpenAlex (PyAlex) search with PRF and lexical scoring")
    parser.add_argument("topic", type=str, help="Search topic")
    parser.add_argument("--days", type=int, default=30, help="Publication date window in days (e.g., 7, 30, 180, 365)")
    parser.add_argument("--max", type=int, default=50, help="Max results to return after ranking")
    parser.add_argument("--debug", action="store_true", help="Print internal debug info")
    args = parser.parse_args()

    results = openalex_search(
        args.topic,
        days=args.days,
        max_results=args.max,
        debug=args.debug
    )

    print(f"Found: {len(results)}")
    for i, p in enumerate(results[:10], 1):
        conf = p.get("confidence", 0.0)
        date_show = p.get("updated") or p.get("published") or ""
        venue = p.get("venue") or ""
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={conf:.3f}]  [{date_show}]  ({venue})")
        print(f"    abs: {p['links'].get('abs')}")
        if p['links'].get('pdf'):
            print(f"    pdf: {p['links'].get('pdf')}")
        if p.get("doi_url"):
            print(f"    doi: {p['doi_url']}")

    with open("openalex_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: openalex_sample.json")
