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
  python crossref_parser.py "computer vision industrial automation" --days 30 --max 10 --debug
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

import httpx
from dateutil import parser as dateparse
from bs4 import BeautifulSoup  # only for light abstract cleanup; optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Config ----------------
CROSSREF_API = "https://api.crossref.org/works"
MAILTO = os.getenv("EMAIL", "").strip()

HEADER = f"topic-radar/1.0 (Crossref client; mailto:{MAILTO})" if MAILTO else "topic-radar/1.0 (Crossref client)"
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

STOPWORDS = {
    "and","or","for","the","a","an","of","in","on","to","with","by","at","from","into",
    "via","towards","toward","study","new","novel"
}

# ---------------- Time helpers ----------------
UTC = dt.timezone.utc

def _now_utc() -> dt.datetime:
    """Current UTC datetime."""
    return dt.datetime.now(UTC)

def _date_window(days: int) -> Tuple[str, str]:
    """Return ISO dates for the last N days."""
    end = _now_utc().date()
    start = end - dt.timedelta(days=days)
    return start.isoformat(), end.isoformat()

def _first_date(parts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract ISO date from Crossref date-parts object."""
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

def _recency_score(date_iso: Optional[str], half_life_days: int = 60) -> float:
    """Recency prior via exponential half-life (0...1)."""
    if not date_iso:
        return 0.0
    try:
        d = dt.date.fromisoformat(date_iso[:10])
    except Exception:
        return 0.0
    days_old = (dt.date.today() - d).days
    return math.exp(-math.log(2) * max(days_old, 0) / max(half_life_days, 1))

# ---------------- Text utils ----------------
def _tokens(s: str) -> List[str]:
    """Lowercase tokenization, drop stopwords/digits, keep order."""
    raw = [t for t in re.split(r"\W+", (s or "").lower()) if t]
    out, seen = [], set()
    for t in raw:
        if t in STOPWORDS or t.isdigit() or not (1 <= len(t) <= 32):
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _compile_word_patterns(tokens: List[str]) -> List[re.Pattern]:
    """Regex for exact token matches (case-insensitive word boundaries)."""
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tokens]

def _coverage(text: Optional[str], pats: List[re.Pattern]) -> Tuple[float, List[str]]:
    """Fraction of tokens matched + unique hits."""
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

def _strip_html(text: Optional[str]) -> Optional[str]:
    """Strip tags."""
    if not text:
        return text
    try:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    except Exception:
        return text

# ---------------- PRF ----------------
def _doc_tokens(title: Optional[str], abstract: Optional[str]) -> List[str]:
    """Tokens for PRF from title+abstract."""
    text = f"{title or ''} {abstract or ''}"
    return [t for t in _tokens(text) if PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN]

def prf_terms(cands: List[Dict[str, Any]], base_tokens: List[str],
              k_docs: int = PRF_TOP_DOCS, top_terms: int = PRF_TOP_TERMS) -> List[str]:
    """Extract PRF terms via simple TF-IDF from top-k candidate docs."""
    docs = [_doc_tokens(c.get("title"), c.get("abstract")) for c in cands[:k_docs]]
    if not docs:
        return []
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    N = len(docs)
    base = set(base_tokens)
    scores = defaultdict(float)
    for d in docs:
        tf = Counter(d)
        for t, f in tf.items():
            if t in base or t.isdigit() or not (PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN):
                continue
            idf = math.log((N + 1) / (1 + df[t])) + 1.0
            scores[t] += (f / max(len(d), 1)) * idf
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:top_terms]]

# ---------------- HTTP ----------------
def _client() -> httpx.Client:
    """Configured HTTPX client with polite headers and timeouts."""
    headers = {"User-Agent": HEADER}
    return httpx.Client(headers=headers, timeout=TIMEOUT, follow_redirects=True)

def _get(params: Dict[str, Any], debug: bool=False) -> Dict[str, Any]:
    """GET /works with params; handle rate limits/backoff; return parsed JSON."""
    if MAILTO:
        params = {**params, "mailto": MAILTO}
    tries = 0
    while True:
        tries += 1
        with _client() as c:
            if debug:
                print("[DEBUG] GET", CROSSREF_API, "params=", params)
            r = c.get(CROSSREF_API, params=params)
            if debug:
                print("[DEBUG] status", r.status_code, "bytes", len(r.content))
            if r.status_code == 429:
                # Backoff and retry
                wait = min(5 * tries, 30)
                if debug:
                    print(f"[DEBUG] 429 Too Many Requests. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()

# ---------------- Normalization ----------------
def _best_title(item: Dict[str, Any]) -> Optional[str]:
    t = item.get("title") or []
    if isinstance(t, list) and t:
        return str(t[0]).strip() or None
    if isinstance(t, str):
        return t.strip() or None
    return None

def _best_venue(item: Dict[str, Any]) -> Optional[str]:
    ct = item.get("container-title") or []
    if isinstance(ct, list) and ct:
        return str(ct[0]).strip() or None
    if isinstance(ct, str):
        return ct.strip() or None
    return None

def _best_pub_date(item: Dict[str, Any]) -> Optional[str]:
    """Prefer published-online/print/issued; fallback to created/indexed."""
    for key in ("published-online", "published-print", "issued"):
        iso = _first_date(item.get(key))
        if iso:
            return iso
    # fallbacks
    for key in ("created", "indexed"):
        try:
            dtxt = item[key]["date-time"]
            return dateparse.parse(dtxt).date().isoformat()
        except Exception:
            continue
    return None

def _updated_date(item: Dict[str, Any]) -> Optional[str]:
    """Use 'indexed' as a proxy for update time if present."""
    try:
        dtxt = item["indexed"]["date-time"]
        return dateparse.parse(dtxt).date().isoformat()
    except Exception:
        return _best_pub_date(item)

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

def _abstract_text(item: Dict[str, Any]) -> Optional[str]:
    abs_raw = item.get("abstract")
    return _strip_html(abs_raw) if abs_raw else None

def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Crossref item into a consistent paper dict."""
    title = _best_title(item)
    abstract = _abstract_text(item)
    pub = _best_pub_date(item)
    upd = _updated_date(item)
    venue = _best_venue(item)
    links = _links(item)
    return {
        "source": "Crossref",
        "doi": item.get("DOI"),
        "title": title,
        "abstract": abstract,
        "published": pub,
        "updated": upd,
        "venue": venue,
        "authors": _authors(item),
        "links": links,
        "is_referenced_by_count": item.get("is-referenced-by-count"),
        "type": item.get("type"),
    }

# ---------------- Query building ----------------
def _and_tokens(tokens: List[str]) -> str:
    """Crossref query string where tokens are space-joined (interpreted as AND-like)."""
    return " ".join(tokens)

def _or_tokens(tokens: List[str]) -> str:
    """OR-like using spaces + explicit OR to bias matching."""
    return " OR ".join(tokens)

def build_initial_queries(topic: str) -> List[Dict[str, Any]]:
    """
    Build three request parameter sets:
      1) query.title = AND tokens
      2) query       = OR tokens
      3) query.bibliographic = "full phrase"
    Date filter is added at call-site to keep this pure.
    """
    toks = _tokens(topic) or ["universal", "search"]
    phrase = " ".join(toks)
    return [
        {"query.title": _and_tokens(toks)},
        {"query": _or_tokens(toks)},
        {"query.bibliographic": f"\"{phrase}\""},
    ]

def _filter_for_window(days: int) -> str:
    """Crossref filter string for publication date window."""
    frm, to = _date_window(days)
    return f"from-pub-date:{frm},until-pub-date:{to}"

# ---------------- Retrieval ----------------
def fetch_query(params_core: Dict[str, Any], days: int, want: int, debug: bool) -> List[Dict[str, Any]]:
    """
    Fetch up to 'want' normalized items for a single parameter set.
    Uses cursor paging politely.
    """
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
        if debug:
            total = msg.get("total-results")
            print(f"[DEBUG] page={pages} items={len(items)} total={total} rows={rows}")
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
    """Run title/OR/bibliographic strategies, merge unique until pool_target."""
    qs = build_initial_queries(topic)
    if debug:
        print("[DEBUG] initial queries (Crossref):")
        for q in qs: print("   ", q)

    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(60, min(PER_PAGE, pool_target))

    for qi, core in enumerate(qs, 1):
        chunk = fetch_query(core, days=days, want=per_try, debug=debug)
        added = 0
        for r in chunk:
            key = r.get("doi") or (r.get("links") or {}).get("abs") or (r.get("title","").strip().lower())
            if key and key not in seen:
                seen.add(key); pool.append(r); added += 1
        if debug:
            print(f"[DEBUG] Q{qi} -> got={len(chunk)} added_unique={added} pool={len(pool)}")
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------------- Scoring & selection ----------------
def score_entry(entry: Dict[str, Any], base_pats: List[re.Pattern], prf_pats: List[re.Pattern]) -> Dict[str, Any]:
    """Compute coverage-based confidence and attach match breakdown."""
    title = entry.get("title") or ""
    abstract = entry.get("abstract") or ""
    tb, th_b = _coverage(title, base_pats)
    ab, ah_b = _coverage(abstract, base_pats)
    tp, th_p = _coverage(title, prf_pats) if prf_pats else (0.0, [])
    ap, ah_p = _coverage(abstract, prf_pats) if prf_pats else (0.0, [])
    # prefer published; fallback to updated
    rec = _recency_score(entry.get("published") or entry.get("updated"))
    conf = W_TITLE_BASE*tb + W_ABS_BASE*ab + W_TITLE_PRF*tp + W_ABS_PRF*ap + W_RECENCY*rec
    entry["match"] = {
        "base": {"title_hits": th_b, "abstract_hits": ah_b, "title_coverage": round(tb,4), "abstract_coverage": round(ab,4)},
        "prf":  {"title_hits": th_p, "abstract_hits": ah_p, "title_coverage": round(tp,4), "abstract_coverage": round(ap,4)}
    }
    entry["confidence"] = round(conf, 6)
    return entry

def dedupe(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by DOI / URL / lowercased title."""
    seen = set(); out = []
    for e in entries:
        key = e.get("doi") or (e.get("links") or {}).get("abs") or (e.get("title","").strip().lower())
        if key and key not in seen:
            seen.add(key); out.append(e)
    return out

# ---------------- Orchestrator ----------------
def crossref_search(topic: str, days: int = 30, max_results: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Run pipeline on Crossref: retrieve → PRF expand → retrieve → score → rank → top-K.
    """
    base_tokens = _tokens(topic)
    base_pats = _compile_word_patterns(base_tokens)

    # Initial pool
    pool_target = max(5 * max_results, 150)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)
    if debug:
        print(f"[DEBUG] initial pool: {len(pool)}")

    # PRF expansion & re-retrieve (query = OR of PRF terms)
    prf = prf_terms(pool, base_tokens, k_docs=min(PRF_TOP_DOCS, len(pool)), top_terms=PRF_TOP_TERMS)
    prf = [t for t in prf if t not in base_tokens]
    if debug:
        print("[DEBUG] PRF terms:", prf)

    if prf:
        core = {"query": " OR ".join(prf[:PRF_TOP_TERMS])}
        more = fetch_query(core, days=days, want=pool_target, debug=debug)
        before = len(pool)
        pool.extend(more)
        pool = dedupe(pool)
        if debug:
            print(f"[DEBUG] PRF retrieve added: {len(pool) - before}, pool now: {len(pool)}")

    # Score + keep entries with any hit
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

    scored.sort(key=lambda x: (x.get("confidence", 0.0), x.get("published") or "", x.get("updated") or ""), reverse=True)
    return scored[:max_results]

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossref works search with PRF and lexical scoring")
    parser.add_argument("topic", type=str, help="Search topic")
    parser.add_argument("--days", type=int, default=30, help="Publication date window in days (e.g., 7, 30, 180, 365)")
    parser.add_argument("--max", type=int, default=50, help="Max results to return after ranking")
    parser.add_argument("--debug", action="store_true", help="Print internal debug info")
    args = parser.parse_args()

    results = crossref_search(
        args.topic,
        days=args.days,
        max_results=args.max,
        debug=args.debug
    )

    print(f"Found: {len(results)}")
    for i, p in enumerate(results[:10], 1):
        conf = p.get("confidence", 0.0)
        date_show = p.get("published") or p.get("updated") or ""
        venue = p.get("venue") or ""
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={conf:.3f}]  [{date_show}]  ({venue})")
        print(f"    abs: {(p.get('links') or {}).get('abs')}")
        if (p.get('links') or {}).get('pdf'):
            print(f"    pdf: {(p['links']).get('pdf')}")
        if (p.get('links') or {}).get('doi'):
            print(f"    doi: {(p['links']).get('doi')}")

    with open("crossref_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: crossref_sample.json")
