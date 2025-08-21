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
  python news_parser.py "computer vision industrial automation" --days 30 --max 10 --debug
"""

from __future__ import annotations
import argparse
import datetime as dt
import math
import re
import time
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Config ----------------
HEADERS = {
    "User-Agent": f"topic-radar/1.0 (news rss client; contact:{os.getenv('CROSSREF_MAILTO','').strip() or 'n/a'})"
}
TIMEOUT = 25
SLEEP = 0.5   

# Feeds list 
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

# Minimal stopword
STOPWORDS = {
    "and","or","for","the","a","an","of","in","on","to","with","by","at","from","into",
    "via","towards","toward","study","new","novel",
    "how","why","what","when","where"
}

# ---------------- Time helpers ----------------
def now_utc() -> dt.datetime:
    """Return timezone-aware current UTC datetime."""
    return dt.datetime.now(dt.timezone.utc)

def cutoff_date(days: int) -> dt.date:
    """Return cutoff date for a trailing N-day window."""
    return now_utc().date() - dt.timedelta(days=days)

def parse_entry_date(entry: feedparser.FeedParserDict) -> Optional[str]:
    """Extract ISO date from feed entry using published/updated fallbacks."""
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
                if d > now_utc().date():
                    d = d - dt.timedelta(days=1)
                return d.isoformat()
            except Exception:
                continue
    return None

def recency_score(date_iso: Optional[str], half_life_days: int = 14) -> float:
    """Exponential half-life recency prior (0...1)."""
    if not date_iso:
        return 0.0
    try:
        d = dt.date.fromisoformat(date_iso[:10])
    except Exception:
        return 0.0
    days_old = (dt.date.today() - d).days
    return math.exp(-math.log(2) * max(days_old, 0) / max(half_life_days, 1))

# ---------------- Text utils ----------------
def strip_html(text: Optional[str]) -> Optional[str]:
    """Best-effort HTML→text."""
    if not text:
        return text
    try:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    except Exception:
        return text

def tokens(s: str) -> List[str]:
    """Lowercase tokenization, drop stopwords/digits, keep order."""
    raw = [t for t in re.split(r"\W+", (s or "").lower()) if t]
    out, seen = [], set()
    for t in raw:
        if t in STOPWORDS or t.isdigit() or not (1 <= len(t) <= 32):
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def compile_word_patterns(tt: List[str]) -> List[re.Pattern]:
    """Compile word-boundary regex patterns for tokens."""
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tt]

def coverage(text: Optional[str], pats: List[re.Pattern]) -> Tuple[float, List[str]]:
    """Fraction [0..1] of tokens matched + unique matched tokens."""
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

def phrase_in_title(title: Optional[str], base_tokens: List[str]) -> bool:
    """Optional: detect exact phrase occurrence in title (soft signal)."""
    if not title or not base_tokens:
        return False
    phrase = " ".join(base_tokens)
    return re.search(re.escape(phrase), title, re.IGNORECASE) is not None

# ---------------- PRF ----------------
def doc_tokens(title: Optional[str], summary: Optional[str]) -> List[str]:
    """Tokens for PRF from title+summary with length constraints."""
    text = f"{title or ''} {summary or ''}"
    return [t for t in tokens(text) if PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN]

def prf_terms(cands: List[Dict[str, Any]], base_tokens: List[str],
              k_docs: int = PRF_TOP_DOCS, top_terms: int = PRF_TOP_TERMS) -> List[str]:
    """Extract PRF terms via simple TF-IDF from top-k candidate docs."""
    docs = [doc_tokens(c.get("title"), c.get("summary")) for c in cands[:k_docs]]
    if not docs:
        return []
    # DF
    df: Dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    N = len(docs)
    base = set(base_tokens)
    scores: Dict[str, float] = {}
    for d in docs:
        tf: Dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        for t, f in tf.items():
            if t in base or t.isdigit() or not (PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN):
                continue
            idf = math.log((N + 1) / (1 + df.get(t, 0))) + 1.0
            scores[t] = scores.get(t, 0.0) + (f / max(len(d), 1)) * idf
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:top_terms]]

# ---------------- HTTP + Feed fetch ----------------
def http_get_text(url: str, debug: bool=False) -> str:
    """HTTP GET with UA & timeout; return text."""
    if debug:
        print("[DEBUG] GET", url)
    with httpx.Client(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True) as c:
        r = c.get(url)
        if debug:
            print("[DEBUG] status", r.status_code, "bytes", len(r.content))
        r.raise_for_status()
        return r.text

def fetch_feed(url: str, debug: bool=False) -> feedparser.FeedParserDict:
    """Download and parse an RSS/Atom feed."""
    raw = http_get_text(url, debug=debug)
    feed = feedparser.parse(raw)
    if getattr(feed, "bozo", 0) and debug:
        print("[DEBUG] feedparser bozo:", getattr(feed, "bozo_exception", ""))
    time.sleep(SLEEP)
    return feed

# ---------------- Normalization ----------------
def normalize_entry(source: str, e: feedparser.FeedParserDict) -> Dict[str, Any]:
    """Normalize RSS entry into a consistent dict."""
    title = getattr(e, "title", None)
    link  = getattr(e, "link", None)
    # summary/content handling
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

def normalize_url(u: Optional[str]) -> Optional[str]:
    """Drop query/fragment to improve dedup stability."""
    if not u:
        return None
    try:
        p = urlparse(u)
        cleaned = p._replace(query="", fragment="")
        return urlunparse(cleaned)
    except Exception:
        return u

def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by normalized URL or lowercased title."""
    seen = set(); out: List[Dict[str, Any]] = []
    for it in items:
        key = normalize_url(it.get("url")) or (it.get("title","").strip().lower())
        if key and key not in seen:
            seen.add(key); out.append(it)
    return out

# ---------------- Scoring ----------------
def score_item(it: Dict[str, Any], base_pats: List[re.Pattern], prf_pats: List[re.Pattern], base_tokens: List[str]) -> Dict[str, Any]:
    """Compute coverage-based confidence for a news item."""
    t = it.get("title") or ""
    s = it.get("summary") or ""
    tb, th_b = coverage(t, base_pats)
    sb, sh_b = coverage(s, base_pats)
    tp, th_p = coverage(t, prf_pats) if prf_pats else (0.0, [])
    sp, sh_p = coverage(s, prf_pats) if prf_pats else (0.0, [])
    rec = recency_score(it.get("date"))
    conf = W_TITLE_BASE*tb + W_SUMM_BASE*sb + W_TITLE_PRF*tp + W_SUMM_PRF*sp + W_RECENCY*rec
    it["match"] = {
        "base": {"title_hits": th_b, "summary_hits": sh_b, "title_coverage": round(tb,4), "summary_coverage": round(sb,4)},
        "prf":  {"title_hits": th_p, "summary_hits": sh_p, "title_coverage": round(tp,4), "summary_coverage": round(sp,4)},
    }
    it["confidence"] = round(conf, 6)
    return it

# ---------------- Orchestrator ----------------
def news_search(topic: str, days: int = 14, max_results: int = 30, debug: bool=False) -> List[Dict[str, Any]]:
    """Fetch all feeds, filter by date window, PRF-expand, score, rank, top-K."""
    # Base tokens & patterns
    base_tokens = tokens(topic)
    base_pats = compile_word_patterns(base_tokens)

    # Aggregate
    items: List[Dict[str, Any]] = []
    for name, url in FEEDS:
        try:
            feed = fetch_feed(url, debug=debug)
            for e in feed.entries:
                it = normalize_entry(name, e)
                items.append(it)
            if debug:
                print(f"[DEBUG] {name}: +{len(feed.entries)} (pool={len(items)})")
        except Exception as ex:
            if debug:
                print(f"[DEBUG] fetch error {name}:", str(ex))

    # Window filter
    cut = cutoff_date(days)
    items = [it for it in items if (it.get("date") and dt.date.fromisoformat(it["date"]) >= cut)]
    if debug:
        print("[DEBUG] after window filter:", len(items))

    # Dedup
    items = dedupe(items)
    if debug:
        print("[DEBUG] after dedupe:", len(items))

    # PRF terms from top pool (titles+summaries)
    prf = prf_terms(items, base_tokens, k_docs=min(PRF_TOP_DOCS, len(items)), top_terms=PRF_TOP_TERMS)
    prf = [t for t in prf if t not in base_tokens]
    prf_pats = compile_word_patterns(prf) if prf else []
    if debug:
        print("[DEBUG] PRF terms:", prf)

    # Score & keep with any hit (base or prf)
    scored: List[Dict[str, Any]] = []
    for it in items:
        it = score_item(it, base_pats, prf_pats, base_tokens)
        ok = (it["match"]["base"]["title_hits"] or it["match"]["base"]["summary_hits"] or
              it["match"]["prf"]["title_hits"]  or it["match"]["prf"]["summary_hits"])
        if ok:
            scored.append(it)

    # Rank
    scored.sort(key=lambda x: (x.get("confidence", 0.0), x.get("date") or ""), reverse=True)
    return scored[:max_results]

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="News RSS parser with PRF and lexical scoring")
    ap.add_argument("topic", type=str, help="Search topic")
    ap.add_argument("--days", type=int, default=14, help="Past N days window")
    ap.add_argument("--max", type=int, default=30, help="Max results to return")
    ap.add_argument("--debug", action="store_true", help="Verbose debug")
    args = ap.parse_args()

    res = news_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(res)}")
    for i, p in enumerate(res[:10], 1):
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={p.get('confidence',0):.3f}]  [{p.get('date','')}]  ({p.get('source','')})")
        print(f"    url: {p.get('url')}")
    with open("news_sample.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        print("Saved: news_sample.json")
