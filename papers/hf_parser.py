# hf_parser.py
"""
Hugging Face Papers parser.

Fixes:
- Robust date extraction: <date> tag, regex fallback, and page-level fallback.
- Query-first: crawl recent dailies with ?q=..., filter by window.
- Window-first: fetch month/week/day pages to cover window, filter by window.
- Merge + PRF + lexical scoring + recency; rank and return top-K.

Usage:
  python hf_parser.py "computer vision industrial automation" --days 10 --max 10 --debug
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
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Config ----------
BASE = "https://huggingface.co"
HEADERS = {
    "User-Agent": f"topic-radar/1.3 (HF Papers HTML crawler) {os.getenv('PYALEX_EMAIL','')}".strip()
}
TIMEOUT = 30
SLEEP = 0.75
PER_PAGE_LIMIT = 300  

# Scoring weights
W_TITLE_BASE = 0.55
W_ABS_BASE   = 0.20
W_TITLE_PRF  = 0.15
W_ABS_PRF    = 0.05
W_RECENCY    = 0.03
W_PHRASE     = 0.02

# PRF params
PRF_TOP_DOCS  = 30
PRF_TOP_TERMS = 12
PRF_MIN_LEN   = 3
PRF_MAX_LEN   = 24

# Minimal stopwords
STOPWORDS = {
    "and","or","for","the","a","an","of","in","on","to","with","by","at","from","into",
    "via","towards","toward","study","novel"
}

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
    match: Dict[str, Any] = None
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- Time helpers ----------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _daterange(days: int) -> List[dt.date]:
    today = _now_utc().date()
    return [today - dt.timedelta(d) for d in range(days)]

def _parse_card_date_text(txt: str) -> Optional[str]:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        d = dateparse.parse(txt, default=_now_utc())
        dd = d.date()
        if dd > _now_utc().date():
            dd = dt.date(dd.year-1, dd.month, dd.day)
        return dd.isoformat()
    except Exception:
        return None

def _recency_score(date_iso: Optional[str], half_life_days: int = 60) -> float:
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
    raw = [t for t in re.split(r"\W+", (s or "").lower()) if t]
    out, seen = [], set()
    for t in raw:
        if t in STOPWORDS or t.isdigit() or not (1 <= len(t) <= 32):
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _compile_word_patterns(tokens: List[str]) -> List[re.Pattern]:
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tokens]

def _coverage(text: Optional[str], pats: List[re.Pattern]) -> Tuple[float, List[str]]:
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

def _phrase_in_title(title: Optional[str], tokens: List[str]) -> bool:
    if not title or not tokens: return False
    phrase = " ".join(tokens)
    return re.search(re.escape(phrase), title, re.IGNORECASE) is not None

# ---------- PRF ----------
def _doc_tokens(title: Optional[str], summary: Optional[str]) -> List[str]:
    text = f"{title or ''} {summary or ''}"
    return [t for t in _tokens(text) if PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN]

def prf_terms(cands: List[HFItem], base_tokens: List[str],
              k_docs: int = PRF_TOP_DOCS, top_terms: int = PRF_TOP_TERMS) -> List[str]:
    docs = [_doc_tokens(c.title, c.summary) for c in cands[:k_docs]]
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

# ---------- HTTP ----------
def _client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)

def _get(url: str, debug: bool=False) -> str:
    if debug: print("[DEBUG] GET", url)
    with _client() as c:
        r = c.get(url)
        if debug: print("[DEBUG] status", r.status_code, "bytes", len(r.content))
        r.raise_for_status()
        return r.text

# ---------- URL builders ----------
def _url_day(d: dt.date, q: Optional[str] = None) -> str:
    u = f"{BASE}/papers/date/{d.isoformat()}"
    return f"{u}?{urlencode({'q': q})}" if q else u

def _url_week(iso_year: int, iso_week: int) -> str:
    return f"{BASE}/papers/week/{iso_year}-W{iso_week:02d}"

def _url_month(year: int, month: int) -> str:
    return f"{BASE}/papers/month/{year}-{month:02d}"

# ---------- Parsing ----------
DATE_REGEX = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE
)

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

        # Authors line
        parent_txt = art.get_text(" ", strip=True)
        m = re.search(r"(\d+\s*authors?)", parent_txt, re.IGNORECASE)
        authors = m.group(1) if m else None

        # arXiv/PDF links from slug 
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
def fetch_day(d: dt.date, q: Optional[str] = None, debug: bool=False) -> List[HFItem]:
    html = _get(_url_day(d, q=q), debug=debug)
    time.sleep(SLEEP)
    return _parse_cards(html, fallback_date=d)

def fetch_week(d: dt.date, debug: bool=False) -> List[HFItem]:
    iso_year, iso_week, _ = d.isocalendar()
    html = _get(_url_week(iso_year, iso_week), debug=debug)
    time.sleep(SLEEP)
    monday = dt.date.fromisocalendar(iso_year, iso_week, 1)
    return _parse_cards(html, fallback_date=monday)

def fetch_month(d: dt.date, debug: bool=False) -> List[HFItem]:
    html = _get(_url_month(d.year, d.month), debug=debug)
    time.sleep(SLEEP)
    fallback = dt.date(d.year, d.month, 15)
    return _parse_cards(html, fallback_date=fallback)

# ---------- Scoring ----------
def score_item(it: HFItem, base_pats: List[re.Pattern], prf_pats: List[re.Pattern], base_tokens: List[str]) -> HFItem:
    tb, th_b = _coverage(it.title,   base_pats)
    ab, ah_b = _coverage(it.summary, base_pats)
    tp, th_p = _coverage(it.title,   prf_pats) if prf_pats else (0.0, [])
    ap, ah_p = _coverage(it.summary, prf_pats) if prf_pats else (0.0, [])
    rec = _recency_score(it.date)
    phrase_bonus = 1.0 if _phrase_in_title(it.title, base_tokens) else 0.0
    conf = (W_TITLE_BASE*tb + W_ABS_BASE*ab + W_TITLE_PRF*tp + W_ABS_PRF*ap + W_RECENCY*rec + W_PHRASE*phrase_bonus)
    it.confidence = round(conf, 6)
    it.match = {
        "base": {"title_hits": th_b, "abstract_hits": ah_b, "title_coverage": round(tb,4), "abstract_coverage": round(ab,4)},
        "prf":  {"title_hits": th_p, "abstract_hits": ah_p, "title_coverage": round(tp,4), "abstract_coverage": round(ap,4)}
    }
    return it

def dedupe(items: List[HFItem]) -> List[HFItem]:
    seen = set(); out = []
    for it in items:
        key = (it.links.get("arxiv") or it.hf_slug or (it.title or "").strip().lower())
        if key and key not in seen:
            seen.add(key); out.append(it)
    return out

# ---------- query-first ----------
def crawl_recent_query_pool(topic: str, days_back: int, pool_limit: int, debug: bool=False) -> List[HFItem]:
    tokens = _tokens(topic)
    q = " ".join(tokens) if tokens else topic.strip()
    pool: List[HFItem] = []
    for d in _daterange(days_back):
        try:
            items = fetch_day(d, q=q, debug=debug)
        except Exception as e:
            if debug: print("[DEBUG] fetch_day(query) error", d, str(e))
            items = []
        if items:
            pool.extend(items)
            if debug: print(f"[DEBUG] query-day {d}: +{len(items)}  pool={len(pool)}")
        if len(pool) >= pool_limit:
            break
    return dedupe(pool)[:pool_limit]

def run_query_first(topic: str, days: int, max_results: int, debug: bool=False) -> List[HFItem]:
    days_back = max(days, 60)
    pool_limit = max(2*max_results, 100)
    pool = crawl_recent_query_pool(topic, days_back=days_back, pool_limit=pool_limit, debug=debug)
    cutoff = _now_utc().date() - dt.timedelta(days=days)
    return [it for it in pool if (it.date and dt.date.fromisoformat(it.date) >= cutoff)]

# ---------- window-first ----------
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
    today = _now_utc().date()
    start = today - dt.timedelta(days=days)
    pool: List[HFItem] = []

    if days > 31:
        for y, m in _months_in_range(start, today):
            try:
                items = fetch_month(dt.date(y, m, 15), debug=debug)
            except Exception as e:
                if debug: print("[DEBUG] fetch_month error", y, m, str(e)); items = []
            pool.extend(items)
            if debug: print(f"[DEBUG] month {y}-{m:02d}: +{len(items)}  pool={len(pool)}")
    elif days > 7:
        for iy, iw in _weeks_in_range(start, today):
            try:
                monday = dt.date.fromisocalendar(iy, iw, 1)
                items = fetch_week(monday, debug=debug)
            except Exception as e:
                if debug: print("[DEBUG] fetch_week error", iy, iw, str(e)); items = []
            pool.extend(items)
            if debug: print(f"[DEBUG] week {iy}-W{iw:02d}: +{len(items)}  pool={len(pool)}")
    else:
        for d in _daterange(days):
            try:
                items = fetch_day(d, q=None, debug=debug)
            except Exception as e:
                if debug: print("[DEBUG] fetch_day error", d, str(e)); items = []
            pool.extend(items)
            if debug: print(f"[DEBUG] day {d}: +{len(items)}  pool={len(pool)}")

    pool = dedupe(pool)
    cutoff = _now_utc().date() - dt.timedelta(days=days)
    return [it for it in pool if (it.date and dt.date.fromisoformat(it.date) >= cutoff)]

# ---------- Combined orchestrator ----------
def hf_search(topic: str, days: int = 30, max_results: int = 50, debug: bool=False) -> List[Dict[str, Any]]:
    base_tokens = _tokens(topic)
    base_pats = _compile_word_patterns(base_tokens)

    q_pool = run_query_first(topic, days=days, max_results=max_results, debug=debug)
    if debug: print("[DEBUG] query-first pool:", len(q_pool))

    w_pool = fetch_window(days=days, debug=debug)
    if debug: print("[DEBUG] window-first pool:", len(w_pool))

    merged = dedupe(q_pool + w_pool)
    if debug: print("[DEBUG] merged unique:", len(merged))

    prf = prf_terms(merged, base_tokens, k_docs=min(PRF_TOP_DOCS, len(merged)), top_terms=PRF_TOP_TERMS)
    prf = [t for t in prf if t not in base_tokens]
    prf_pats = _compile_word_patterns(prf) if prf else []
    if debug: print("[DEBUG] PRF terms:", prf)

    scored: List[HFItem] = []
    for it in merged:
        it = score_item(it, base_pats, prf_pats, base_tokens)
        ok = (it.match["base"]["title_hits"] or it.match["base"]["abstract_hits"] or
              it.match["prf"]["title_hits"]  or it.match["prf"]["abstract_hits"])
        if ok:
            scored.append(it)

    scored.sort(key=lambda x: (x.confidence, x.date or ""), reverse=True)
    return [it.to_dict() for it in scored[:max_results]]

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HF Papers parser (combined strategies, robust dates)")
    ap.add_argument("topic", type=str, help="Search topic")
    ap.add_argument("--days", type=int, default=30, help="Past N days window")
    ap.add_argument("--max", type=int, default=50, help="Max results after ranking")
    ap.add_argument("--debug", action="store_true", help="Verbose debug")
    args = ap.parse_args()

    res = hf_search(
        args.topic,
        days=args.days,
        max_results=args.max,
        debug=args.debug
    )

    print(f"Found: {len(res)}")
    for i, p in enumerate(res[:10], 1):
        print(f"{i:>2}. {p.get('title','')[:120]}  [conf={p.get('confidence',0):.3f}]  [{p.get('date','')}]")
        print(f"    HF:  {p['links'].get('abs')}")
        if p['links'].get('arxiv'):
            print(f"    abs: {p['links']['arxiv']}")
        if p['links'].get('pdf'):
            print(f"    pdf: {p['links']['pdf']}")
    with open("hf_sample.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        print("Saved: hf_sample.json")
