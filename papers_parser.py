# -*- coding: utf-8 -*-
"""
Sources
1) PyAlex (OpenAlex via "pyalex")
   How: Works().search(topic).filter(from_publication_date=..., language="en")
   Tips: set env var PYALEX_EMAIL to enter OpenAlex "polite pool".

2) arXiv API (Atom)
   How: https://export.arxiv.org/api/query?search_query=...&sortBy=lastUpdatedDate
   Notes: use HTTPS and follow redirects; filter by date client-side.

3) Hugging Face /papers (trending)
   How: HTML parse; filter titles by topic tokens (any token match).

Common schema (dict)
--------------------
{ 'title', 'url', 'venue', 'date', 'doi', 'source', 'cited_by', 'score' }

Public API
----------
async def search_papers(topic: str, days: int = 30, limit_total: int = 120, debug: bool = False) -> dict:
    Returns {
        "with_doi": List[Paper],
        "no_doi":   List[Paper],
        "all":      List[Paper],   # merged, ranked
        "debug":    str            # only when debug=True
    }

CLI (quick test)
----------------
python papers_parser.py "computer vision industrial automation" --days 30 --limit 100 --debug
"""

from __future__ import annotations

import os
import re
import math
import json
import asyncio
import logging
import traceback
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import httpx
import feedparser
from bs4 import BeautifulSoup

# Optional: load .env during development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =========================
# Debug / logging utilities
# =========================

logger = logging.getLogger("papers_parser")
LOG_FMT = "[%(levelname)s] %(message)s"

class Debug:
    """Collects debug lines for returning in API or saving in CLI."""
    def __init__(self) -> None:
        self.lines: List[str] = []

    def add(self, msg: str, **kv: Any) -> None:
        if kv:
            try:
                payload = json.dumps(kv, ensure_ascii=False, default=str)
            except Exception:
                payload = str(kv)
            self.lines.append(f"{msg} | {payload}")
        else:
            self.lines.append(msg)

    def text(self) -> str:
        return "\n".join(self.lines)

def setup_logging(debug: bool) -> None:
    logging.basicConfig(level=(logging.DEBUG if debug else logging.INFO), format=LOG_FMT)


# =========================
# Environment / etiquette
# =========================

PYALEX_EMAIL = os.getenv("PYALEX_EMAIL", "").strip()
UA = f"papers-parser/0.4 ({PYALEX_EMAIL})" if PYALEX_EMAIL else "papers-parser/0.4"

# Configure PyAlex if available (polite pool + retries)
try:
    import pyalex  # type: ignore

    if PYALEX_EMAIL:
        pyalex.config.email = PYALEX_EMAIL
    pyalex.config.max_retries = 2
    pyalex.config.retry_backoff_factor = 0.25
    pyalex.config.retry_http_codes = [429, 500, 503]
    HAVE_PYALEX = True
except Exception:
    HAVE_PYALEX = False


# =========================
# Data model & helpers
# =========================

@dataclass
class Paper:
    title: str | None
    url: str | None
    venue: str | None
    date: str | None        # ISO YYYY-MM-DD if available
    doi: str | None
    source: str             # e.g., "OpenAlex (pyalex)", "arXiv", "HF Papers"
    cited_by: int | None = None
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def since_iso(days: int) -> str:
    """UTC-based 'since' date for API filters: YYYY-MM-DD."""
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).date().isoformat()

def parse_date_iso(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None

def recency_score(pub_date: str | None, half_life_days: int = 30) -> float:
    """Exponential decay 0..1; newer = higher."""
    d = parse_date_iso(pub_date)
    if not d:
        return 0.0
    days_old = (dt.date.today() - d).days
    return math.exp(-max(days_old, 0) / max(half_life_days, 1))

def any_token_match(title: str | None, query: str) -> bool:
    """Return True if any token from query appears in title (case-insensitive)."""
    if not title:
        return False
    tokens = [t for t in re.split(r"\W+", query.lower()) if t]
    t = title.lower()
    return any(tok in t for tok in tokens)

def make_httpx(debug: Debug) -> httpx.AsyncClient:
    """HTTPX client with UA and redirects enabled (arXiv often redirects)."""
    debug.add("HTTPX client created", user_agent=UA)
    return httpx.AsyncClient(timeout=30, headers={"User-Agent": UA}, follow_redirects=True)


# =========================
# Source: OpenAlex via PyAlex
# =========================

async def fetch_openalex_pyalex(topic: str, days: int, limit: int, debug: Debug) -> List[Paper]:
    """Primary scholarly source; filter by date, language (en), with fallback."""
    items: List[Paper] = []

    if not HAVE_PYALEX:
        debug.add("PyAlex not available — skipping")
        return items

    try:
        since = since_iso(days)
        debug.add("PyAlex query init", topic=topic, since=since, limit=limit, email=PYALEX_EMAIL or None)

        # First try with language filter (en)
        q = (
            pyalex.Works()
            .search(topic)
            .filter(from_publication_date=since)
            .filter(language="en")
            .select(["id","doi","display_name","primary_location","host_venue","publication_date","cited_by_count"])
        )

        got = 0
        for page in q.paginate(per_page=25):
            for w in page:
                loc = w.get("primary_location") or {}
                url = (
                    loc.get("landing_page_url") or
                    (loc.get("source") or {}).get("homepage_url") or
                    w.get("id")
                )
                items.append(Paper(
                    title=w.get("display_name"),
                    url=url,
                    venue=(w.get("host_venue") or {}).get("display_name"),
                    date=w.get("publication_date"),
                    doi=w.get("doi"),
                    source="OpenAlex (pyalex)",
                    cited_by=int(w.get("cited_by_count") or 0),
                ))
                got += 1
                if got >= limit:
                    break
            if got >= limit:
                break

        debug.add("PyAlex results (language='en')", count=len(items))

        # Fallback: without language filter (some items lack language field)
        if len(items) == 0:
            debug.add("PyAlex retry without language filter")
            q2 = (
                pyalex.Works()
                .search(topic)
                .filter(from_publication_date=since)
                .select(["id","doi","display_name","primary_location","host_venue","publication_date","cited_by_count"])
            )
            got = 0
            for page in q2.paginate(per_page=25):
                for w in page:
                    loc = w.get("primary_location") or {}
                    url = (
                        loc.get("landing_page_url") or
                        (loc.get("source") or {}).get("homepage_url") or
                        w.get("id")
                    )
                    items.append(Paper(
                        title=w.get("display_name"),
                        url=url,
                        venue=(w.get("host_venue") or {}).get("display_name"),
                        date=w.get("publication_date"),
                        doi=w.get("doi"),
                        source="OpenAlex (pyalex, no-lang)",
                        cited_by=int(w.get("cited_by_count") or 0),
                    ))
                    got += 1
                    if got >= limit:
                        break
                if got >= limit:
                    break
            debug.add("PyAlex results (no language)", count=len(items))

    except Exception as e:
        debug.add("PyAlex error", error=str(e), tb=traceback.format_exc())

    return items


# =========================
# Source: arXiv Atom API
# =========================

ARXIV_API = "https://export.arxiv.org/api/query"  # HTTPS is important

async def fetch_arxiv(topic: str, days: int, limit: int, debug: Debug) -> List[Paper]:
    """Fast-moving preprints; query by tokens/phrase; filter by date client-side."""
    items: List[Paper] = []
    try:
        terms = [t for t in re.split(r"\s+", topic.strip()) if t]
        phrase = "+".join(terms)
        # Build a broader query: exact phrase + OR across tokens
        q_parts = [f'all:"{phrase}"'] if phrase else []
        q_parts += [f"all:{t}" for t in terms]
        q = "+OR+".join(q_parts) if q_parts else "all:*"

        params = {
            "search_query": q,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": min(limit, 100),
        }
        from urllib.parse import urlencode
        url = f"{ARXIV_API}?{urlencode(params)}"
        debug.add("arXiv request", url=url)

        async with make_httpx(debug) as c:
            r = await c.get(url)
            debug.add("arXiv response", status=r.status_code, bytes=len(r.content))
            r.raise_for_status()
            fp = feedparser.parse(r.text)
            if getattr(fp, "bozo", 0):
                debug.add("arXiv feed bozo", bozo_exception=str(getattr(fp, "bozo_exception", "")))

        cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
        for e in fp.entries:
            date_str = (getattr(e, "updated", "") or getattr(e, "published", ""))[:10]
            try:
                dt_ = dt.datetime.fromisoformat(date_str) if date_str else None
            except Exception:
                dt_ = None
            if dt_ and dt_ < cutoff:
                continue
            items.append(Paper(
                title=getattr(e, "title", None),
                url=getattr(e, "link", None),
                venue="arXiv",
                date=date_str or None,
                doi=getattr(e, "arxiv_doi", None),
                source="arXiv",
                cited_by=None,
            ))
            if len(items) >= limit:
                break

        debug.add("arXiv parsed entries", count=len(items))

    except Exception as e:
        debug.add("arXiv error", error=str(e), tb=traceback.format_exc())

    return items


# =========================
# Source: Hugging Face /papers
# =========================

HF_PAPERS = "https://huggingface.co/papers"

async def fetch_hf_papers(topic: str, limit: int, debug: Debug) -> List[Paper]:
    """Trending curated list; we do lightweight HTML parsing and token matching."""
    items: List[Paper] = []
    try:
        async with make_httpx(debug) as c:
            r = await c.get(HF_PAPERS)
            debug.add("HF papers response", status=r.status_code, bytes=len(r.content))
            r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        total_cards = 0
        for a in soup.select("a[href*='/papers/']"):
            title = a.get_text(strip=True)
            href = a.get("href")
            if not title or not href:
                continue
            total_cards += 1
            # broader match: any token from the query
            if not any_token_match(title, topic):
                continue
            items.append(Paper(
                title=title,
                url=f"https://huggingface.co{href}",
                venue="HF Papers (trending)",
                date=dt.date.today().isoformat(),  # page is "today's picks"-style
                doi=None,
                source="HF Papers",
                cited_by=None,
            ))
            if len(items) >= limit:
                break

        debug.add("HF papers parsed", total_cards=total_cards, matched=len(items))

    except Exception as e:
        debug.add("HF papers error", error=str(e), tb=traceback.format_exc())

    return items


# =========================
# Ranking / de-duplication
# =========================

def compute_score(p: Paper) -> float:
    """
    Final score = 0.7 * recency + 0.3 * citation_bonus
      - recency: exponential decay (half-life 30 days)
      - citation_bonus: log1p(citations)/10 when available
    """
    r = recency_score(p.date, half_life_days=30)
    c = (math.log1p(p.cited_by or 0) / 10.0) if p.cited_by else 0.0
    return 0.7 * r + 0.3 * c

def dedupe(papers: List[Paper], debug: Debug) -> List[Paper]:
    """Prefer DOI; else normalize by (title_lower, year)."""
    seen = set()
    out: List[Paper] = []
    for p in papers:
        key = p.doi or ((p.title or "").strip().lower(), (p.date or "")[:4])
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    debug.add("Dedup complete", before=len(papers), after=len(out))
    return out


# =========================
# Public API
# =========================

async def search_papers(topic: str, days: int = 30, limit_total: int = 120, debug: bool = False) -> Dict[str, Any]:
    """
    Aggregate & rank papers for a topic within a date window (7/30/365).
    Returns with_doi / no_doi / all (and 'debug' text if debug=True).
    """
    setup_logging(debug)
    dbg = Debug()
    dbg.add("Params", topic=topic, days=days, limit_total=limit_total, user_agent=UA, pyalex_email=bool(PYALEX_EMAIL), have_pyalex=HAVE_PYALEX)

    topic = (topic or "").strip()
    if not topic:
        dbg.add("Empty topic — abort")
        result = {"with_doi": [], "no_doi": [], "all": []}
        if debug:
            result["debug"] = dbg.text()
        return result

    tasks = [
        fetch_openalex_pyalex(topic, days, limit=max(20, limit_total // 2), debug=dbg),
        fetch_arxiv(topic, days, limit=max(20, limit_total // 3), debug=dbg),
        fetch_hf_papers(topic, limit=max(10, limit_total // 6), debug=dbg),
    ]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        dbg.add("gather error", error=str(e), tb=traceback.format_exc())
        results = []

    merged: List[Paper] = []
    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            dbg.add("source failed at gather", idx=idx, error=str(r))
            continue
        merged.extend(r)
    dbg.add("Merged count", count=len(merged))

    merged = dedupe(merged, dbg)
    for p in merged:
        p.score = compute_score(p)
    merged.sort(key=lambda x: x.score, reverse=True)

    with_doi  = [p.to_dict() for p in merged if p.doi]
    no_doi    = [p.to_dict() for p in merged if not p.doi]
    all_items = [p.to_dict() for p in merged]

    if len(all_items) == 0:
        dbg.add("No results hints",
                suggestions=[
                    "Try simpler/different keywords",
                    "Increase days (30 or 365)",
                    "Check your network/proxy for access to export.arxiv.org and huggingface.co",
                    "Install pyalex and set PYALEX_EMAIL; the code also retries without language filter"
                ])

    result: Dict[str, Any] = {"with_doi": with_doi, "no_doi": no_doi, "all": all_items}
    if debug:
        result["debug"] = dbg.text()
    return result


# =========================
# CLI entrypoint
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Papers Parser (PyAlex + arXiv + HF /papers)")
    parser.add_argument("topic", type=str, help="Search topic (English)")
    parser.add_argument("--days", type=int, default=30, choices=[7, 30, 365], help="Date window")
    parser.add_argument("--limit", type=int, default=120, help="Approx total cap across sources")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and save to papers_debug.txt")
    args = parser.parse_args()

    data = asyncio.run(search_papers(args.topic, days=args.days, limit_total=args.limit, debug=args.debug))

    # Summary + top-5
    print(f"Total: {len(data['all'])} | DOI: {len(data['with_doi'])} | NoDOI: {len(data['no_doi'])}")
    for i, p in enumerate(data["all"][:5], 1):
        print(f"{i:>2}. {p['title']}  [{p['source']}]  {p.get('date','')}")

    # Save full JSON
    with open("papers_sample.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print("Saved: papers_sample.json")

    # Save debug (if enabled)
    if args.debug and "debug" in data:
        with open("papers_debug.txt", "w", encoding="utf-8") as f:
            f.write(data["debug"])
        print("\n--- DEBUG WRITTEN TO papers_debug.txt ---")
        # also print first lines to console for quick glance
        print("\n".join(data["debug"].splitlines()[:40]))
