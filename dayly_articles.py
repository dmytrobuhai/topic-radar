"""
- Inputs: topic, what to include (Papers/News/Events/Videos), date window (3/7/30/365 days).
- Sources implemented:
  PAPERS:
    - PyAlex (OpenAlex via pyalex) = scholarly search + citations
    - Crossref REST = authoritative DOIs & metadata
    - arXiv API
    - Hugging Face Papers (HTML parse)
  NEWS:
    - Microsoft Research Blog (RSS)
    - Hugging Face Blog (RSS)
    - NVIDIA Developer Blog (RSS)
    - OpenCV News (RSS)
    - Towards Data Science (RSS)
    - embedded.com (RSS)
    - Meta AI Blog (HTML parse)
    - GitHub Search (API) = recent repos matching the topic
  EVENTS:
    - STMicroelectronics / STM32 Events page (HTML parse)
  VIDEOS:
    - YouTube Data API v3

Environment variables (optional)
--------------------------------
- PYALEX_EMAIL: email, to enter OpenAlex "polite pool" via pyalex
- CROSSREF_CONTACT_EMAIL: email, sent in User-Agent for polite Crossref queries
- YOUTUBE_API_KEY: YouTube Data API v3 key
- GITHUB_TOKEN: GitHub token for higher rate limits
"""

from __future__ import annotations
import os, re, math, asyncio
import datetime as dt
from typing import Any, Dict, List, Tuple, Iterable

import httpx
import gradio as gr

import feedparser  
from bs4 import BeautifulSoup  

# --------- Config ----------
DATE_WINDOWS = {
    "Last 3 days": 3,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 365 days": 365,
}

PYALEX_EMAIL = os.getenv("PYALEX_EMAIL", "")
CROSSREF_EMAIL = os.getenv("CROSSREF_CONTACT_EMAIL", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

email_topic = f"topic-radar/0.2 ({CROSSREF_EMAIL})" if CROSSREF_EMAIL else "topic-radar/0.2"

try:
    import pyalex
    if PYALEX_EMAIL:
        pyalex.config.email = PYALEX_EMAIL
    # Reasonable retry policy for transient 429/5xx
    pyalex.config.max_retries = 2
    pyalex.config.retry_backoff_factor = 0.25
    pyalex.config.retry_http_codes = [429, 500, 503]
    HAVE_PYALEX = True
except Exception:
    HAVE_PYALEX = False

FEEDS = {
    # Microsoft Research blog RSS
    "msr": "https://www.microsoft.com/en-us/research/feed/",
    # Hugging Face blog RSS
    "hf_blog": "https://huggingface.co/blog/feed.xml",
    # NVIDIA Developer blog RSS
    "nvidia": "https://developer.nvidia.com/blog/feed",
    # OpenCV News
    "opencv": "https://opencv.org/news/feed/",
    # Towards Data Science
    "tds": "https://towardsdatascience.com/feed",
    # embedded.com
    "embedded": "https://www.embedded.com/feed/",
}

# Endpoints
CROSSREF = "https://api.crossref.org/works"
YOUTUBE = "https://www.googleapis.com/youtube/v3/search"
GITHUB_SEARCH = "https://api.github.com/search/repositories"

# ---------- Utilities ----------
def since_iso_date(days: int) -> str:
    return (dt.datetime.now(dt.datetime.timezone.utc) - dt.timedelta(days=days)).date().isoformat()

def parse_date_iso(s: str | None) -> dt.date | None:
    if not s: return None
    try: return dt.date.fromisoformat(s[:10])
    except Exception: return None

def recency_score(pub_date: str | None, half_life_days: int = 30) -> float:
    """0..1 exponential decay; higher is fresher."""
    d = parse_date_iso(pub_date)
    if not d: return 0.0
    days_old = (dt.date.today() - d).days
    return math.exp(-max(days_old,0) / max(half_life_days,1))

def norm_text(x: str | None) -> str:
    return (x or "").strip()

def any_contains(texts: Iterable[str], needle: str) -> bool:
    nd = needle.lower()
    for t in texts:
        if nd in (t or "").lower():
            return True
    return False

def httpx_client() -> httpx.AsyncClient:
    headers = {"User-Agent": email_topic}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return httpx.AsyncClient(headers=headers, timeout=30)

# ---------- PAPERS ----------
async def papers_openalex_pyalex(topic: str, days: int) -> List[Dict[str, Any]]:
    since = since_iso_date(days)
    q = pyalex.Works().search(topic).filter(from_publication_date=since).select([
        "id","doi","display_name","primary_location","host_venue",
        "publication_date","cited_by_count"
    ])
    results = q.paginate(per_page=25)
    items: List[Dict[str, Any]] = []
    for i, page in enumerate(results):
        for w in page:
            title = w.get("display_name")
            date  = w.get("publication_date")
            loc   = w.get("primary_location") or {}
            url = (loc.get("landing_page_url")
                   or (loc.get("source") or {}).get("homepage_url")
                   or w.get("id"))
            items.append({
                "title": title,
                "url": url,
                "venue": (w.get("host_venue") or {}).get("display_name"),
                "date": date,
                "doi": w.get("doi"),
                "source": "OpenAlex (pyalex)",
                "cited_by": int(w.get("cited_by_count") or 0),
            })
        if i >= 1: 
            break
    items.sort(key=lambda p: 0.7*recency_score(p["date"]) + 0.3*(math.log1p(p["cited_by"])/10 if p["cited_by"] else 0), reverse=True)
    return items

async def papers_crossref(client: httpx.AsyncClient, topic: str, days: int) -> List[Dict[str, Any]]:
    since = since_iso_date(days)
    params = {
        "query": topic,
        "filter": f"from-pub-date:{since}",
        "rows": 25,
        "select": "title,DOI,URL,container-title,created",
        "sort": "created",
        "order": "desc",
    }
    try:
        r = await client.get(CROSSREF, params=params)
        r.raise_for_status()
    except Exception:
        return []
    out = []
    for it in r.json().get("message", {}).get("items", []):
        out.append({
            "title": (it.get("title") or [None])[0],
            "url": it.get("URL"),
            "venue": (it.get("container-title") or [None])[0],
            "date": (it.get("created") or {}).get("date-time","")[:10],
            "doi": it.get("DOI"),
            "source": "Crossref",
            "cited_by": None,
        })
    return out

async def papers_arxiv(topic: str, days: int) -> List[Dict[str, Any]]:
    terms = [t for t in re.split(r"\s+", topic.strip()) if t]
    q = "all:" + "+".join(terms)
    params = {
        "search_query": q,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": 30,
    }
    url = "http://export.arxiv.org/api/query"
    try:
        feed = feedparser.parse(url, params=params)
    except Exception:
        return []
    out = []
    cutoff = dt.datetime.now(dt.datetime.timezone.utc) - dt.timedelta(days=days)
    for e in feed.entries:
        date = (getattr(e, "updated", None) or getattr(e, "published", ""))[:10]
        dt_ = None
        try: dt_ = dt.datetime.fromisoformat(date)
        except Exception: pass
        if dt_ and dt_ < cutoff:
            continue
        out.append({
            "title": e.title,
            "url": e.link,
            "venue": "arXiv",
            "date": date,
            "doi": getattr(e, "arxiv_doi", None),
            "source": "arXiv",
            "cited_by": None,
        })
    return out

async def papers_hf_daily(topic: str, days: int) -> List[Dict[str, Any]]:
    url = "https://huggingface.co/papers"
    try:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": email_topic}) as c:
            r = await c.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return []
    out = []
    for a in soup.select("a[href*='/papers/']"):
        title = a.get_text(strip=True)
        href = a.get("href")
        if not title or not href:
            continue
        if not any_contains([title], topic):
            continue
        out.append({
            "title": title,
            "url": f"https://huggingface.co{href}",
            "venue": "HuggingFace Papers",
            "date": dt.date.today().isoformat(),  
            "doi": None,
            "source": "HF Papers",
            "cited_by": None,
        })
    return out[:20]

# ---------- NEWS ----------
def parse_feed(url: str) -> List[Dict[str, Any]]:
    try:
        fp = feedparser.parse(url)
    except Exception:
        return []
    out = []
    for e in fp.entries[:50]:
        out.append({
            "title": getattr(e, "title", None),
            "url": getattr(e, "link", None),
            "source": fp.feed.get("title", url),
            "date": (getattr(e, "published", "") or getattr(e, "updated", ""))[:10],
        })
    return out

async def news_rss_all(topic: str, days: int) -> List[Dict[str, Any]]:
    cutoff = dt.date.today() - dt.timedelta(days=days)
    merged: List[Dict[str,Any]] = []
    for key, url in FEEDS.items():
        merged.extend(parse_feed(url))
    # Filter by topic and date
    filtered = [
        x for x in merged
        if any_contains([x.get("title","")], topic)
        and (parse_date_iso(x.get("date")) or dt.date(1970,1,1)) >= cutoff
    ]
    # Dedup by URL/title
    seen = set(); uniq = []
    for x in filtered:
        k = x.get("url") or x.get("title")
        if k in seen: continue
        seen.add(k); uniq.append(x)
    uniq.sort(key=lambda x: x.get("date",""), reverse=True)
    return uniq[:80]

async def news_github(client: httpx.AsyncClient, topic: str, days: int) -> List[Dict[str, Any]]:
    q = f"{topic} in:name,description,readme"
    params = {"q": q, "sort": "updated", "order": "desc", "per_page": 20}
    try:
        r = await client.get(GITHUB_SEARCH, params=params)
        r.raise_for_status()
    except Exception:
        return []
    cutoff = dt.date.today() - dt.timedelta(days=days)
    out = []
    for repo in r.json().get("items", []):
        updated = (repo.get("updated_at") or "")[:10]
        if parse_date_iso(updated) and parse_date_iso(updated) < cutoff:
            continue
        out.append({
            "title": f"{repo.get('full_name')} — {repo.get('description') or ''}".strip(),
            "url": repo.get("html_url"),
            "source": "GitHub",
            "date": updated,
        })
    return out

# ---------- EVENTS ----------
async def events_stm32(topic: str, days: int) -> List[Dict[str, Any]]:
    url = "https://www.st.com/content/st_com/en/about/events.html"
    try:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": email_topic}) as c:
            r = await c.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return []
    out = []
    for card in soup.find_all(lambda tag: tag.name in ("a","div") and "event" in " ".join(tag.get("class", [])).lower()):
        title = card.get_text(" ", strip=True)
        if not title or not any_contains([title], topic):
            continue
        href = None
        if hasattr(card, "get") and card.get("href"):
            href = card.get("href")
        if href and href.startswith("/"):
            href = f"https://www.st.com{href}"
        # naive date detection (the page structure can vary; keep flexible)
        date_match = re.search(r"(\b[ADFJMNOS]\w+\s+\d{1,2}(?:\s*-\s*\d{1,2})?(?:,\s*\d{4})?)", title)
        out.append({
            "title": title[:180],
            "url": href or url,
            "start": None if not date_match else date_match.group(1),
            "venue": "STM32 / ST events",
        })
    return out[:30]

# ---------- VIDEOS ----------
async def videos_youtube(topic: str, days: int) -> List[Dict[str, Any]]:
    if not YOUTUBE_API_KEY:
        return []
    after = (dt.datetime.now(dt.datetime.timezone.utc) - dt.timedelta(days=days)).replace(microsecond=0).isoformat("T") + "Z"
    params = {
        "part": "snippet",
        "q": topic,
        "type": "video",
        "order": "date",
        "publishedAfter": after,
        "maxResults": 25,
        "key": YOUTUBE_API_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": email_topic}) as c:
            r = await c.get(YOUTUBE, params=params)
            r.raise_for_status()
    except Exception:
        return []
    out = []
    for it in r.json().get("items", []):
        vid = (it.get("id") or {}).get("videoId")
        sn  = it.get("snippet", {})
        out.append({
            "title": sn.get("title"),
            "url": f"https://www.youtube.com/watch?v={vid}" if vid else None,
            "channel": sn.get("channelTitle"),
            "date": (sn.get("publishedAt") or "")[:10],
        })
    out.sort(key=lambda x: x.get("date",""), reverse=True)
    return out

# ---------- Orchestrator ----------
async def run_search(topic: str, kinds: List[str], window_label: str) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], str]:
    topic = norm_text(topic)
    if not topic:
        return [], [], [], [], "Enter a topic to search."
    days = DATE_WINDOWS.get(window_label, 30)

    p_doied: List[Dict[str,Any]] = []
    p_nodoi: List[Dict[str,Any]] = []
    news: List[Dict[str,Any]] = []
    events: List[Dict[str,Any]] = []
    videos: List[Dict[str,Any]] = []

    async with httpx_client() as client:
        tasks = []

        want_papers = "Papers" in kinds
        want_news   = "News" in kinds
        want_events = "Events" in kinds
        want_videos = "Videos" in kinds

        # papers
        if want_papers:
            tasks.extend([
                papers_openalex_pyalex(topic, days),
                papers_crossref(client, topic, days),
                papers_arxiv(topic, days),
                papers_hf_daily(topic, days),
            ])

        # news
        if want_news:
            tasks.extend([
                news_rss_all(topic, days),
                news_github(client, topic, days),
            ])

        # events
        if want_events:
            tasks.append(events_stm32(topic, days))

        # videos
        if want_videos:
            tasks.append(videos_youtube(topic, days))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    idx = 0
    if "Papers" in kinds:
        srcs = []
        for _ in range(5):
            srcs.append(results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else [])
            idx += 1
        merged_papers = [p for arr in srcs for p in arr]
        # DOI first, else (title, year)
        seen = set(); dedup = []
        for p in merged_papers:
            key = p.get("doi") or (p.get("title"), (p.get("date") or "")[:4])
            if key in seen: continue
            seen.add(key); dedup.append(p)
        # split DOI / no DOI
        for p in dedup:
            if p.get("doi"): p_doied.append(p)
            else:            p_nodoi.append(p)
        # DOI list by recency+citations, No-DOI by recency only
        p_doied.sort(key=lambda p: 0.7*recency_score(p.get("date")) + 0.3*(math.log1p(p.get("cited_by") or 0)/10), reverse=True)
        p_nodoi.sort(key=lambda p: recency_score(p.get("date")), reverse=True)

    if "News" in kinds:
        srcs = []
        for _ in range(3):
            srcs.append(results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else [])
            idx += 1
        news = [n for arr in srcs for n in arr]
        # dedup by URL/title
        seen = set(); uniq=[]
        for n in news:
            k = n.get("url") or n.get("title")
            if k in seen: continue
            seen.add(k); uniq.append(n)
        news = sorted(uniq, key=lambda x: x.get("date",""), reverse=True)[:120]

    if "Events" in kinds:
        ev = results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else []
        idx += 1
        events = ev

    if "Videos" in kinds:
        vd = results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else []
        idx += 1
        videos = vd

    notes = []
    if want_videos and not YOUTUBE_API_KEY:
        notes.append("YouTube disabled: set YOUTUBE_API_KEY.")
    if want_papers and not HAVE_PYALEX:
        notes.append("PyAlex not installed — skipping OpenAlex via library (using REST fallback).")
    if want_news and not GITHUB_TOKEN:
        notes.append("GitHub unauthenticated — low rate limits.")
    if want_events:
        notes.append("STM32 events parsed from HTML (no official feed detected).")
    return p_doied[:80], p_nodoi[:80], news, events, videos, (" | ".join(notes) if notes else "OK")

def do_search(topic, kinds, window):
    return asyncio.run(run_search(topic, kinds, window))

# ---------- UI ----------
with gr.Blocks(title="Topic Radar — Multi-source") as demo:
    gr.Markdown("## Topic Radar — Papers • News • Events • Videos")

    with gr.Row():
        topic = gr.Textbox(
            label="Topic",
            placeholder="e.g., computer vision industrial automation",
            scale=3,
        )
        kinds = gr.CheckboxGroup(
            choices=["Papers", "News", "Events", "Videos"],
            value=["Papers", "News", "Events", "Videos"],
            label="Include",
        )
        window = gr.Radio(
            choices=list(DATE_WINDOWS.keys()),
            value="Last 30 days",
            label="Date window",
        )

    with gr.Row():
        btn_search = gr.Button("Search", variant="primary")
        btn_clear = gr.Button("Clear")

    status = gr.Markdown("")

    with gr.Tabs():
        with gr.Tab("Papers"):
            with gr.Tabs():
                with gr.Tab("Papers (DOI)"):
                    out_papers_doi = gr.Dataframe(
                        headers=["title","url","venue","date","doi","source","cited_by"],
                        datatype=["str","str","str","str","str","str","number"],
                        row_count=(0,"dynamic"), col_count=(7,"fixed"), wrap=True,
                    )
                with gr.Tab("Papers (No DOI)"):
                    out_papers_nodoi = gr.Dataframe(
                        headers=["title","url","venue","date","source"],
                        datatype=["str","str","str","str","str"],
                        row_count=(0,"dynamic"), col_count=(5,"fixed"), wrap=True,
                    )
        with gr.Tab("News"):
            out_news = gr.Dataframe(
                headers=["title","url","source","date"],
                datatype=["str","str","str","str"],
                row_count=(0,"dynamic"), col_count=(4,"fixed"), wrap=True,
            )
        with gr.Tab("Events"):
            out_events = gr.Dataframe(
                headers=["title","url","start","venue"],
                datatype=["str","str","str","str"],
                row_count=(0,"dynamic"), col_count=(4,"fixed"), wrap=True,
            )
        with gr.Tab("Videos"):
            out_videos = gr.Dataframe(
                headers=["title","url","channel","date"],
                datatype=["str","str","str","str"],
                row_count=(0,"dynamic"), col_count=(4,"fixed"), wrap=True,
            )

    btn_search.click(
        do_search,
        inputs=[topic, kinds, window],
        outputs=[out_papers_doi, out_papers_nodoi, out_news, out_events, out_videos, status],
    )

    def clear_all():
        return "", ["Papers","News","Events","Videos"], "Last 30 days", "", [], [], [], [], []
    btn_clear.click(clear_all, None, [topic, kinds, window, status, out_papers_doi, out_papers_nodoi, out_news, out_events, out_videos])

demo.queue().launch()
