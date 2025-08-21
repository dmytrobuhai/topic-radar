# youtube_parser.py
"""
YouTube search with date window, PRF expansion, lexical scoring, and ranking.

- Multi-strategy retrieval (phrase, AND-like tokens) within [now-days, now]
- Pseudo-Relevance Feedback (PRF) from title+description
- Confidence = title_coverage*w1 + desc_coverage*w2 + PRF_title*w3 + PRF_desc*w4 + recency*w5
- Dedupe, sort, JSON-friendly output, CLI

Env:
  setx YT_API_KEY "your key"


Usage:
  python youtube_api.py "computer vision industrial automation" --days 10 --max 10 --debug
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Config ----------------
YT_KEY = os.getenv("YT_API_KEY", "").strip()
YT_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
HEADERS = {"User-Agent": "topic-radar/1.0 (youtube client)"}
TIMEOUT = 30
PER_PAGE = 30           
SLEEP = 0.6             

# Scoring weights 
W_TITLE_BASE = 0.60
W_DESC_BASE  = 0.25
W_TITLE_PRF  = 0.10
W_DESC_PRF   = 0.03
W_RECENCY    = 0.02

# PRF params
PRF_TOP_DOCS  = 40
PRF_TOP_TERMS = 10
PRF_MIN_LEN   = 3
PRF_MAX_LEN   = 24

# Stopwords 
STOPWORDS = {
    "and","or","for","the","a","an","of","in","on","to","with","by","at",
    "from","into","using","via","study",
    "new","novel","how","what","why","when","where"
}

# ---------------- Time helpers ----------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _to_rfc3339(dttm: dt.datetime) -> str:
    return dttm.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _date_window(days: int) -> Tuple[str, str]:
    end_dt = _now_utc()
    start_dt = (end_dt - dt.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return _to_rfc3339(start_dt), _to_rfc3339(end_dt)

def _recency_score(date_iso: Optional[str], half_life_days: int = 21) -> float:
    if not date_iso:
        return 0.0
    try:
        d = dateparse.parse(date_iso).date()
    except Exception:
        return 0.0
    today = _now_utc().date()
    days_old = (today - d).days
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
    """Compile word-boundary regexes for tokens."""
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tokens]

def _coverage(text: Optional[str], pats: List[re.Pattern]) -> Tuple[float, List[str]]:
    """Return fraction [0...1] of tokens matched and the matched unique tokens."""
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

# ---------------- PRF ----------------
def _doc_tokens(title: Optional[str], desc: Optional[str]) -> List[str]:
    """Tokens for PRF from title+description with length constraints."""
    text = f"{title or ''} {desc or ''}"
    return [t for t in _tokens(text) if PRF_MIN_LEN <= len(t) <= PRF_MAX_LEN]

def prf_terms(cands: List[Dict[str, Any]], base_tokens: List[str],
              k_docs: int = PRF_TOP_DOCS, top_terms: int = PRF_TOP_TERMS) -> List[str]:
    """Extract PRF terms via simple TF-IDF from top-k candidate videos."""
    docs = [_doc_tokens(c.get("title"), c.get("description")) for c in cands[:k_docs]]
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
    """Return configured HTTP client."""
    return httpx.Client(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)

def _yt_get(url: str, params: Dict[str, Any], debug: bool=False) -> Dict[str, Any]:
    """GET YouTube API endpoint with key; JSON or raise."""
    if not YT_KEY:
        raise RuntimeError("Missing YT_API_KEY env var")
    qp = {"key": YT_KEY, **params}
    with _client() as c:
        if debug:
            print("[DEBUG] GET", url, "params=", {k: v for k, v in qp.items() if k != "key"})
        r = c.get(url, params=qp)
        if debug:
            print("[DEBUG] status", r.status_code, "bytes", len(r.content))
        if r.status_code == 403 and b"quota" in r.content.lower():
            raise RuntimeError("YouTube quota exceeded or access forbidden")
        r.raise_for_status()
        return r.json()

# ---------------- Normalization ----------------
def _normalize_search_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a single search result (snippet) to internal dict (without stats)."""
    if it.get("id", {}).get("kind") != "youtube#video":
        return None
    vid = it["id"].get("videoId")
    sn = it.get("snippet") or {}
    title = sn.get("title")
    desc = sn.get("description")

    published_at = sn.get("publishedAt")
    published_iso = None

    if published_at:
        try:
            dt_obj = dateparse.parse(published_at)
            published_iso = dt_obj.strftime("%Y-%m-%d")
        except Exception:
            published_iso = published_at


    channel = sn.get("channelTitle")
    channel_id = sn.get("channelId")
    url = f"https://www.youtube.com/watch?v={vid}" if vid else None
    return {
        "source": "YouTube",
        "video_id": vid,
        "title": title,
        "description": desc,
        "published": published_iso,
        "channel": channel,
        "channel_id": channel_id,
        "url": url,
        "stats": {}, 
    }

def _attach_stats(items: List[Dict[str, Any]], debug: bool=False) -> None:
    """Batch-fetch statistics for given items and attach in-place."""
    ids = [it["video_id"] for it in items if it.get("video_id")]
    if not ids:
        return
    for i in range(0, len(ids), 30):
        chunk = ids[i:i+30]
        data = _yt_get(YT_VIDEOS, {
            "part": "statistics,contentDetails",
            "id": ",".join(chunk),
            "maxResults": 30
        }, debug=debug)
        by_id = {x["id"]: x for x in (data.get("items") or [])}
        for it in items:
            vid = it.get("video_id")
            if not vid or vid not in by_id:
                continue
            st = by_id[vid].get("statistics", {})
            it["stats"] = {
                "viewCount": int(st.get("viewCount", 0)) if st.get("viewCount") else None,
                "likeCount": int(st.get("likeCount", 0)) if st.get("likeCount") else None,
                "commentCount": int(st.get("commentCount", 0)) if st.get("commentCount") else None,
            }
        if debug:
            print(f"[DEBUG] stats attached for {len(chunk)} ids")
        time.sleep(SLEEP)

# ---------------- Query building ----------------
def build_initial_queries(topic: str) -> List[str]:
    """Return two query strings: 'AND-like' tokens and an exact-phrase variant."""
    toks = _tokens(topic) or ["universal", "search"]
    q_and = " ".join(toks)
    phrase = " ".join(toks)
    q_phrase = f"\"{phrase}\""
    out: List[str] = []
    for q in (q_phrase, q_and):
        if q not in out:
            out.append(q)
    return out

# ---------------- Retrieval ----------------
def fetch_query(q: str, days: int, want: int, debug: bool=False) -> List[Dict[str, Any]]:
    """Fetch up to 'want' normalized videos for a query within the date window."""
    after, before = _date_window(days)
    collected: List[Dict[str, Any]] = []
    page_token = None

    while len(collected) < want:
        params = {
            "part": "snippet",
            "type": "video",
            "q": q,
            "maxResults": min(PER_PAGE, want - len(collected)),
            "order": "date",                 
            "publishedAfter": after,
            "publishedBefore": before,
            "safeSearch": "none",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _yt_get(YT_SEARCH, params, debug=debug)
        items = data.get("items") or []
        for raw in items:
            norm = _normalize_search_item(raw)
            if norm:
                collected.append(norm)
                if len(collected) >= want:
                    break
        page_token = data.get("nextPageToken")
        if debug:
            total = data.get("pageInfo", {}).get("totalResults")
            print(f"[DEBUG] q='{q}' +{len(items)} (col={len(collected)}/{want}) next={bool(page_token)} total={total}")
        if not page_token or len(collected) >= want:
            break
        time.sleep(SLEEP)

    _attach_stats(collected, debug=debug)
    return collected

def multi_strategy_retrieve(topic: str, days: int, pool_target: int, debug: bool=False) -> List[Dict[str, Any]]:
    """Run phrase and AND-like queries; merge unique until pool_target."""
    qs = build_initial_queries(topic)
    if debug:
        print("[DEBUG] initial queries (YouTube):")
        for q in qs: print("   ", q)

    seen = set()
    pool: List[Dict[str, Any]] = []
    per_try = max(60, min(PER_PAGE, pool_target))

    for qi, q in enumerate(qs, 1):
        chunk = fetch_query(q, days=days, want=per_try, debug=debug)
        added = 0
        for v in chunk:
            key = v.get("video_id") or v.get("url")
            if key and key not in seen:
                seen.add(key); pool.append(v); added += 1
        if debug:
            print(f"[DEBUG] Q{qi} -> got={len(chunk)} added_unique={added} pool={len(pool)}")
        if len(pool) >= pool_target:
            break
    return pool[:pool_target]

# ---------------- Scoring & selection ----------------
def score_video(v: Dict[str, Any], base_pats: List[re.Pattern], prf_pats: List[re.Pattern]) -> Dict[str, Any]:
    """Compute confidence and attach match breakdown."""
    title = v.get("title") or ""
    desc  = v.get("description") or ""
    tb, th_b = _coverage(title, base_pats)
    db, dh_b = _coverage(desc,  base_pats)
    tp, th_p = _coverage(title, prf_pats) if prf_pats else (0.0, [])
    dp, dh_p = _coverage(desc,  prf_pats) if prf_pats else (0.0, [])
    rec = _recency_score(v.get("published"))
    conf = W_TITLE_BASE*tb + W_DESC_BASE*db + W_TITLE_PRF*tp + W_DESC_PRF*dp + W_RECENCY*rec

    v["match"] = {
        "base": {"title_hits": th_b, "desc_hits": dh_b, "title_coverage": round(tb,4), "desc_coverage": round(db,4)},
        "prf":  {"title_hits": th_p, "desc_hits": dh_p, "title_coverage": round(tp,4), "desc_coverage": round(dp,4)},
    }
    v["confidence"] = round(conf, 6)
    return v

def dedupe(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by video_id/url."""
    seen = set(); out: List[Dict[str, Any]] = []
    for v in videos:
        key = v.get("video_id") or v.get("url")
        if key and key not in seen:
            seen.add(key); out.append(v)
    return out

def youtube_search(topic: str, days: int = 14, max_results: int = 30, debug: bool=False) -> List[Dict[str, Any]]:
    """Run pipeline: retrieve → PRF expand → retrieve → score → rank → top-K."""
    base_tokens = _tokens(topic)
    base_pats = _compile_word_patterns(base_tokens)

    pool_target = max(5 * max_results, 150)
    pool = multi_strategy_retrieve(topic, days, pool_target=pool_target, debug=debug)
    if debug:
        print(f"[DEBUG] initial pool: {len(pool)}")

    # PRF expansion
    prf = prf_terms(pool, base_tokens, k_docs=min(PRF_TOP_DOCS, len(pool)), top_terms=PRF_TOP_TERMS)
    prf = [t for t in prf if t not in base_tokens]
    if debug:
        print("[DEBUG] PRF terms:", prf)

    if prf:
        q_expanded = " ".join(base_tokens + prf[:PRF_TOP_TERMS])
        more = fetch_query(q_expanded, days=days, want=pool_target, debug=debug)
        before = len(pool)
        pool.extend(more)
        pool = dedupe(pool)
        if debug:
            print(f"[DEBUG] PRF retrieve added: {len(pool) - before}, pool now: {len(pool)}")

    prf_pats = _compile_word_patterns(prf) if prf else []
    scored: List[Dict[str, Any]] = []
    for v in pool:
        v = score_video(v, base_pats, prf_pats)
        ok = (
            v["match"]["base"]["title_hits"] or v["match"]["base"]["desc_hits"] or
            v["match"]["prf"]["title_hits"]  or v["match"]["prf"]["desc_hits"]
        )
        if ok:
            scored.append(v)

    # Sort
    def _engagement(x: Dict[str, Any]) -> float:
        vc = (x.get("stats") or {}).get("viewCount") or 0
        return math.log1p(vc) / 15.0  

    scored.sort(
        key=lambda x: (
            x.get("confidence", 0.0),
            x.get("published") or "",
            _engagement(x),
        ),
        reverse=True
    )
    return scored[:max_results]

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YouTube search with PRF and lexical scoring")
    ap.add_argument("topic", type=str, help="Search topic")
    ap.add_argument("--days", type=int, default=14, help="Published within last N days")
    ap.add_argument("--max", type=int, default=30, help="Max results to return")
    ap.add_argument("--debug", action="store_true", help="Verbose debug")
    args = ap.parse_args()

    results = youtube_search(args.topic, days=args.days, max_results=args.max, debug=args.debug)
    print(f"Found: {len(results)}")
    for i, v in enumerate(results[:10], 1):
        conf = v.get("confidence", 0.0)
        date_show = v.get("published") or ""
        ch = v.get("channel") or ""
        views = (v.get("stats") or {}).get("viewCount")
        print(f"{i:>2}. {v.get('title','')[:120]}  [conf={conf:.3f}]  [{date_show}]  ({ch})  views={views}")
        print(f"    url: {v.get('url')}")
    with open("youtube_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: youtube_sample.json")
