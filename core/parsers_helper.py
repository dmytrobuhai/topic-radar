# core/parsers_helper.py
from __future__ import annotations
import datetime as dt
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

# ---- constants ----
DEFAULT_TIMEOUT = 40
DEFAULT_HEADER = "topic-radar/1.0 (+parsers_helper)"

# stopwords set
DEFAULT_STOPWORDS = {
    "and","or","for","the","a","an","of","in","on","to","with","by","at","from","into",
    "via","towards","toward","study","results","new","novel",
    "how","why","what","when","where"
}

# ---- Time helpers ----
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def today_utc() -> dt.date:
    return now_utc().date()

def recency_score(date_iso: Optional[str], *, half_life_days: int = 60) -> float:
    """
    Exponential half-life decay [0...1].
    """
    if not date_iso:
        return 0.0
    try:
        d = dt.date.fromisoformat(date_iso[:10])
    except Exception:
        return 0.0
    days_old = (today_utc() - d).days
    return math.exp(-math.log(2) * max(days_old, 0) / max(half_life_days, 1))

# ---- Text helpers ----
def tokens(s: str, *, stopwords: Optional[set[str]] = None,
           keep_unique_in_order: bool = True) -> List[str]:
    """
    Lowercase tokenization, drops stopwords/digits, filters silly lengths, returns tokens.
    """
    stw = stopwords or DEFAULT_STOPWORDS
    raw = [t for t in re.split(r"\W+", (s or "").lower()) if t]
    out: List[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in stw or t.isdigit() or not (1 <= len(t) <= 32):
            continue
        if keep_unique_in_order:
            if t not in seen:
                seen.add(t)
                out.append(t)
        else:
            out.append(t)
    return out

def compile_word_patterns(tt: Sequence[str]) -> List[re.Pattern]:
    """Regex patterns for exact token hits with word boundaries."""
    return [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in tt]

def coverage(text: Optional[str], pats: Sequence[re.Pattern]) -> Tuple[float, List[str]]:
    """
    Coverage is fraction of the pattern list that is present at least once.
    """
    if not text:
        return 0.0, []
    hits: List[str] = []
    for p in pats:
        if p.search(text):
            tok = p.pattern.replace(r"\b", "")
            tok = re.sub(r"\\(.)", r"\1", tok).strip("^$")
            hits.append(tok.lower())
    uniq = list(dict.fromkeys(hits))
    denom = max(len(pats), 1)
    return len(uniq) / denom, uniq

def phrase_in_text(text: Optional[str], base_tokens: Sequence[str]) -> bool:
    """Detect exact phrase occurrence."""
    if not text or not base_tokens:
        return False
    phrase = " ".join(base_tokens)
    return re.search(re.escape(phrase), text, re.IGNORECASE) is not None

# ---- PRF (pseudo-relevance feedback) ----
def prf_terms_from_pairs(
    pairs: Iterable[Tuple[Optional[str], Optional[str]]],
    base_tokens: Sequence[str],
    *,
    top_docs: int = 30,
    top_terms: int = 12,
    min_len: int = 3,
    max_len: int = 24,
    stopwords: Optional[set[str]] = None,
) -> List[str]:
    """
    PRF terms from (title, abstract/summary) pairs using a simple TF-IDF heuristic.
    - pairs: iterable of (title, body) strings
    - base_tokens: tokens to exclude from candidates
    """
    stw = stopwords or DEFAULT_STOPWORDS
    docs: List[List[str]] = []
    for i, (t, b) in enumerate(pairs):
        if i >= top_docs:
            break
        text = f"{t or ''} {b or ''}"
        doc = [w for w in tokens(text, stopwords=stw) if min_len <= len(w) <= max_len]
        docs.append(doc)
    if not docs:
        return []

    # Document frequencies
    df: Dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1

    N = len(docs)
    base = set(base_tokens)
    scores: Dict[str, float] = defaultdict(float)

    for d in docs:
        tf: Dict[str, int] = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        for w, f in tf.items():
            if w in base or w.isdigit() or not (min_len <= len(w) <= max_len):
                continue
            idf = math.log((N + 1) / (1 + df.get(w, 0))) + 1.0
            scores[w] += (f / max(len(d), 1)) * idf

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:top_terms]]

# ---- Confidence ----
def compute_confidence(
    *,
    title_base_cov: float,
    abs_base_cov: float,
    title_prf_cov: float,
    abs_prf_cov: float,
    recency: float,
    weights: Dict[str, float],
    scale_100: bool = True
) -> float:
    """
    Combines components via weights (W_TITLE_BASE, W_ABS_BASE, W_TITLE_PRF, W_ABS_PRF, W_RECENCY).
    Returns clamped confidence (0..100 if scale_100 else 0..1).
    """
    s = (
        weights.get("W_TITLE_BASE", 0.55) * title_base_cov +
        weights.get("W_ABS_BASE",   0.20) * abs_base_cov   +
        weights.get("W_TITLE_PRF",  0.15) * title_prf_cov  +
        weights.get("W_ABS_PRF",    0.05) * abs_prf_cov    +
        weights.get("W_RECENCY",    0.05) * recency
    )
    if scale_100:
        s *= 100.0
    return max(0.0, min(100.0 if scale_100 else 1.0, s))

def pass_threshold(conf: float, *, threshold: float = 25.0) -> bool:
    """Keep only items above or equal to a given (0..100) threshold."""
    return conf >= threshold

# ---- HTTP ----
def http_get_text(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    debug: bool = False,
) -> str:
    hdrs = {"User-Agent": DEFAULT_HEADER, **(headers or {})}
    if debug:
        print("[DEBUG] GET", url, "params=", params or {})
    with httpx.Client(headers=hdrs, timeout=timeout, follow_redirects=True) as c:
        r = c.get(url, params=params)
        if debug:
            print("[DEBUG] status", r.status_code, "bytes", len(r.content))
        r.raise_for_status()
        return r.text

def http_get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    debug: bool = False,
) -> Dict[str, Any]:
    hdrs = {"User-Agent": DEFAULT_HEADER, **(headers or {})}
    if debug:
        print("[DEBUG] GET", url, "params=", params or {})
    with httpx.Client(headers=hdrs, timeout=timeout, follow_redirects=True) as c:
        r = c.get(url, params=params)
        if debug:
            print("[DEBUG] status", r.status_code, "bytes", len(r.content))
        r.raise_for_status()
        return r.json()
