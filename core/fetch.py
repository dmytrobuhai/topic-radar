# core/fetch.py
from __future__ import annotations
from typing import Any, Dict, List

# Soft-import the source modules; keep track of missing ones.
missing_sources: List[str] = []

try:
    import papers.arxiv_parser as ARXIV
except Exception:
    ARXIV = None
    missing_sources.append("arXiv")

try:
    import papers.crossref_parser as CROSSREF
except Exception:
    CROSSREF = None
    missing_sources.append("Crossref")

try:
    import papers.openalex_parser as OPENALEX
except Exception:
    OPENALEX = None
    missing_sources.append("OpenAlex")

try:
    import papers.hf_parser as HF
except Exception:
    HF = None
    missing_sources.append("HuggingFace Papers")

try:
    import news.news_parser as NEWS
except Exception:
    NEWS = None
    missing_sources.append("News")

try:
    import videos.youtube_api as YTV
except Exception:
    YTV = None
    missing_sources.append("YouTube")

def set_module_weights(weights: Dict[str, float]) -> List[str]:
    """Push UI weights into all available modules (if attrs exist)."""
    touched = []
    for mod in (ARXIV, CROSSREF, OPENALEX, HF, NEWS, YTV):
        if not mod:
            continue
        for k, v in weights.items():
            if hasattr(mod, k):
                setattr(mod, k, float(v))
                touched.append(getattr(mod, "__name__", "module"))
    return sorted(set(touched))

def fetch_papers(topic: str, days: int, per_source: Dict[str, int], debug: bool) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"arxiv": [], "crossref": [], "openalex": [], "hf": []}
    if ARXIV and per_source.get("arxiv", 0) > 0:
        out["arxiv"] = ARXIV.arxiv_search(topic, days=days, max_results=per_source["arxiv"], debug=debug) or []
    if CROSSREF and per_source.get("crossref", 0) > 0:
        out["crossref"] = CROSSREF.crossref_search(topic, days=days, max_results=per_source["crossref"], debug=debug) or []
    if OPENALEX and per_source.get("openalex", 0) > 0:
        out["openalex"] = OPENALEX.openalex_search(topic, days=days, max_results=per_source["openalex"], debug=debug) or []
    if HF and per_source.get("hf", 0) > 0:
        out["hf"] = HF.hf_search(topic, days=days, max_results=per_source["hf"], debug=debug) or []
    return out

def select_papers(papers_by_src: Dict[str, List[Dict[str, Any]]], per_source: Dict[str, int]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for src, quota in per_source.items():
        lst = sorted(papers_by_src.get(src, []), key=lambda x: x.get("confidence", 0.0), reverse=True)
        selected.extend(lst[:max(quota, 0)])
    selected.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return selected

def fetch_videos(topic: str, days: int, want: int, debug: bool) -> List[Dict[str, Any]]:
    if not YTV or want <= 0:
        return []
    return YTV.youtube_search(topic, days=days, max_results=want, debug=debug) or []

def fetch_news(topic: str, days: int, want: int, debug: bool) -> List[Dict[str, Any]]:
    if not NEWS or want <= 0:
        return []
    return NEWS.news_search(topic, days=days, max_results=want, debug=debug) or []
