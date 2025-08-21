from __future__ import annotations
import math
from typing import Any, Dict, List

import gradio as gr

# --- parsers ---
from papers.arxiv_parser import arxiv_search
from papers.crossref_parser import crossref_search
from papers.openalex_parser import openalex_search
from papers.hf_parser import hf_search
from news.news_parser import news_search
from videos.youtube_api import youtube_search

# --- UI kit ---
from ui.components import (
    DATE_WINDOWS, DEFAULT_MAX_TOTAL, DEFAULT_WEIGHTS, DEFAULT_PAPER_SPLIT,
    build_header, build_top_controls, render_cards,
    on_papers_change, on_videos_change, on_news_change,
    on_source_pct_change,
)

# ---------------- Allocation helpers ----------------
def _allocate_counts(total: int, ratios: Dict[str, int], keys: List[str]) -> Dict[str, int]:
    s = sum(max(0, ratios.get(k, 0)) for k in keys)
    base = [0.0 for _ in keys] if s <= 0 else [total * (max(0, ratios.get(k, 0)) / 100.0) for k in keys]
    floors = [int(math.floor(x)) for x in base]
    rem = total - sum(floors)
    if rem:
        fracs = sorted([(i, base[i] - floors[i]) for i in range(len(keys))], key=lambda x: x[1], reverse=True)
        for i in range(rem):
            floors[fracs[i % len(keys)][0]] += 1
    return {k: floors[i] for i, k in enumerate(keys)}

def _apply_weights_global(weights: Dict[str, float]) -> None:
    import importlib
    mods = [
        "papers.arxiv_parser",
        "papers.crossref_parser",
        "papers.openalex_parser",
        "papers.hf_parser",
        "news.news_parser",
        "videos.youtube_api",
    ]
    # dst var > slider key
    name_maps = [
        ("W_TITLE_BASE", "W_TITLE_BASE"),
        ("W_ABS_BASE",   "W_ABS_BASE"),
        ("W_TITLE_PRF",  "W_TITLE_PRF"),
        ("W_ABS_PRF",    "W_ABS_PRF"),
        ("W_RECENCY",    "W_RECENCY"),
        ("W_SUMM_BASE",  "W_ABS_BASE"),
        ("W_SUMM_PRF",   "W_ABS_PRF"),
        ("W_DESC_BASE",  "W_ABS_BASE"),
        ("W_DESC_PRF",   "W_ABS_PRF"),
    ]
    for m in mods:
        try:
            mod = importlib.import_module(m)
            for dst_name, src_key in name_maps:
                v = weights.get(src_key)
                if v is not None and hasattr(mod, dst_name):
                    setattr(mod, dst_name, float(v))
        except Exception:
            pass

# ---------------- Search pipeline (streaming) ----------------
CONF_THRESHOLD = 25.0  

def run_search(
    topic: str,
    kinds: List[str],
    window_label: str,
    max_total: int,
    papers_pct: int, videos_pct: int, news_pct: int,
    w_title_base: float, w_abs_base: float, w_title_prf: float, w_abs_prf: float, w_recency: float,
    arxiv_share: int, crossref_share: int, openalex_share: int, hf_share: int,
):
    """Streams 6 outputs: (prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html)."""

    # ---------- local helpers ----------
    def ptxt(p, label): 
        return f"**Progress:** {int(p)}% — {label}"

    CONF_THRESHOLD = 25.0  

    def _auto_scale_threshold(items, thr_pct: float = CONF_THRESHOLD) -> float:
        if not items:
            return thr_pct
        mx = max(float(x.get("confidence", 0.0)) for x in items)
        return thr_pct / 100.0 if mx <= 1.0 else thr_pct

    def _filter_and_sort(items, max_n: int, thr_pct: float = CONF_THRESHOLD):
        if not items:
            return []
        thr = _auto_scale_threshold(items, thr_pct)
        kept = [x for x in items if float(x.get("confidence", 0.0)) >= thr]
        kept.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return kept[: max_n]

    # ---------- initial state ----------
    prog_papers = ptxt(0, "Idle")
    prog_videos = ptxt(0, "Idle")
    prog_news   = ptxt(0, "Idle")
    papers_html = videos_html = news_html = ""
    yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    # ---------- apply weights globally ----------
    _apply_weights_global(dict(
        W_TITLE_BASE=w_title_base, W_ABS_BASE=w_abs_base,
        W_TITLE_PRF=w_title_prf, W_ABS_PRF=w_abs_prf,
        W_RECENCY=w_recency,
    ))

    # ---------- time window & top allocation ----------
    days = DATE_WINDOWS.get(window_label, 30)
    top_alloc = _allocate_counts(
        int(max_total),
        {"Papers": papers_pct, "Videos": videos_pct, "News": news_pct},
        ["Papers", "Videos", "News"]
    )

    # =================== PAPERS ===================
    if "Papers" in kinds and top_alloc["Papers"] > 0:
        prog_papers = ptxt(10, "Fetching papers…")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        per_alloc = _allocate_counts(
            top_alloc["Papers"],
            {"arxiv": arxiv_share, "crossref": crossref_share, "openalex": openalex_share, "hf": hf_share},
            ["arxiv", "crossref", "openalex", "hf"]
        )

        papers: List[Dict[str, Any]] = []

        # arXiv
        if per_alloc["arxiv"] > 0:
            papers += arxiv_search(topic, days=days, max_results=per_alloc["arxiv"], debug=False)
            prog_papers = ptxt(30, "arXiv done…")
            papers_html = render_cards(_filter_and_sort(papers, top_alloc["Papers"]), kind="papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        # Crossref
        if per_alloc["crossref"] > 0:
            more = crossref_search(topic, days=days, max_results=per_alloc["crossref"], debug=False)
            papers += (more or [])
            prog_papers = ptxt(50, "Crossref done…")
            papers_html = render_cards(_filter_and_sort(papers, top_alloc["Papers"]), kind="papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        # OpenAlex
        if per_alloc["openalex"] > 0:
            try:
                more = openalex_search(topic, days=days, max_results=per_alloc["openalex"], debug=False)
                papers += (more or [])
            except Exception:
                pass
            prog_papers = ptxt(70, "OpenAlex done…")
            papers_html = render_cards(_filter_and_sort(papers, top_alloc["Papers"]), kind="papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        # HF Papers
        if per_alloc["hf"] > 0:
            more = hf_search(topic, days=days, max_results=per_alloc["hf"], debug=False)
            papers += (more or [])
            prog_papers = ptxt(85, "HF Papers done…")
            papers_html = render_cards(_filter_and_sort(papers, top_alloc["Papers"]), kind="papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        # finalize papers
        papers_final = _filter_and_sort(papers, top_alloc["Papers"])
        papers_html = render_cards(papers_final, kind="papers")
        prog_papers = ptxt(100, "Papers ready")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html
    else:
        prog_papers = ptxt(100, "Papers skipped")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    # =================== VIDEOS ===================
    if "Videos" in kinds and top_alloc["Videos"] > 0:
        prog_videos = ptxt(20, "Fetching videos…")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        raw_videos = youtube_search(topic, days=max(7, days), max_results=top_alloc["Videos"], debug=False)
        prog_videos = ptxt(60, "YouTube ranked…")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        videos = _filter_and_sort(raw_videos or [], top_alloc["Videos"])
        videos_html = render_cards(videos, kind="videos")
        prog_videos = ptxt(100, "Videos ready")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html
    else:
        prog_videos = ptxt(100, "Videos skipped")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    # =================== NEWS ===================
    if "News" in kinds and top_alloc["News"] > 0:
        prog_news = ptxt(20, "Fetching news…")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        raw_news = news_search(topic, days=max(7, days), max_results=top_alloc["News"], debug=False)
        prog_news = ptxt(60, "News ranked…")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        news_items = _filter_and_sort(raw_news or [], top_alloc["News"])
        news_html = render_cards(news_items, kind="news")
        prog_news = ptxt(100, "News ready")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html
    else:
        prog_news = ptxt(100, "News skipped")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html


# ---------------- UI ----------------
with gr.Blocks(title="Topic Radar", analytics_enabled=False) as demo:
    # Header
    build_header()

    # Controls
    (
        topic, kinds, window, max_total,
        papers_pct, videos_pct, news_pct,
        w_title_base, w_abs_base, w_title_prf, w_abs_prf, w_recency,
        arxiv_share, crossref_share, openalex_share, hf_share,
        btn_go, btn_clear
    ) = build_top_controls(DEFAULT_MAX_TOTAL, DATE_WINDOWS, DEFAULT_WEIGHTS, DEFAULT_PAPER_SPLIT)

    # live % rebalance 
    papers_pct.change(on_papers_change, [papers_pct, videos_pct, news_pct], [papers_pct, videos_pct, news_pct], queue=False, show_progress=False)
    videos_pct.change(on_videos_change, [videos_pct, papers_pct, news_pct], [papers_pct, videos_pct, news_pct], queue=False, show_progress=False)
    news_pct.change(  on_news_change,   [news_pct,   papers_pct, videos_pct], [papers_pct, videos_pct, news_pct], queue=False, show_progress=False)

    # live % rebalance (4 per-source)
    arxiv_share.change(
        lambda *args: on_source_pct_change("arxiv", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    crossref_share.change(
        lambda *args: on_source_pct_change("crossref", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    openalex_share.change(
        lambda *args: on_source_pct_change("openalex", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    hf_share.change(
        lambda *args: on_source_pct_change("hf", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    # Results 
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Papers")
            prog_papers = gr.Markdown("**Progress:** 0% — Idle")
            papers_out = gr.HTML("")

        with gr.Column(scale=1):
            gr.Markdown("### Videos")
            prog_videos = gr.Markdown("**Progress:** 0% — Idle")
            videos_out = gr.HTML("")

        with gr.Column(scale=1):
            gr.Markdown("### News")
            prog_news = gr.Markdown("**Progress:** 0% — Idle")
            news_out = gr.HTML("")


    # Buttons
    btn_clear.click(
        fn=lambda: (
            "", ["Papers", "News", "Videos"], "Last 30 days", DEFAULT_MAX_TOTAL,
            70, 15, 15,
            DEFAULT_WEIGHTS["W_TITLE_BASE"], DEFAULT_WEIGHTS["W_ABS_BASE"],
            DEFAULT_WEIGHTS["W_TITLE_PRF"], DEFAULT_WEIGHTS["W_ABS_PRF"], DEFAULT_WEIGHTS["W_RECENCY"],
            DEFAULT_PAPER_SPLIT["arxiv"], DEFAULT_PAPER_SPLIT["crossref"], DEFAULT_PAPER_SPLIT["openalex"], DEFAULT_PAPER_SPLIT["hf"],
            # далі — шість виходів:
            "**Progress:** 0% — Idle", "", 
            "**Progress:** 0% — Idle", "", 
            "**Progress:** 0% — Idle", ""
        ),
        inputs=[],
        outputs=[
            topic, kinds, window, max_total,
            papers_pct, videos_pct, news_pct,
            w_title_base, w_abs_base, w_title_prf, w_abs_prf, w_recency,
            arxiv_share, crossref_share, openalex_share, hf_share,
            prog_papers, papers_out, prog_videos, videos_out, prog_news, news_out
        ],
        queue=False, show_progress=False
    )
    
    btn_go.click(
        fn=run_search,
        inputs=[
            topic, kinds, window, max_total,
            papers_pct, videos_pct, news_pct,
            w_title_base, w_abs_base, w_title_prf, w_abs_prf, w_recency,
            arxiv_share, crossref_share, openalex_share, hf_share,
        ],
        outputs=[prog_papers, papers_out, prog_videos, videos_out, prog_news, news_out],
        api_name="search",
        queue=True,
        show_progress=False
    )

if __name__ == "__main__":
    demo.launch()
