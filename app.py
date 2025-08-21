# app.py
from __future__ import annotations
from typing import Any, Dict, List

import gradio as gr

from core.alloc import normalize_weights, alloc_by_shares, per_source_split
from core.utils import percent_bar
from core.render import render_papers_html, render_videos_html, render_news_html

# -------------------- Soft imports for backends --------------------
ARXIV = CROSSREF = OPENALEX = HF = NEWS = YTV = None
missing_sources: List[str] = []

try:
    import papers.arxiv_parser as ARXIV
except Exception:
    missing_sources.append("arXiv")
try:
    import papers.crossref_parser as CROSSREF
except Exception:
    missing_sources.append("Crossref")
try:
    import papers.openalex_parser as OPENALEX
except Exception:
    missing_sources.append("OpenAlex")
try:
    import papers.hf_parser as HF
except Exception:
    missing_sources.append("HuggingFace Papers")
try:
    import news.news_parser as NEWS
except Exception:
    missing_sources.append("News")
try:
    import videos.youtube_api as YTV
except Exception:
    missing_sources.append("YouTube")

# -------------------- App constants --------------------
APP_TITLE = "📡 Topic Radar - Papers • News • Videos"
DATE_WINDOWS = {"Last 3 days": 3, "Last 7 days": 7, "Last 30 days": 30, "Last 365 days": 365}
DEFAULT_MAX_TOTAL = 30

DEFAULT_PAPER_SPLIT = dict(arxiv=50, crossref=20, openalex=20, hf=10)
DEFAULT_WEIGHTS = dict(W_TITLE_BASE=0.55, W_ABS_BASE=0.20, W_TITLE_PRF=0.15, W_ABS_PRF=0.05, W_RECENCY=0.05)

# Confidence threshold (25). If confidences are 0..1, we auto-interpret as 0.25.
CONF_THRESHOLD = 25.0

# -------------------- Backends helpers --------------------
def set_module_weights(weights: Dict[str, float]) -> List[str]:
    touched = []
    for mod in (ARXIV, CROSSREF, OPENALEX, HF, NEWS, YTV):
        if not mod:
            continue
        for k, v in weights.items():
            if hasattr(mod, k):
                setattr(mod, k, float(v))
                touched.append(getattr(mod, "__name__", "module"))
    return sorted(set(touched))

def _auto_scale_threshold(items: List[Dict[str, Any]], thr_pct: float = CONF_THRESHOLD) -> float:
    if not items:
        return thr_pct
    mx = max(float(x.get("confidence", 0.0)) for x in items)
    return thr_pct / 100.0 if mx <= 1.0 else thr_pct

def _filter_by_conf(items: List[Dict[str, Any]], thr_pct: float = CONF_THRESHOLD) -> List[Dict[str, Any]]:
    if not items:
        return []
    thr = _auto_scale_threshold(items, thr_pct)
    return [x for x in items if float(x.get("confidence", 0.0)) >= thr]

def select_papers_by_conf(papers_by_src: Dict[str, List[Dict[str, Any]]], per_source: Dict[str, int]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for src, quota in per_source.items():
        lst = _filter_by_conf(papers_by_src.get(src, []), CONF_THRESHOLD)
        lst = sorted(lst, key=lambda x: x.get("confidence", 0.0), reverse=True)
        selected.extend(lst[:max(quota, 0)])
    selected.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return selected

# -------------------- UI helpers: percentage logic --------------------
def _rebalance_for_enabled(kinds: List[str], p: int, v: int, n: int):
    enabled = {"Papers": "Papers" in kinds, "Videos": "Videos" in kinds, "News": "News" in kinds}
    vals = {"Papers": int(p), "Videos": int(v), "News": int(n)}

    # zero disabled
    for k, en in enabled.items():
        if not en:
            vals[k] = 0

    active = [k for k, en in enabled.items() if en]
    if not active:
        return (gr.update(value=0, interactive=False),
                gr.update(value=0, interactive=False),
                gr.update(value=0, interactive=False))

    total = sum(vals[k] for k in active)
    if total == 0:
        eq = 100 // len(active)
        rem = 100 - eq * len(active)
        for k in active:
            vals[k] = eq
        for k in active[:rem]:
            vals[k] += 1
    else:
        for k in active:
            vals[k] = round(100.0 * vals[k] / total)
        drift = 100 - sum(vals[k] for k in active)
        if drift != 0:
            order = sorted(active, key=lambda k: vals[k], reverse=(drift < 0))
            i = 0
            while drift != 0 and i < len(order):
                vals[order[i]] += 1 if drift > 0 else -1
                drift += -1 if drift > 0 else 1
                i = (i + 1) % len(order)

    return (gr.update(value=vals["Papers"], interactive=enabled["Papers"]),
            gr.update(value=vals["Videos"], interactive=enabled["Videos"]),
            gr.update(value=vals["News"],   interactive=enabled["News"]))

def _on_kinds_change(kinds: List[str], p: int, v: int, n: int):
    return _rebalance_for_enabled(kinds, p, v, n)

def _on_pct_change(which: str, kinds: List[str], p: int, v: int, n: int):
    enabled = {"Papers": "Papers" in kinds, "Videos": "Videos" in kinds, "News": "News" in kinds}
    vals = {"Papers": int(p), "Videos": int(v), "News": int(n)}
    vals[which] = max(0, min(100, vals[which]))
    for k, en in enabled.items():
        if not en: vals[k] = 0
    active = [k for k, en in enabled.items() if en]
    if not active:
        return (gr.update(value=0, interactive=False),
                gr.update(value=0, interactive=False),
                gr.update(value=0, interactive=False))
    if len(active) == 1:
        only = active[0]
        vals[only] = 100
        for k in vals:
            if k != only: vals[k] = 0
        return (gr.update(value=vals["Papers"], interactive=enabled["Papers"]),
                gr.update(value=vals["Videos"], interactive=enabled["Videos"]),
                gr.update(value=vals["News"],   interactive=enabled["News"]))

    others = [k for k in active if k != which]
    remainder = 100 - vals[which]
    current_sum_others = sum(vals[k] for k in others)
    if current_sum_others <= 0:
        base = remainder // len(others)
        rem = remainder - base * len(others)
        for k in others: vals[k] = base
        for k in others[:rem]: vals[k] += 1
    else:
        scaled = [remainder * (vals[k] / current_sum_others) for k in others]
        alloc = [int(x) for x in scaled]
        used = sum(alloc)
        left = remainder - used
        rema = sorted([(i, scaled[i] - alloc[i]) for i in range(len(others))], key=lambda t: t[1], reverse=True)
        for j in range(left): alloc[rema[j % len(others)][0]] += 1
        for i, k in enumerate(others): vals[k] = alloc[i]

    return (gr.update(value=vals["Papers"], interactive=enabled["Papers"]),
            gr.update(value=vals["Videos"], interactive=enabled["Videos"]),
            gr.update(value=vals["News"],   interactive=enabled["News"]))

# ----- per-source mix (arxiv/crossref/openalex/hf) auto 100% -----
def _on_source_pct_change(which: str, a: int, c: int, o: int, h: int):
    vals = {"arxiv": int(a), "crossref": int(c), "openalex": int(o), "hf": int(h)}
    vals[which] = max(0, min(100, vals[which]))
    total = sum(vals.values())
    if total == 0:
        # split equally
        eq = 25
        vals = {"arxiv": eq, "crossref": eq, "openalex": eq, "hf": 100 - 3*eq}
        return (gr.update(value=vals["arxiv"]),
                gr.update(value=vals["crossref"]),
                gr.update(value=vals["openalex"]),
                gr.update(value=vals["hf"]))
    # preserve 'which', scale the rest into remaining
    others = [k for k in vals if k != which]
    remainder = 100 - vals[which]
    sum_others = sum(vals[k] for k in others)
    if sum_others <= 0:
        base = remainder // 3
        rem = remainder - base * 3
        for k in others: vals[k] = base
        for k in others[:rem]: vals[k] += 1
    else:
        scaled = [remainder * (vals[k] / sum_others) for k in others]
        alloc = [int(x) for x in scaled]
        used = sum(alloc)
        left = remainder - used
        rema = sorted([(i, scaled[i] - alloc[i]) for i in range(3)], key=lambda t: t[1], reverse=True)
        for j in range(left): alloc[rema[j % 3][0]] += 1
        for i, k in enumerate(others): vals[k] = alloc[i]

    return (gr.update(value=vals["arxiv"]),
            gr.update(value=vals["crossref"]),
            gr.update(value=vals["openalex"]),
            gr.update(value=vals["hf"]))

# -------------------- Search runner (with smoother progress) --------------------
def run_search(
    topic: str,
    kinds: List[str],
    window_label: str,
    max_total: int,
    papers_pct: int, videos_pct: int, news_pct: int,
    w_title_base: float, w_abs_base: float, w_title_prf: float, w_abs_prf: float, w_recency: float,
    arxiv_share: int, crossref_share: int, openalex_share: int, hf_share: int,
):
    days = DATE_WINDOWS.get(window_label, 30)

    weights = dict(
        W_TITLE_BASE=float(w_title_base),
        W_ABS_BASE=float(w_abs_base),
        W_TITLE_PRF=float(w_title_prf),
        W_ABS_PRF=float(w_abs_prf),
        W_RECENCY=float(w_recency),
    )
    set_module_weights(weights)

    include_papers = "Papers" in kinds
    include_videos = "Videos" in kinds
    include_news = "News" in kinds

    cats, shares = [], []
    if include_papers: cats.append("papers"); shares.append(float(papers_pct))
    if include_videos: cats.append("videos"); shares.append(float(videos_pct))
    if include_news:   cats.append("news");   shares.append(float(news_pct))

    if not cats:
        empty = percent_bar(0, "Papers"), "", percent_bar(0, "Videos"), "", percent_bar(0, "News"), ""
        yield empty
        return

    shares = normalize_weights(shares)
    alloc = alloc_by_shares(int(max_total), shares)
    per_cat = dict(zip(cats, alloc))
    target_papers = per_cat.get("papers", 0)
    ps = per_source_split(target_papers, float(arxiv_share), float(crossref_share), float(openalex_share), float(hf_share))

    # placeholders
    prog_papers = percent_bar(0, "Papers")
    prog_videos = percent_bar(0, "Videos")
    prog_news = percent_bar(0, "News")
    papers_html = videos_html = news_html = ""
    yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    # ----- PAPERS -----
    if include_papers and target_papers > 0:
        papers_by_src = {"arxiv": [], "crossref": [], "openalex": [], "hf": []}
        prog = 10
        prog_papers = percent_bar(prog, "Papers")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        active_sources = [k for k, v in ps.items() if v > 0 and k in ("arxiv", "crossref", "openalex", "hf")]
        steps_per_source = 80 / max(len(active_sources), 1)  # leave 10% for final render

        for src in ("arxiv", "crossref", "openalex", "hf"):
            if ps.get(src, 0) <= 0:
                continue

            prog = min(prog + steps_per_source * 0.25, 99)
            prog_papers = percent_bar(prog, "Papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

            if src == "arxiv" and ARXIV:
                papers_by_src["arxiv"] = ARXIV.arxiv_search(topic, days=days, max_results=ps["arxiv"], debug=False) or []
            elif src == "crossref" and CROSSREF:
                papers_by_src["crossref"] = CROSSREF.crossref_search(topic, days=days, max_results=ps["crossref"], debug=False) or []
            elif src == "openalex" and OPENALEX:
                papers_by_src["openalex"] = OPENALEX.openalex_search(topic, days=days, max_results=ps["openalex"], debug=False) or []
            elif src == "hf" and HF:
                papers_by_src["hf"] = HF.hf_search(topic, days=days, max_results=ps["hf"], debug=False) or []

            prog = min(prog + steps_per_source * 0.5, 99)
            prog_papers = percent_bar(prog, "Papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

            prog = min(prog + steps_per_source * 0.25, 99)
            prog_papers = percent_bar(prog, "Papers")
            yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        papers_selected = select_papers_by_conf(papers_by_src, ps)
        papers_html = render_papers_html(papers_selected)

        prog_papers = percent_bar(100, "Papers")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    # ----- VIDEOS -----
    if include_videos and per_cat.get("videos", 0) > 0:
        prog_videos = percent_bar(20, "Videos")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        vids: List[Dict[str, Any]] = []
        if YTV:
            raw = YTV.youtube_search(topic, days=days, max_results=per_cat["videos"], debug=False) or []
            vids = _filter_by_conf(raw, CONF_THRESHOLD)
            vids = sorted(vids, key=lambda x: x.get("confidence", 0.0), reverse=True)[:per_cat["videos"]]

        prog_videos = percent_bar(60, "Videos")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        videos_html = render_videos_html(vids)
        prog_videos = percent_bar(100, "Videos")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    # ----- NEWS -----
    if include_news and per_cat.get("news", 0) > 0:
        prog_news = percent_bar(20, "News")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        nws: List[Dict[str, Any]] = []
        if NEWS:
            raw = NEWS.news_search(topic, days=days, max_results=per_cat["news"], debug=False) or []
            nws = _filter_by_conf(raw, CONF_THRESHOLD)
            nws = sorted(nws, key=lambda x: x.get("confidence", 0.0), reverse=True)[:per_cat["news"]]

        prog_news = percent_bar(60, "News")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

        news_html = render_news_html(nws)
        prog_news = percent_bar(100, "News")
        yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

    yield prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html

# -------------------- Clear --------------------
def clear_all():
    return (
        "",                                 # topic
        ["Papers", "News", "Videos"],       # kinds
        "Last 30 days",                     # window
        DEFAULT_MAX_TOTAL,                  # max_total
        70, 15, 15,                         # papers_pct, videos_pct, news_pct
        DEFAULT_WEIGHTS["W_TITLE_BASE"],
        DEFAULT_WEIGHTS["W_ABS_BASE"],
        DEFAULT_WEIGHTS["W_TITLE_PRF"],
        DEFAULT_WEIGHTS["W_ABS_PRF"],
        DEFAULT_WEIGHTS["W_RECENCY"],
        DEFAULT_PAPER_SPLIT["arxiv"],
        DEFAULT_PAPER_SPLIT["crossref"],
        DEFAULT_PAPER_SPLIT["openalex"],
        DEFAULT_PAPER_SPLIT["hf"],
        percent_bar(0, "Papers"), "", percent_bar(0, "Videos"), "", percent_bar(0, "News"), ""  # outputs
    )

# -------------------- Theme & CSS --------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_text_color="*neutral_50",
    block_title_text_weight="700",
    block_title_text_size="20px",
)

CUSTOM_CSS = """
:root {
  --bg: #0b1020;
  --panel: rgba(255,255,255,0.06);
  --panel-strong: rgba(255,255,255,0.10);
  --border: rgba(255,255,255,0.12);
  --shadow: 0 10px 30px rgba(0,0,0,0.35);
}
body { background: radial-gradient(1200px 800px at 5% -10%, rgba(99,102,241,.16) 0%, rgba(99,102,241,0) 45%),
                    radial-gradient(1000px 700px at 110% 20%, rgba(56,189,248,.14) 0%, rgba(56,189,248,0) 40%),
                    linear-gradient(180deg, #0b1020, #0c1226); }

.tr-container { max-width: 1300px; margin: 0 auto; }
.tr-card { background: var(--panel); border: 1px solid var(--border); backdrop-filter: blur(8px); border-radius: 16px; box-shadow: var(--shadow); padding: 10px; }
.tr-sticky-top { position: sticky; top: 0; z-index: 20; background: linear-gradient(180deg, rgba(11,16,32,0.85), rgba(11,16,32,0.55) 60%, rgba(11,16,32,0)); padding-bottom: 10px; margin-bottom: 10px; backdrop-filter: blur(6px); }
.tr-title { font-weight: 800; letter-spacing: .2px; font-size: 26px; margin: 6px 0 2px 0;
  background: linear-gradient(90deg, #dbe3ff 0%, #b1c6ff 50%, #9ee6ff 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }

.tr-progress { position: relative; height: 12px; background: rgba(255,255,255,0.08); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 10px; }
.tr-progress__bar { height: 100%; background: linear-gradient(90deg, rgba(99,102,241,0.9), rgba(56,189,248,0.9)); box-shadow: inset 0 0 8px rgba(255,255,255,0.25); transition: width .18s ease; }
.tr-progress__label { position: absolute; width: 100%; top: -22px; text-align: right; font-size: 12px; color: #cdd6ff; }

.tr-card-row { border: 1px solid var(--border); background: rgba(255,255,255,0.03); border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
.tr-card-row__head { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
.tr-card-row__title a { color:#e6eeff; text-decoration:none; }
.tr-card-row__title a:hover { text-decoration:underline; }
.tr-card-row__meta { font-size:12px; color:#b9c3ff; opacity:.9; }
.tr-card-row__details summary { cursor:pointer; color:#cbd5ff; }
.tr-card-row__body { color:#e9edff; opacity:.95; padding-top:6px; }
.tr-links a { color:#9ee6ff; }
.tr-empty { color:#cdd6ff; opacity:.8; padding:6px 2px; }

.tr-source-title { font-size: 13px; color: #cbd5ff; opacity:.95; margin: 10px 0 2px 0; }
.tr-source-row { margin-top: 6px; }

"""

# -------------------- UI --------------------
with gr.Blocks(title="Topic Radar", theme=theme, css=CUSTOM_CSS, fill_height=True, analytics_enabled=False) as demo:
    with gr.Column(elem_classes=["tr-container", "tr-sticky-top"]):
        # Title only (subtitle removed)
        gr.HTML(f"<div class='tr-title'>{APP_TITLE}</div>")
        with gr.Group(elem_classes=["tr-card"]):
            with gr.Row():
                topic = gr.Textbox(label="Topic", placeholder="e.g., computer vision industrial automation, SLAM, OPC UA security", autofocus=True)
                kinds = gr.CheckboxGroup(choices=["Papers", "News", "Videos"], value=["Papers","News","Videos"], label="Include")
                window = gr.Radio(choices=list(DATE_WINDOWS.keys()), value="Last 30 days", label="Date window")
                max_total = gr.Slider(10, 100, value=DEFAULT_MAX_TOTAL, step=1, label="Max results total")

            with gr.Row():
                papers_pct = gr.Slider(0, 100, value=70, step=1, label="Papers %")
                videos_pct = gr.Slider(0, 100, value=15, step=1, label="Videos %")
                news_pct   = gr.Slider(0, 100, value=15, step=1, label="News %")

            with gr.Accordion("Advanced (weights & source mix)", open=False):
                with gr.Row(elem_classes=["tr-weights"]):
                    w_title_base = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["W_TITLE_BASE"], step=0.01, label="Title (base)")
                    w_abs_base   = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["W_ABS_BASE"], step=0.01, label="Abstract/Desc (base)")
                    w_title_prf  = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["W_TITLE_PRF"], step=0.01, label="Title (PRF)")
                    w_abs_prf    = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["W_ABS_PRF"], step=0.01, label="Abstract/Desc (PRF)")
                    w_recency    = gr.Slider(0, 1, value=DEFAULT_WEIGHTS["W_RECENCY"], step=0.01, label="Recency")

                # Small label + a bit of space before per-source sliders
                gr.Markdown("**Papers per-source mix (% of the Papers quota)**", elem_classes=["tr-source-title"])
                with gr.Row(elem_classes=["tr-source-row"]):
                    arxiv_share    = gr.Slider(0, 100, value=DEFAULT_PAPER_SPLIT["arxiv"], step=1, label="arXiv %")
                    crossref_share = gr.Slider(0, 100, value=DEFAULT_PAPER_SPLIT["crossref"], step=1, label="Crossref %")
                    openalex_share = gr.Slider(0, 100, value=DEFAULT_PAPER_SPLIT["openalex"], step=1, label="OpenAlex %")
                    hf_share       = gr.Slider(0, 100, value=DEFAULT_PAPER_SPLIT["hf"], step=1, label="HuggingFace %")

            with gr.Row():
                btn_go = gr.Button("🔎 Search", variant="primary")
                btn_clear = gr.Button("↺ Clear")

    with gr.Column(elem_classes=["tr-container"]):
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["tr-card"]):
                gr.Markdown("### Papers")
                prog_papers = gr.HTML(percent_bar(0, "Papers"))
                papers_html = gr.HTML("")
            with gr.Column(scale=1, elem_classes=["tr-card"]):
                gr.Markdown("### Videos")
                prog_videos = gr.HTML(percent_bar(0, "Videos"))
                videos_html = gr.HTML("")
            with gr.Column(scale=1, elem_classes=["tr-card"]):
                gr.Markdown("### News")
                prog_news = gr.HTML(percent_bar(0, "News"))
                news_html = gr.HTML("")

    # live constraints for Papers/News/Videos %
    kinds.change(
        _on_kinds_change,
        inputs=[kinds, papers_pct, videos_pct, news_pct],
        outputs=[papers_pct, videos_pct, news_pct],
        queue=False,
    )
    papers_pct.change(
        lambda *args: _on_pct_change("Papers", *args),
        inputs=[kinds, papers_pct, videos_pct, news_pct],
        outputs=[papers_pct, videos_pct, news_pct],
        queue=False,
    )
    videos_pct.change(
        lambda *args: _on_pct_change("Videos", *args),
        inputs=[kinds, papers_pct, videos_pct, news_pct],
        outputs=[papers_pct, videos_pct, news_pct],
        queue=False,
    )
    news_pct.change(
        lambda *args: _on_pct_change("News", *args),
        inputs=[kinds, papers_pct, videos_pct, news_pct],
        outputs=[papers_pct, videos_pct, news_pct],
        queue=False,
    )

    # auto-100% logic for per-source sliders
    arxiv_share.change(
        lambda *args: _on_source_pct_change("arxiv", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    crossref_share.change(
        lambda *args: _on_source_pct_change("crossref", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    openalex_share.change(
        lambda *args: _on_source_pct_change("openalex", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )
    hf_share.change(
        lambda *args: _on_source_pct_change("hf", *args),
        inputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[arxiv_share, crossref_share, openalex_share, hf_share],
        queue=False,
    )

    btn_go.click(
        run_search,
        inputs=[topic, kinds, window, max_total,
                papers_pct, videos_pct, news_pct,
                w_title_base, w_abs_base, w_title_prf, w_abs_prf, w_recency,
                arxiv_share, crossref_share, openalex_share, hf_share],
        outputs=[prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html],
        show_progress=False,
        queue=True,
    )

    btn_clear.click(
        clear_all,
        inputs=None,
        outputs=[topic, kinds, window, max_total,
                 papers_pct, videos_pct, news_pct,
                 w_title_base, w_abs_base, w_title_prf, w_abs_prf, w_recency,
                 arxiv_share, crossref_share, openalex_share, hf_share,
                 prog_papers, papers_html, prog_videos, videos_html, prog_news, news_html],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
