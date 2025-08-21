from __future__ import annotations
import html
from typing import Any, Dict, List, Tuple

import gradio as gr

# ---------------- Constants ----------------
APP_TITLE = "📡 Topic Radar — Papers • News • Videos"

DATE_WINDOWS = {
    "Last 3 days": 3,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last year": 365,
}

PREVIEW_LEN = 100

DEFAULT_MAX_TOTAL = 50

DEFAULT_WEIGHTS = {
    "W_TITLE_BASE": 0.55,
    "W_ABS_BASE":   0.20,
    "W_TITLE_PRF":  0.15,
    "W_ABS_PRF":    0.05,
    "W_RECENCY":    0.05,
}

DEFAULT_PAPER_SPLIT = {
    "arxiv": 50,
    "crossref": 20,
    "openalex": 20,
    "hf": 10,
}

# ---------------- UI builders ----------------
def build_header():
    gr.Markdown(f"# {APP_TITLE}")

def build_top_controls(default_max_total, date_windows, default_weights, default_split):
    with gr.Group():
        with gr.Row():
            topic = gr.Textbox(
                label="Topic",
                placeholder="e.g., computer vision industrial automation, SLAM, OPC UA security",
                autofocus=True
            )
            kinds = gr.CheckboxGroup(
                choices=["Papers", "News", "Videos"],
                value=["Papers","News","Videos"],
                label="Include"
            )
            window = gr.Radio(
                choices=list(date_windows.keys()),
                value="Last 30 days",
                label="Date window"
            )
            max_total = gr.Slider(10, 100, value=default_max_total, step=1, label="Max results total")

        with gr.Row():
            papers_pct = gr.Slider(0, 100, value=70, step=1, label="Papers %")
            videos_pct = gr.Slider(0, 100, value=15, step=1, label="Videos %")
            news_pct   = gr.Slider(0, 100, value=15, step=1, label="News %")

        with gr.Accordion("Advanced (weights & source mix)", open=False):
            with gr.Row():
                w_title_base = gr.Slider(0, 1, value=default_weights["W_TITLE_BASE"], step=0.01, label="Title (base)")
                w_abs_base   = gr.Slider(0, 1, value=default_weights["W_ABS_BASE"], step=0.01, label="Abstract/Desc (base)")
                w_title_prf  = gr.Slider(0, 1, value=default_weights["W_TITLE_PRF"], step=0.01, label="Title (PRF)")
                w_abs_prf    = gr.Slider(0, 1, value=default_weights["W_ABS_PRF"], step=0.01, label="Abstract/Desc (PRF)")
                w_recency    = gr.Slider(0, 1, value=default_weights["W_RECENCY"], step=0.01, label="Recency")

            gr.Markdown("**Papers per-source mix (% of the Papers quota)**")
            with gr.Row():
                arxiv_share    = gr.Slider(0, 100, value=default_split["arxiv"],    step=1, label="arXiv %")
                crossref_share = gr.Slider(0, 100, value=default_split["crossref"], step=1, label="Crossref %")
                openalex_share = gr.Slider(0, 100, value=default_split["openalex"], step=1, label="OpenAlex %")
                hf_share       = gr.Slider(0, 100, value=default_split["hf"],       step=1, label="HuggingFace %")

        with gr.Row():
            btn_go = gr.Button("🔎 Search", variant="primary")
            btn_clear = gr.Button("Clear")

    return (
        topic, kinds, window, max_total,
        papers_pct, videos_pct, news_pct,
        w_title_base, w_abs_base, w_title_prf, w_abs_prf, w_recency,
        arxiv_share, crossref_share, openalex_share, hf_share,
        btn_go, btn_clear
    )

# ---------------- Rendering helpers ----------------
def _safe(x: str) -> str:
    if not x:
        return ""
    return html.escape(str(x))

def _fmt(x: float) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"


def render_cards(items: List[Dict[str, Any]], kind: str) -> str:
    if not items:
        return "<em>No results.</em>"

    out = []
    for it in items:
        title = _safe(it.get("title") or it.get("display_name") or "")
        conf  = _fmt(it.get("confidence", 0.0))

        more_links = []
        if kind == "papers":
            links = it.get("links") or {}
            url   = links.get("abs") or links.get("openalex") or links.get("doi") or ""
            date  = it.get("updated") or it.get("published") or ""
            venue = it.get("venue") or it.get("source") or it.get("primary_category") or ""
            full_text = _safe(it.get("summary") or "")
            if links.get("abs"): more_links.append(f"<a href='{html.escape(links['abs'])}' target='_blank'>abstract</a>")
            if links.get("pdf"): more_links.append(f"<a href='{html.escape(links['pdf'])}' target='_blank'>pdf</a>")
            if it.get("doi_url"): more_links.append(f"<a href='{html.escape(it['doi_url'])}' target='_blank'>doi</a>")

        elif kind == "videos":
            url = it.get("url") or ""
            date = it.get("published") or ""
            venue = it.get("channel") or ""
            full_text = _safe(it.get("description") or "")
            views = (it.get("stats") or {}).get("viewCount")
            if url: more_links.append(f"<a href='{html.escape(url)}' target='_blank'>watch</a>")
            if views: more_links.append(f"{int(views):,} views")

        else:  # news
            url = it.get("url") or ""
            date = it.get("date") or ""
            venue = it.get("source") or ""
            full_text = _safe(it.get("summary") or "")
            if url: more_links.append(f"<a href='{html.escape(url)}' target='_blank'>open</a>")

        url = html.escape(url or "#")
        meta = " • ".join([p for p in [f"conf {conf}", date, venue] if p])

        preview = full_text[:PREVIEW_LEN]
        remainder = full_text[PREVIEW_LEN:]
        details_html = f"<details><summary>Details</summary><div>{full_text}</div></details>" if remainder else ""

        links_html = ("<div>" + " | ".join(more_links) + "</div>") if more_links else ""

        out.append(f"""
<div>
  <div><strong><a href="{url}" target="_blank">{title}</a></strong></div>
  <div><small>{meta}</small></div>
  <div>{preview}{'…' if remainder else ''}</div>
  {details_html}
  {links_html}
  <hr/>
</div>
""")
    return "\n".join(out)

# ---------------- Rebalance utils ----------------
def _round_and_fix(values: List[float], target_sum: int) -> List[int]:
    floors = [int(v) for v in values]
    rem = target_sum - sum(floors)
    if rem:
        fracs = sorted([(i, values[i] - floors[i]) for i in range(len(values))], key=lambda x: x[1], reverse=True)
        for i in range(rem):
            floors[fracs[i % len(values)][0]] += 1
    return floors

def rebalance_three(changed: int, other1: int, other2: int) -> Tuple[int, int, int]:
    changed = max(0, min(100, int(changed)))
    remaining = 100 - changed
    base = max(0, int(other1)) + max(0, int(other2))
    if base <= 0:
        o1 = remaining // 2
        o2 = remaining - o1
    else:
        scale = remaining / base
        o1f = max(0, int(other1)) * scale
        o2f = max(0, int(other2)) * scale
        o1, o2 = _round_and_fix([o1f, o2f], remaining)
    return changed, o1, o2

def on_source_pct_change(which: str, a: int, c: int, o: int, h: int):
    vals = {"arxiv": int(a), "crossref": int(c), "openalex": int(o), "hf": int(h)}

    fixed = max(0, min(100, vals[which]))
    vals[which] = fixed

    others = [k for k in ("arxiv", "crossref", "openalex", "hf") if k != which]
    remainder = 100 - fixed
    current_sum_others = sum(max(0, vals[k]) for k in others)

    if remainder <= 0:
        for k in others:
            vals[k] = 0
        return (vals["arxiv"], vals["crossref"], vals["openalex"], vals["hf"])

    if current_sum_others <= 0:
        base = remainder // 3
        r = remainder - base * 3
        for k in others:
            vals[k] = base
        for k in others[:r]:
            vals[k] += 1
        return (vals["arxiv"], vals["crossref"], vals["openalex"], vals["hf"])

    scaled = [remainder * (max(0, vals[k]) / current_sum_others) for k in others]
    floors = [int(x) for x in scaled]
    used = sum(floors)
    left = remainder - used

    fracs = sorted([(i, scaled[i] - floors[i]) for i in range(3)], key=lambda t: t[1], reverse=True)
    for j in range(left):
        floors[fracs[j % 3][0]] += 1

    for i, k in enumerate(others):
        vals[k] = floors[i]

    return (vals["arxiv"], vals["crossref"], vals["openalex"], vals["hf"])



def rebalance_four(changed: int, others: List[int], changed_index: int = 0) -> Tuple[int, int, int, int]:
    vals = [0, 0, 0, 0]
    inp  = [int(changed)] + [int(x) for x in others]
    a, c, o, h = inp[0], inp[1], inp[2], inp[3]
    pack = [a, c, o, h]

    pack[changed_index] = max(0, min(100, pack[changed_index]))
    remaining = 100 - pack[changed_index]

    others_vals = [pack[i] for i in range(4) if i != changed_index]
    base = sum(max(0, v) for v in others_vals)
    if base <= 0:
        split = _round_and_fix([remaining/3]*3, remaining)
    else:
        scale = remaining / base
        split = _round_and_fix([max(0, v)*scale for v in others_vals], remaining)

    pos = 0
    for i in range(4):
        if i == changed_index:
            vals[i] = pack[i]
        else:
            vals[i] = split[pos]; pos += 1
    return tuple(vals)

# ---------------- Change callback helpers  ----------------
def on_papers_change(p: int, v: int, n: int):
    return rebalance_three(changed=p, other1=v, other2=n)

def on_videos_change(v: int, p: int, n: int):
    new_v, new_p, new_n = rebalance_three(changed=v, other1=p, other2=n)
    return new_p, new_v, new_n  

def on_news_change(n: int, p: int, v: int):
    new_n, new_p, new_v = rebalance_three(changed=n, other1=p, other2=v)
    return new_p, new_v, new_n 

def on_arxiv_change(a: int, c: int, o: int, h: int):
    return rebalance_four(changed=a, others=[c, o, h], changed_index=0)

def on_crossref_change(c: int, a: int, o: int, h: int):
    return rebalance_four(changed=c, others=[a, o, h], changed_index=1)

def on_openalex_change(o: int, a: int, c: int, h: int):
    return rebalance_four(changed=o, others=[a, c, h], changed_index=2)

def on_hf_change(h: int, a: int, c: int, o: int):
    return rebalance_four(changed=h, others=[a, c, o], changed_index=3)
