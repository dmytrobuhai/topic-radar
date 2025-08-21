# core/render.py
from __future__ import annotations
import html
from typing import Any, Dict, List

from core.utils import fmt_date, link, join_authors

def _shorten(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    t = t[:limit].rsplit(" ", 1)[0]
    return t + " …"

def render_papers_html(items: List[Dict[str, Any]]) -> str:
    parts = []
    for e in items:
        title = e.get("title") or "untitled"
        links = e.get("links") or {}
        abs_url = links.get("abs") or links.get("openalex") or links.get("event")
        pdf_url = links.get("pdf")
        doi_url = e.get("doi_url")
        src = e.get("source") or ""
        date = fmt_date(e.get("updated") or e.get("published"))
        auth = join_authors(e.get("authors"))
        summary = _shorten(e.get("summary") or "", 800)

        parts.append(f"""
<div class="tr-card-row">
  <div class="tr-card-row__head">
    <div class="tr-card-row__title">{link(abs_url, title)}</div>
    <div class="tr-card-row__meta">{html.escape(src)} · {html.escape(date)}</div>
  </div>
  <details class="tr-card-row__details">
    <summary>Details</summary>
    <div class="tr-card-row__body">
      <div><b>Authors:</b> {html.escape(auth)}</div>
      <div><b>Summary:</b> {html.escape(summary) if summary else '—'}</div>
      <div class="tr-links">
        {(f'<a target="_blank" href="{html.escape(abs_url)}">abs</a>' if abs_url else '')}
        {(f' · <a target="_blank" href="{html.escape(pdf_url)}">pdf</a>' if pdf_url else '')}
        {(f' · <a target="_blank" href="{html.escape(doi_url)}">doi</a>' if doi_url else '')}
      </div>
    </div>
  </details>
</div>
""")
    return "\n".join(parts) if parts else "<div class='tr-empty'>No papers found.</div>"

def render_videos_html(items: List[Dict[str, Any]]) -> str:
    parts = []
    for e in items:
        title = e.get("title") or "untitled"
        url = e.get("url")
        ch = e.get("channel") or ""
        date = fmt_date(e.get("published"))
        desc = _shorten(e.get("description") or "", 600)
        parts.append(f"""
<div class="tr-card-row">
  <div class="tr-card-row__head">
    <div class="tr-card-row__title">{link(url, title)}</div>
    <div class="tr-card-row__meta">{html.escape(ch)} · {html.escape(date)}</div>
  </div>
  <details class="tr-card-row__details">
    <summary>Details</summary>
    <div class="tr-card-row__body">
      <div><b>Description:</b> {html.escape(desc) if desc else '—'}</div>
      <div class="tr-links">{link(url, "watch")}</div>
    </div>
  </details>
</div>
""")
    return "\n".join(parts) if parts else "<div class='tr-empty'>No videos found.</div>"

def render_news_html(items: List[Dict[str, Any]]) -> str:
    parts = []
    for e in items:
        title = e.get("title") or "untitled"
        url = e.get("url")
        src = e.get("source") or ""
        date = fmt_date(e.get("date"))
        summary = _shorten(e.get("summary") or "", 600)
        parts.append(f"""
<div class="tr-card-row">
  <div class="tr-card-row__head">
    <div class="tr-card-row__title">{link(url, title)}</div>
    <div class="tr-card-row__meta">{html.escape(src)} · {html.escape(date)}</div>
  </div>
  <details class="tr-card-row__details">
    <summary>Details</summary>
    <div class="tr-card-row__body">
      <div><b>Summary:</b> {html.escape(summary) if summary else '—'}</div>
      <div class="tr-links">{link(url, "open")}</div>
    </div>
  </details>
</div>
""")
    return "\n".join(parts) if parts else "<div class='tr-empty'>No news found.</div>"
