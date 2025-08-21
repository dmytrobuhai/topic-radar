# core/utils.py
from __future__ import annotations
import html
from typing import Any, Dict, List, Optional

def percent_bar(pct: float, label: str = "") -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return f"""
<div class="tr-progress">
  <div class="tr-progress__bar" style="width:{pct:.0f}%;"></div>
  <div class="tr-progress__label">{html.escape(label)} {pct:.0f}%</div>
</div>
"""

def fmt_date(d: Optional[str]) -> str:
    return (d or "").split("T")[0] if d else "—"

def link(href: Optional[str], text: str) -> str:
    if not href:
        return html.escape(text)
    return f'<a href="{html.escape(href)}" target="_blank">{html.escape(text)}</a>'

def join_authors(authors: Optional[List[Dict[str, Any]] | List[str] | str]) -> str:
    if not authors:
        return "—"
    if isinstance(authors, str):
        names = [authors.strip()] if authors.strip() else []
    elif isinstance(authors, list):
        names = []
        for a in authors:
            if isinstance(a, dict):
                nm = a.get("name")
                if nm: names.append(str(nm))
            elif isinstance(a, str) and a.strip():
                names.append(a.strip())
    else:
        names = []
    if len(names) <= 2:
        return ", ".join(names) if names else "—"
    return f"{len(names)} authors"
