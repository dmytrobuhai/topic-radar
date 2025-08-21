# core/alloc.py
from __future__ import annotations
from typing import List, Dict

def normalize_weights(parts: List[float]) -> List[float]:
    parts = [max(0.0, float(p)) for p in parts]
    s = sum(parts)
    if s <= 0:
        return [1.0 / len(parts)] * len(parts)
    return [p / s for p in parts]

def alloc_by_shares(total: int, shares: List[float]) -> List[int]:
    total = max(0, int(total))
    shares = normalize_weights(shares)
    raw = [total * s for s in shares]
    base = [int(x) for x in raw]
    used = sum(base)
    left = total - used
    rema = sorted(range(len(shares)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in range(left):
        base[rema[i % len(shares)]] += 1
    return base

def per_source_split(target_papers: int, arxiv: float, crossref: float, openalex: float, hf: float) -> Dict[str, int]:
    shares = normalize_weights([arxiv, crossref, openalex, hf])
    raw = [target_papers * s for s in shares]
    base = [int(x) for x in raw]
    used = sum(base)
    left = target_papers - used
    order = sorted(range(4), key=lambda i: (raw[i]-base[i]), reverse=True)
    for i in range(left):
        base[order[i % 4]] += 1
    return dict(arxiv=base[0], crossref=base[1], openalex=base[2], hf=base[3])
