"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum over input rankings of w_i / (k + rank_i(d)), rank starting at 1.
Rank-based, so it needs no score calibration between BM25 and cosine similarity. Optional
per-ranking weights w_i let a weaker retriever contribute without dominating (weighted RRF).
"""
from __future__ import annotations


def rrf(rankings: list[list[dict]], k: int = 60, key: str = "chunk_id", weights: list[float] | None = None) -> list[dict]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            kid = item[key]
            scores[kid] = scores.get(kid, 0.0) + w / (k + rank)
            items.setdefault(kid, item)
    ordered = sorted(scores, key=lambda x: -scores[x])
    return [{**items[kid], "score": scores[kid], "retriever": "rrf"} for kid in ordered]
