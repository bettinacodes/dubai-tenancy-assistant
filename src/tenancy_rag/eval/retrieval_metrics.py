"""Article-level retrieval metrics.

Retrieval returns ranked *chunks*; the unit of relevance is the *parent* (article / FAQ pair /
section / page), because the parent is what gets cited. A chunk ranking is collapsed to a parent
ranking by first occurrence (a fixed-512 window counts for every parent it overlaps).

Per query, with relevant set R (gold `relevant_ids`) and primary set P (gold `primary_ids`, the
binding provision that must be cited):
  hit@k        1 if any relevant parent is in the top-k               (answerability)
  recall@k     |R ∩ top-k| / |R|                                       (coverage of all evidence)
  primary@k    1 if any primary parent is in the top-k (queries with P only) (can it cite the law?)
  mrr@10       1 / rank of the first relevant parent (0 if not in top-10)
  ndcg@10      graded gain: primary = 2, other relevant = 1
"""
from __future__ import annotations

import math
from statistics import mean

KS = (1, 3, 5, 10, 20, 30)


def parents_from_chunks(ranked_chunks: list[dict]) -> list[str]:
    """Collapse a ranked chunk list to a ranked, de-duplicated parent list."""
    seen, out = set(), []
    for c in ranked_chunks:
        for pid in c.get("parent_ids") or [c["parent_id"]]:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def query_metrics(ranked_parents: list[str], relevant: set[str], primary: set[str], ks=KS) -> dict:
    m = {}
    for k in ks:
        top = set(ranked_parents[:k])
        m[f"hit@{k}"] = float(bool(top & relevant))
        m[f"recall@{k}"] = len(top & relevant) / len(relevant) if relevant else 0.0
        if primary:
            m[f"primary@{k}"] = float(bool(top & primary))
    rr = 0.0
    for i, pid in enumerate(ranked_parents[:10]):
        if pid in relevant:
            rr = 1.0 / (i + 1)
            break
    m["mrr@10"] = rr
    gain = lambda pid: 2.0 if pid in primary else (1.0 if pid in relevant else 0.0)
    ideal = sorted((gain(p) for p in relevant | primary), reverse=True)[:10]
    m["ndcg@10"] = _dcg([gain(p) for p in ranked_parents[:10]]) / _dcg(ideal) if ideal else 0.0
    return m


def aggregate(per_query: list[dict]) -> dict:
    keys = sorted({k for q in per_query for k, v in q.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for k in keys:
        vals = [q[k] for q in per_query if k in q]
        out[k] = round(mean(vals), 4) if vals else None
    out["n"] = len(per_query)
    return out
