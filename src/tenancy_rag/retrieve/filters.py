"""Metadata handling applied to candidate lists (works for any retriever).

* Superseded provisions are removed from the *current-law* ranking. They are kept in the index
  and returned separately (`historical`) so the system can answer "what changed?" without ever
  presenting them as current law.
* Guidance that misstates the binding text (data/known_conflicts.json -> `quality_flags`) is
  demoted below every unflagged candidate, never deleted: its procedural parts may still help.
* Jurisdiction filter: only Dubai records are eligible for Dubai questions (all are Dubai today;
  the hook exists for the federal baseline corpus).
"""
from __future__ import annotations

import json
from functools import lru_cache

from tenancy_rag import config


@lru_cache(maxsize=1)
def _parent_meta() -> dict[str, dict]:
    out = {}
    for line in (config.PROCESSED / "parents.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        out[p["parent_id"]] = {"status": p["status"], "flagged": bool(p.get("quality_flags")),
                               "tier": p["authority_tier"], "jurisdiction": p["jurisdiction"]}
    return out


def _pids(c: dict) -> list[str]:
    return c.get("parent_ids") or [c["parent_id"]]


def split_superseded(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    """(current, historical). A fixed window is 'current' if any parent it covers is in force."""
    meta = _parent_meta()
    current, historical = [], []
    for c in cands:
        live = any(meta[p]["status"] == "in_force" for p in _pids(c))
        (current if live else historical).append(c)
    return current, historical


def demote_flagged(cands: list[dict]) -> list[dict]:
    meta = _parent_meta()
    ok = [c for c in cands if not any(meta[p]["flagged"] for p in _pids(c))]
    bad = [c for c in cands if any(meta[p]["flagged"] for p in _pids(c))]
    return ok + bad


def filter_jurisdiction(cands: list[dict], jurisdiction: str | None = "dubai") -> list[dict]:
    if jurisdiction is None:
        return cands
    meta = _parent_meta()
    return [c for c in cands if any(meta[p]["jurisdiction"] == jurisdiction for p in _pids(c))]
