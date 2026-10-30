"""Deterministic lookup for explicit legal references in the query.

"What does Article 26 of Law 26/2007 say?" is not a search problem but a lookup: the lexical
probe showed dense retrieval ranks the named article first only 25% of the
time. When the query names both an article and a known instrument, the referenced provision is
fetched directly - always its *current* version (an amended article resolves to the L33/2008 text).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from tenancy_rag import config

# instrument patterns -> source id
INSTRUMENTS = {
    "dubai-law-26-2007": re.compile(r"\blaw\s*(?:no\.?\s*)?\(?26\)?\s*(?:of|/)\s*2007", re.I),
    "dubai-law-33-2008": re.compile(r"\blaw\s*(?:no\.?\s*)?\(?33\)?\s*(?:of|/)\s*2008", re.I),
    "dubai-decree-43-2013": re.compile(r"\bdecree\s*(?:no\.?\s*)?\(?43\)?\s*(?:of|/)\s*2013", re.I),
    "dubai-decree-26-2013": re.compile(r"\bdecree\s*(?:no\.?\s*)?\(?26\)?\s*(?:of|/)\s*2013", re.I),
}
ARTICLE = re.compile(r"\bart(?:icle)?\.?\s*\(?\s*(\d{1,2})\s*\)?", re.I)


@lru_cache(maxsize=1)
def _parents() -> dict[str, dict]:
    return {json.loads(l)["parent_id"]: json.loads(l) for l in (config.PROCESSED / "parents.jsonl").open(encoding="utf-8")}


def resolve(source_id: str, article: int) -> str | None:
    """Parent id of the current text of `article` in `source_id` (follows supersession)."""
    parents = _parents()
    pid = f"{source_id}__art-{article}"
    if pid not in parents:
        return None
    while parents[pid]["status"] == "superseded" and parents[pid]["superseded_by"]:
        pid = parents[pid]["superseded_by"][0]
    return pid


def lookup(query: str) -> list[str]:
    """Parent ids explicitly referenced by the query (article + instrument both present)."""
    sources = [sid for sid, pat in INSTRUMENTS.items() if pat.search(query)]
    if not sources:
        return []
    out = []
    for a in ARTICLE.findall(query):
        for sid in sources:
            pid = resolve(sid, int(a))
            if pid and pid not in out:
                out.append(pid)
    return out
