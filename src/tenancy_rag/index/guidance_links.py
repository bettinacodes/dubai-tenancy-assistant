"""Link each guidance record to the binding provision(s) it most likely paraphrases.

Day 4 error analysis: FAQ answers that restate a law outrank the law itself, and only 4 guidance
records name an article explicitly, so regex cross-references cannot recover the link. Instead
each in-force guidance parent is linked to its most similar in-force legislation/decree parent
by bge-m3 cosine similarity (full parent texts). At query time, a retrieved guidance record
pulls its linked provision into the candidate list ("authority-aware expansion").
Output: data/processed/guidance_links.json  {guidance_parent_id: [{"parent_id", "sim"}, ...]}
"""
from __future__ import annotations

import json

import numpy as np

from tenancy_rag import config
from tenancy_rag.index.embed import Embedder

LINKS = config.PROCESSED / "guidance_links.json"
BINDING = ("legislation", "decree")
SKIP_ARTICLES = {"art-1"}  # citation-title articles ("This Law will be cited as ...") are never useful targets


def build_links(top: int = 2) -> dict:
    parents = [json.loads(l) for l in (config.PROCESSED / "parents.jsonl").open(encoding="utf-8")]
    live = [p for p in parents if p["status"] == "in_force"]
    law = [p for p in live if p["authority_tier"] in BINDING and p["parent_id"].split("__")[-1] not in SKIP_ARTICLES]
    guide = [p for p in live if p["authority_tier"] not in BINDING]
    emb = Embedder()
    L = emb.encode([p["text"] for p in law], "passage")
    G = emb.encode([p["text"] for p in guide], "passage")
    sims = G @ L.T
    links = {}
    for i, g in enumerate(guide):
        order = np.argsort(-sims[i])[:top]
        links[g["parent_id"]] = [{"parent_id": law[j]["parent_id"], "sim": round(float(sims[i, j]), 4)} for j in order]
    LINKS.write_text(json.dumps(links, indent=1), encoding="utf-8")
    return links


def load_links() -> dict:
    return json.loads(LINKS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build_links()
