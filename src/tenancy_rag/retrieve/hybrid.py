"""The retrieval pipeline. Every stage is switchable so each ablation row is just a config.

  dense (bge-m3, no prefix) ─┐
                             ├─ RRF ─ metadata filters ─ authority-aware expansion ─ cross-encoder rerank ─ demote flagged
  BM25 (legal tokens, code prefix) ┘

Stage choices were made on the retrieval ablations (prefix per retriever, superseded leakage,
FAQ crowding). `search` returns ranked chunks for the current law; `last_historical` keeps the
superseded candidates seen, for "what changed?" questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from tenancy_rag import config
from tenancy_rag.retrieve.filters import demote_flagged, split_superseded
from tenancy_rag.retrieve.fusion import rrf


@dataclass(frozen=True)
class RetrievalConfig:
    dense: bool = True
    bm25: bool = True
    chunkset: str = "struct"            # struct | fixed512
    dense_prefix: str = "none"          # code | nl | none   (Day 4: none is best for dense)
    bm25_prefix: str = "code"           # code | nl | none   (Day 4: code is best for BM25)
    dense_model: str = config.EMBED_MODEL
    first_k: int = config.DENSE_TOP_K   # candidates per first-stage retriever
    rrf_k: int = config.RRF_K
    bm25_weight: float = config.BM25_WEIGHT  # weighted RRF: dense weight is 1.0
    filters: bool = True                # drop superseded from current-law ranking, demote flagged guidance
    expand: bool = False                # authority-aware expansion via guidance_links
    expand_min_sim: float = 0.70
    expand_from_top: int = 10
    rerank: bool = True
    rerank_n: int = config.RERANK_N     # candidates given to the cross-encoder
    citation_lookup: bool = True        # explicit "Article N of Law X" references are fetched directly


PRESETS = {  # ablation grid; all rows share the dev-tuned fusion/rerank parameters
    "1_dense":               RetrievalConfig(bm25=False, filters=False, rerank=False),
    "2_bm25":                RetrievalConfig(dense=False, filters=False, rerank=False),
    "3_hybrid_rrf":          RetrievalConfig(filters=False, rerank=False),
    "4_hybrid_rerank":       RetrievalConfig(filters=False),
    "4f_hybrid_filters":     RetrievalConfig(rerank=False),
    "4fr_hybrid_filters_rerank": RetrievalConfig(),
    "4frx_full":             RetrievalConfig(expand=True),
    "final":                 RetrievalConfig(bm25=False),  # dense + filters + rerank(10) + citation lookup
    "1fr_dense_filters_rerank": RetrievalConfig(bm25=False),
    "5_fixed512":            RetrievalConfig(chunkset="fixed512", filters=False, bm25_prefix="none"),
    "6_no_prefix":           RetrievalConfig(filters=False, dense_prefix="none", bm25_prefix="none"),
}


class HybridRetriever:
    def __init__(self, cfg: RetrievalConfig = RetrievalConfig()):
        from tenancy_rag.index.bm25_index import BM25Index, load_chunks
        self.cfg = cfg
        self.chunks = load_chunks(cfg.chunkset)
        self.dense = None
        if cfg.dense:
            from tenancy_rag.index.vector_store import DenseIndex
            self.dense = DenseIndex(self.chunks, chunkset=cfg.chunkset, model_name=cfg.dense_model,
                                    prefix_mode=cfg.dense_prefix)
        self.bm25 = BM25Index(self.chunks, prefix_mode=cfg.bm25_prefix) if cfg.bm25 else None
        self.by_parent: dict[str, list[dict]] = {}
        for c in self.chunks:
            for pid in c.get("parent_ids") or [c["parent_id"]]:
                self.by_parent.setdefault(pid, []).append(c)
        self.links = {}
        if cfg.expand:
            from tenancy_rag.index.guidance_links import load_links
            self.links = load_links()
        self.last_historical: list[dict] = []

    def _first_stage(self, query: str) -> list[dict]:
        lists = []
        if self.dense:
            lists.append(self.dense.search(query, k=self.cfg.first_k))
        if self.bm25:
            lists.append(self.bm25.search(query, k=self.cfg.first_k))
        return lists[0] if len(lists) == 1 else rrf(lists, k=self.cfg.rrf_k, weights=[1.0, self.cfg.bm25_weight])

    def _expand(self, cands: list[dict]) -> list[dict]:
        """After each guidance hit in the top-N, insert the binding provision it paraphrases."""
        present = {c["chunk_id"] for c in cands}
        out = []
        for i, c in enumerate(cands):
            out.append(c)
            if i >= self.cfg.expand_from_top or c["authority_tier"] in ("legislation", "decree"):
                continue
            for link in self.links.get(c["parent_id"], [])[:1]:
                if link["sim"] < self.cfg.expand_min_sim:
                    continue
                for lc in self.by_parent.get(link["parent_id"], []):
                    if lc["chunk_id"] not in present:
                        present.add(lc["chunk_id"])
                        out.append({**lc, "retriever": "expansion", "expanded_from": c["parent_id"]})
        return out

    def search(self, query: str, k: int = 30) -> list[dict]:
        cands = self._first_stage(query)
        self.last_historical = []
        if self.cfg.filters:
            cands, self.last_historical = split_superseded(cands)
        if self.cfg.expand:
            cands = self._expand(cands)
        if self.cfg.rerank:
            from tenancy_rag.retrieve.rerank import rerank
            head, tail = cands[: self.cfg.rerank_n], cands[self.cfg.rerank_n:]
            cands = rerank(query, head) + tail
        if self.cfg.filters:
            cands = demote_flagged(cands)
        if self.cfg.citation_lookup:
            cands = self._pin_cited(query, cands)
        return cands[:k]

    def _pin_cited(self, query: str, cands: list[dict]) -> list[dict]:
        """Put the provision(s) the user explicitly named at the top (current version)."""
        from tenancy_rag.retrieve.citation_lookup import lookup
        pids = lookup(query)
        if not pids:
            return cands
        pinned = [{**c, "retriever": "citation_lookup"} for pid in pids for c in self.by_parent.get(pid, [])]
        ids = {c["chunk_id"] for c in pinned}
        return pinned + [c for c in cands if c["chunk_id"] not in ids]


def preset(name: str, **overrides) -> RetrievalConfig:
    return replace(PRESETS[name], **overrides)
