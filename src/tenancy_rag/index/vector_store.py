"""Persistent vector store (ChromaDB) holding chunk embeddings plus filterable metadata.

One collection per index variant: {chunkset}__{model}__{prefix|noprefix}. Embeddings are
computed by `Embedder` (cached) and handed to Chroma, so Chroma is used purely as the persistent
ANN store with metadata filtering (status, jurisdiction, authority tier) - FAISS would do the
same job without the metadata layer.
"""
from __future__ import annotations

import chromadb

from tenancy_rag import config
from tenancy_rag.index.embed import Embedder, chunk_text, precision, slug

_META_KEYS = ("parent_id", "instrument_short", "authority_tier", "jurisdiction", "status", "locator")


def _client():
    return chromadb.PersistentClient(path=str(config.INDEX / "chroma"))


def collection_name(chunkset: str, model_name: str, prefix_mode: str) -> str:
    return f"{chunkset}__{slug(model_name)}-{precision()}__{prefix_mode}".replace(".", "-")


class DenseIndex:
    def __init__(self, chunks: list[dict], chunkset: str = "struct", model_name: str = config.EMBED_MODEL,
                 prefix_mode: str = "code", rebuild: bool = False):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.embedder = Embedder(model_name)
        client = _client()
        name = collection_name(chunkset, model_name, prefix_mode)
        if rebuild:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine", "hnsw:search_ef": 200})
        if self.col.count() != len(chunks):
            if self.col.count():
                client.delete_collection(name)
                self.col = client.create_collection(name, metadata={"hnsw:space": "cosine", "hnsw:search_ef": 200})
            embs = self.embedder.encode([chunk_text(c, prefix_mode) for c in chunks], "passage")
            metas = [{k: (c.get(k) if c.get(k) is not None else "") for k in _META_KEYS} for c in chunks]
            self.col.add(ids=[c["chunk_id"] for c in chunks], embeddings=embs.tolist(), metadatas=metas)

    def search(self, query: str, k: int = 30, where: dict | None = None) -> list[dict]:
        q = self.embedder.encode([query], "query")[0]
        res = self.col.query(query_embeddings=[q.tolist()], n_results=min(k, len(self.chunks)), where=where)
        return [{**self.by_id[cid], "score": 1.0 - d, "retriever": "dense"}
                for cid, d in zip(res["ids"][0], res["distances"][0])]
