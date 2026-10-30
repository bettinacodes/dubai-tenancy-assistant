"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3 (multilingual, fp16 on the GPU).

A bi-encoder embeds query and passage separately; the cross-encoder reads them *together*, so
it can check whether this specific provision answers this specific question. Too slow for the
whole corpus, so it re-orders only the first-stage top-N candidates.
The passage shown to the reranker is the natural-language prefix + chunk text, so it knows which
instrument / article it is reading.
"""
from __future__ import annotations

from functools import lru_cache

from tenancy_rag import config


def use_fp16() -> bool:
    """fp16 only on GPUs with tensor cores. Measured on the GTX 1650 (Turing, no tensor cores):
    fp16 27.0 s vs fp32 4.7 s for 30 pairs, identical scores. Colab's T4 has tensor cores."""
    import torch
    return torch.cuda.is_available() and "GTX" not in torch.cuda.get_device_name(0)


@lru_cache(maxsize=1)
def _model():
    import torch
    from sentence_transformers import CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = CrossEncoder(config.RERANK_MODEL, device=device, max_length=512,
                     model_kwargs={"torch_dtype": torch.float16} if use_fp16() else {})
    return m


def rerank(query: str, cands: list[dict], top_n: int | None = None, batch_size: int = 8) -> list[dict]:
    if not cands:
        return []
    pairs = [(query, c.get("prefix_nl", "") + c["text"]) for c in cands]
    scores = _model().predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(zip(cands, scores), key=lambda x: -float(x[1]))
    out = [{**c, "rerank_score": float(s)} for c, s in ranked]
    return out[:top_n] if top_n else out
