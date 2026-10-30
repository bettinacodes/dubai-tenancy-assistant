"""Dense embeddings via sentence-transformers, with a per-model on-disk cache.

Each (model, text) is embedded once and reused across every index variant and ablation run.
Models compared in notebook 02 (model selection):
  BAAI/bge-m3                              multilingual, 8k context   (the system's choice)
  intfloat/multilingual-e5-large           multilingual, needs "query: "/"passage: " prefixes
  sentence-transformers/all-MiniLM-L6-v2   English-only, small        (weak baseline)
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

from tenancy_rag import config

MODEL_SPECS = {
    "BAAI/bge-m3": {"query": "", "passage": "", "max_len": 512},
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: ", "max_len": 512},
    "sentence-transformers/all-MiniLM-L6-v2": {"query": "", "passage": "", "max_len": 256},
}


def slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def use_fp16() -> bool:
    """fp16 only on tensor-core GPUs. On the GTX 1650 fp16 was 5.5x SLOWER than fp32
    (42.6 s vs 7.8 s for 64 chunks). Colab's T4 has tensor cores."""
    import torch
    return torch.cuda.is_available() and "GTX" not in torch.cuda.get_device_name(0)


def precision() -> str:
    return "fp16" if use_fp16() else "fp32"


@lru_cache(maxsize=2)
def _model(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer
    device = config.EMBED_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SentenceTransformer(model_name, device=device)
    if device == "cuda" and use_fp16():
        m.half()
    m.max_seq_length = MODEL_SPECS[model_name]["max_len"]
    return m


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.name, self.spec = model_name, MODEL_SPECS[model_name]
        # precision is part of the cache key: fp16 and fp32 vectors differ slightly
        self.cache_dir = config.INDEX / "emb_cache" / f"{slug(model_name)}-{precision()}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, np.ndarray] = {}
        self._load_cache()

    def _load_cache(self):
        f = self.cache_dir / "cache.npz"
        if f.exists():
            d = np.load(f)
            self._mem = {k: d[k] for k in d.files}

    def _save_cache(self):
        np.savez(self.cache_dir / "cache.npz", **self._mem)

    @staticmethod
    def _key(text: str) -> str:
        return "h" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str], kind: str = "passage", batch_size: int = 16) -> np.ndarray:
        texts = [self.spec[kind] + t for t in texts]
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._mem))
        if missing:
            vecs = _model(self.name).encode(missing, batch_size=batch_size, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=len(missing) > 64)
            for t, v in zip(missing, vecs):
                self._mem[self._key(t)] = v.astype(np.float32)
            self._save_cache()
        return np.stack([self._mem[self._key(t)] for t in texts])


PREFIX_MODES = ("code", "nl", "none")


def chunk_text(c: dict, prefix_mode: str = "code") -> str:
    """Text that is embedded / indexed: contextual prefix (code-style, natural-language, or none) + chunk."""
    prefix = {"code": c.get("prefix", ""), "nl": c.get("prefix_nl", ""), "none": ""}[prefix_mode]
    return prefix + c["text"]
