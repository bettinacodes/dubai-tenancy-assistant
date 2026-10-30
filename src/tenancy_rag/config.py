"""Central configuration: paths, model names and corpus metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVAL = DATA / "eval"
INDEX = DATA / "index"
CACHE = DATA / "cache"
MANIFEST = RAW / "manifest.jsonl"
CHUNKS = PROCESSED / "chunks.jsonl"

# The single "as at" date the whole index represents; shown on every answer.
CORPUS_AS_AT = "2026-10-09"

# Models (all free / open weights; no paid API anywhere in the pipeline)
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
MT_MODEL = "facebook/nllb-200-distilled-600M"
INTENT_BASE_MODEL = "distilbert-base-multilingual-cased"
GEN_MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"        # transformers backend, 4-bit (alternative)
GEN_MODEL_HF_SMALL = "Qwen/Qwen2.5-3B-Instruct"  # comparison model
GEN_MODEL_OLLAMA = "qwen2.5:3b"                  # local development (4 GB GPU)
GEN_MODEL_OLLAMA_7B = "qwen2.5:7b"               # Colab T4 demo + the local 7B evaluation runs

# Retrieval hyperparameters - tuned on the DEV split only
BM25_TOP_K = 30
DENSE_TOP_K = 30
RRF_K = 20            # stage A: k=20 >= k=60 at every BM25 weight
BM25_WEIGHT = 0.5     # stage A + lexical probe: dense-heavy fusion keeps a keyword safety net
RERANK_N = 10         # stage B: depth 10 == 30 on MRR, fastest
RERANK_TOP_N = 6      # parents handed to the generator

USER_AGENT = "MustajirResearchBot/0.1 (non-commercial research; polite, robots.txt-respecting)"

# Devices for auxiliary models (None = cuda if available). Local GTX 1650 (4 GB) also hosts the LLM,
# so the small NLI model runs on CPU by default; override with env TENANCY_NLI_DEVICE.
import os as _os
NLI_DEVICE = _os.environ.get("TENANCY_NLI_DEVICE", "cpu")
# Query-time embedding: chunk vectors are cached, so a 0.28 s CPU query embedding frees ~2.2 GB of
# VRAM for the reranker + LLM on the 4 GB GTX 1650. Env TENANCY_EMBED_DEVICE=cpu|cuda.
EMBED_DEVICE = _os.environ.get("TENANCY_EMBED_DEVICE") or None
