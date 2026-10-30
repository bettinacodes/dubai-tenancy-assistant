"""Retrieval evaluation harness: run a retriever over the gold questions and score it.

Each run saves data/eval/runs/retrieval__<config>.json with per-query rankings (top-10 parents)
and metrics, plus aggregates for all / dev / test, so every reported number regenerates.
Usage:  python -m tenancy_rag.eval.run_retrieval [config ...]
"""
from __future__ import annotations

import json
import sys
import time

from tenancy_rag import config
from tenancy_rag.eval.retrieval_metrics import aggregate, parents_from_chunks, query_metrics

RUNS = config.EVAL / "runs"
TOP_K = 30  # chunks retrieved per query (before any fusion / reranking)


def load_gold() -> list[dict]:
    return [json.loads(l) for l in (config.EVAL / "gold.jsonl").open(encoding="utf-8")]


def build_retriever(name: str):
    """Config names: <dense|dense-e5|dense-minilm|bm25|bm25-nostem>__<struct|struct-nl|struct-noprefix|fixed512>."""
    from tenancy_rag.index.bm25_index import BM25Index, load_chunks
    if name.startswith("hyb:"):  # hyb:<preset>[:key=val,key=val]
        from tenancy_rag.retrieve.hybrid import HybridRetriever, preset
        parts = name.split(":")
        overrides = {}
        if len(parts) > 2:
            for kv in parts[2].split(","):
                k, v = kv.split("=")
                overrides[k] = v.lower() == "true" if v.lower() in ("true", "false") else (float(v) if "." in v else int(v) if v.isdigit() else v)
        return HybridRetriever(preset(parts[1], **overrides))
    kind, chunkset = name.split("__")
    prefix_mode = {"struct": "code", "struct-nl": "nl", "struct-noprefix": "none", "fixed512": "none"}[chunkset]
    chunks = load_chunks("fixed512" if chunkset == "fixed512" else "struct")
    if kind.startswith("bm25"):
        return BM25Index(chunks, prefix_mode=prefix_mode, stem=kind != "bm25-nostem")
    model = {"dense": config.EMBED_MODEL, "dense-e5": "intfloat/multilingual-e5-large",
             "dense-minilm": "sentence-transformers/all-MiniLM-L6-v2"}[kind]
    from tenancy_rag.index.vector_store import DenseIndex
    return DenseIndex(chunks, chunkset="fixed512" if chunkset == "fixed512" else "struct", model_name=model, prefix_mode=prefix_mode)


def evaluate(name: str, retriever=None, gold=None, save: bool = True) -> dict:
    gold = gold or load_gold()
    retriever = retriever or build_retriever(name)
    per_query, t0 = [], time.perf_counter()
    for g in gold:
        ranked = parents_from_chunks(retriever.search(g["question"], k=TOP_K))
        m = query_metrics(ranked, set(g["relevant_ids"]), set(g["primary_ids"]))
        per_query.append({"id": g["id"], "split": g["split"], "category": g["category"], "style": g["style"],
                          **m, "top10": ranked[:10]})
    secs = time.perf_counter() - t0
    result = {"config": name, "ms_per_query": round(1000 * secs / len(gold), 1),
              "all": aggregate(per_query),
              "dev": aggregate([q for q in per_query if q["split"] == "dev"]),
              "test": aggregate([q for q in per_query if q["split"] == "test"]),
              "per_query": per_query}
    if save:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"retrieval__{name.replace(':', '~').replace('=', '-')}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def fmt(result: dict, split: str = "all") -> str:
    a = result[split]
    return (f"{result['config']:34s} hit@1 {a['hit@1']:.2f}  hit@5 {a['hit@5']:.2f}  hit@10 {a['hit@10']:.2f}  "
            f"prim@1 {a['primary@1']:.2f}  prim@10 {a['primary@10']:.2f}  rec@10 {a['recall@10']:.2f}  "
            f"MRR {a['mrr@10']:.2f}  nDCG {a['ndcg@10']:.2f}  ({result['ms_per_query']} ms/q)")


SUMMARY_COLS = ["hit@1", "hit@5", "hit@10", "hit@30", "primary@1", "primary@10", "primary@30",
                "recall@10", "recall@30", "mrr@10", "ndcg@10"]


def summary_table(split: str = "test"):
    """Collect every saved retrieval run into one table (notebook 02)."""
    import pandas as pd
    rows = []
    for f in sorted(RUNS.glob("retrieval__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"config": r["config"], **{c: r[split].get(c) for c in SUMMARY_COLS}, "n": r[split]["n"],
                     "ms_per_query": r["ms_per_query"]})
    df = pd.DataFrame(rows).set_index("config")
    df.to_csv(RUNS / f"retrieval_summary_{split}.csv")
    return df


DAY4_CONFIGS = ["bm25__struct", "bm25__struct-noprefix", "bm25-nostem__struct", "bm25__fixed512",
                "dense__struct", "dense__struct-nl", "dense__struct-noprefix", "dense__fixed512"]

def main(names: list[str] | None = None) -> None:
    for n in names or DAY4_CONFIGS:
        print(fmt(evaluate(n)), flush=True)
    for split in ("all", "dev", "test"):
        summary_table(split)


if __name__ == "__main__":
    main(sys.argv[1:])
