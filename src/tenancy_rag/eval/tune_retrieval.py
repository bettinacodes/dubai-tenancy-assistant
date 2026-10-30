"""Stage-wise hyperparameter tuning on the DEV split only (15 questions); test stays held out.

Stage A (first stage, no reranker): dense-only vs weighted RRF (bm25_weight, rrf_k), judged by
         primary@30 / recall@30 - what the reranker will get to see.
Stage B (reranker depth): rerank_n, judged by MRR@10 / primary@1, with latency recorded.
Stage C (authority-aware expansion): on/off and the similarity threshold.
Every trial is appended to data/eval/runs/tuning_log.jsonl. Note: on 15 queries one question
moves a metric by 0.067, so ties are broken toward the simpler / faster configuration.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from tenancy_rag import config
from tenancy_rag.eval.retrieval_metrics import aggregate, parents_from_chunks, query_metrics
from tenancy_rag.eval.run_retrieval import RUNS, load_gold
from tenancy_rag.retrieve.hybrid import HybridRetriever, RetrievalConfig

LOG = RUNS / "tuning_log.jsonl"


def score(cfg: RetrievalConfig, items: list[dict], stage: str, retriever=None) -> dict:
    r = retriever or HybridRetriever(cfg)
    r.cfg = cfg
    t0, per = time.perf_counter(), []
    for g in items:
        ranked = parents_from_chunks(r.search(g["question"], k=30))
        per.append(query_metrics(ranked, set(g["relevant_ids"]), set(g["primary_ids"])))
    agg = aggregate(per)
    row = {"stage": stage, "split": items[0]["split"], "ms_per_query": round(1000 * (time.perf_counter() - t0) / len(items)),
           **{k: agg[k] for k in ("hit@1", "hit@10", "primary@1", "primary@10", "primary@30", "recall@30", "mrr@10", "ndcg@10")},
           "cfg": {k: v for k, v in asdict(cfg).items() if k != "dense_model"}}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def show(row: dict, label: str) -> None:
    print(f"{label:38s} prim@1 {row['primary@1']:.2f} prim@10 {row['primary@10']:.2f} prim@30 {row['primary@30']:.2f} "
          f"rec@30 {row['recall@30']:.2f} MRR {row['mrr@10']:.2f} nDCG {row['ndcg@10']:.2f} ({row['ms_per_query']} ms/q)", flush=True)


def stage_a(dev):
    base = RetrievalConfig(rerank=False)
    r = HybridRetriever(base)               # one retriever object; only fusion params change
    trials = {"dense only": RetrievalConfig(bm25=False, rerank=False)}
    for w in (0.25, 0.5, 1.0):
        for k in (20, 60):
            trials[f"rrf bm25_w={w} k={k}"] = RetrievalConfig(rerank=False, bm25_weight=w, rrf_k=k)
    out = {}
    for label, cfg in trials.items():
        rr = HybridRetriever(cfg) if not cfg.bm25 else r
        out[label] = score(cfg, dev, "A", rr)
        show(out[label], label)
    return out


def stage_b(dev, first: RetrievalConfig):
    from dataclasses import replace
    r = HybridRetriever(replace(first, rerank=True))
    out = {}
    for n in (10, 20, 30):
        cfg = replace(first, rerank=True, rerank_n=n)
        out[n] = score(cfg, dev, "B", r)
        show(out[n], f"rerank_n={n}")
    return out


def stage_c(dev, best: RetrievalConfig):
    from dataclasses import replace
    r = HybridRetriever(replace(best, expand=True))
    out = {"off": score(replace(best, expand=False), dev, "C", r)}
    show(out["off"], "expand off")
    for t in (0.65, 0.70, 0.75):
        cfg = replace(best, expand=True, expand_min_sim=t)
        out[t] = score(cfg, dev, "C", r)
        show(out[t], f"expand min_sim={t}")
    return out


if __name__ == "__main__":
    import sys
    dev = [g for g in load_gold() if g["split"] == "dev"]
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    if stage == "A":
        stage_a(dev)
