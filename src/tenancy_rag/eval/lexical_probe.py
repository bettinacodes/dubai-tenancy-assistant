"""Lexical probe: exact-reference queries ("Article 26 of Law 26/2007") absent from the gold set.

Used to decide whether a low-weight BM25 earns its place in the first stage.
"""
from __future__ import annotations

import json

from tenancy_rag import config
from tenancy_rag.eval.retrieval_metrics import aggregate, parents_from_chunks, query_metrics
from tenancy_rag.retrieve.hybrid import HybridRetriever, RetrievalConfig

TRIALS = {
    "dense only": RetrievalConfig(bm25=False, rerank=False, citation_lookup=False),
    "rrf w=0.25 k=20": RetrievalConfig(rerank=False, bm25_weight=0.25, rrf_k=20, citation_lookup=False),
    "rrf w=0.5 k=20": RetrievalConfig(rerank=False, bm25_weight=0.5, rrf_k=20, citation_lookup=False),
    "rrf w=1.0 k=60": RetrievalConfig(rerank=False, citation_lookup=False),
    "dense only + lookup": RetrievalConfig(bm25=False, rerank=False),
    "rrf w=0.5 k=20 + lookup": RetrievalConfig(rerank=False, bm25_weight=0.5, rrf_k=20),
}


def run() -> dict:
    probes = [json.loads(l) for l in (config.EVAL / "lexical_probes.jsonl").open(encoding="utf-8")]
    shared = HybridRetriever(RetrievalConfig(rerank=False))
    results = {}
    for label, cfg in TRIALS.items():
        r = HybridRetriever(cfg) if not cfg.bm25 else shared
        r.cfg = cfg
        per, not_first = [], []
        for g in probes:
            m = query_metrics(parents_from_chunks(r.search(g["question"], k=30)), set(g["relevant_ids"]), set(g["primary_ids"]))
            per.append(m)
            if not m["primary@1"]:
                not_first.append(g["id"])
        a = aggregate(per)
        results[label] = {**{k: a[k] for k in ("primary@1", "primary@3", "primary@10", "mrr@10")}, "not_first": not_first}
        print(f"{label:18s} prim@1 {a['primary@1']:.2f} prim@3 {a['primary@3']:.2f} prim@10 {a['primary@10']:.2f} "
              f"MRR {a['mrr@10']:.2f}  not first: {not_first}")
    (config.EVAL / "runs" / "lexical_probe.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
