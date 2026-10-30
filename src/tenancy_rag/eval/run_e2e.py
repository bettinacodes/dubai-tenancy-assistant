"""Staged end-to-end evaluation (runs on the 4 GB local GPU by never loading two big models at once).

Stage 1  route + calculators + retrieval (reranker on GPU) for every item -> e2e_stage1__<set>.jsonl
Stage 2  generation + validation for one generator (Ollama) from the saved sources -> e2e_stage2__<set>__<model>.jsonl
         The validation trace yields two systems from the same run (identical first attempt, cached):
           * with validator  : the final response (retry / fallbacks)
           * without validator (ablation 7): attempt 1 shown as-is
Stage 3  scoring -> eval/score_e2e.py
Sets: "en" (50 gold + 20 adversarial), "ar"/"hi" (50 gold, machine-translated) with variant
"direct" (retrieve in the original language) or "mt" (NLLB to English first).
"""
from __future__ import annotations

import json
import sys
import time

from tenancy_rag import config
from tenancy_rag.pipeline import HISTORICAL_Q, _parents, detect_language, run_tools
from tenancy_rag.understand.router import route

RUNS = config.EVAL / "runs"


def _load(p):
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def items_for(set_name: str) -> list[dict]:
    if set_name == "en":
        return [{**g, "set": "gold"} for g in _load(config.EVAL / "gold.jsonl")] + \
               [{**a, "set": "adversarial"} for a in _load(config.EVAL / "adversarial.jsonl")]
    lang, variant = set_name.split("_")          # e.g. ar_direct, hi_mt
    rows = _load(config.EVAL / "multilingual" / f"gold_{lang}.jsonl")
    return [{**r, "set": f"gold_{lang}", "retrieval_query": r["question"] if variant == "direct" else r["back_translation"],
             "routing_query": r["question"] if variant == "direct" else r["back_translation"]} for r in rows]


def stage1(set_name: str) -> list[dict]:
    from tenancy_rag.pipeline import Assistant
    a = Assistant(backend="ollama")
    out = []
    for i, it in enumerate(items_for(set_name), 1):
        t0 = time.perf_counter()
        rq = it.get("routing_query", it["question"])
        r = route(rq)
        rec = {"id": it["id"], "set": it["set"], "question": it["question"], "lang": detect_language(it["question"]),
               "route": r.route, "intent": r.intent, "flags": r.flags, "slots": {k: v for k, v in r.slots.items()}}
        if r.route in ("crisis", "out_of_scope", "decline_drafting", "refuse_injection"):
            rec.update(tool=None, sources=[], ranked_parents=[], historical=False)
        else:
            q = it.get("retrieval_query", r.query)
            tool = run_tools(r.slots, r.query)
            historical = bool(HISTORICAL_Q.search(q))
            ranked = []
            for c in a.retriever.search(q, k=30):
                for pid in c.get("parent_ids") or [c["parent_id"]]:
                    if pid not in ranked:
                        ranked.append(pid)
            srcs = a._sources(q, tool, historical)
            rec.update(tool=tool, historical=historical, ranked_parents=ranked[:30],
                       sources=[s["parent_id"] for s in srcs], retrieval_query=q)
        rec["stage1_s"] = round(time.perf_counter() - t0, 2)
        out.append(rec)
        print(f"[{i}] {it['id']} {rec['route']} {rec['stage1_s']}s", flush=True)
    path = RUNS / f"e2e_stage1__{set_name}.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in out) + "\n", encoding="utf-8")
    return out


def _source_dicts(pids: list[str]) -> list[dict]:
    P = _parents()
    return [{"label": f"S{i}", "parent_id": pid, "instrument": P[pid]["instrument"], "locator": P[pid]["locator"],
             "heading": P[pid].get("article_heading"), "text": P[pid]["text"], "status": P[pid]["status"],
             "authority_tier": P[pid]["authority_tier"], "flagged": bool(P[pid].get("quality_flags")),
             "url": P[pid]["source_url"], "superseded_by": P[pid].get("superseded_by", [])} for i, pid in enumerate(pids, 1)]


def stage2(set_name: str, model: str, backend: str = "ollama") -> list[dict]:
    from tenancy_rag.generate.answer import generate_answer
    s1 = _load(RUNS / f"e2e_stage1__{set_name}.jsonl")
    out = []
    for i, rec in enumerate(s1, 1):
        if not rec["sources"]:
            out.append({**rec, "model": model, "mode": rec["route"], "answer": None, "attempts": [], "cited": []})
            continue
        t0 = time.perf_counter()
        srcs = _source_dicts(rec["sources"])
        q = rec["question"] if rec["lang"] == "English" else f"{rec['question']}\n(English translation: {rec.get('retrieval_query', '')})"
        gen = generate_answer(q, srcs, tool=rec["tool"], language=rec["lang"], advice=rec["route"] == "advice",
                              historical=rec["historical"], backend=backend, model=model)
        label2pid = {s["label"]: s["parent_id"] for s in srcs}
        first = gen["attempts"][0] if gen["attempts"] else None
        out.append({**rec, "model": model, "mode": gen["mode"], "answer": gen["answer"],
                    "cited": [label2pid[l] for l in gen["cited_labels"] if l in label2pid],
                    "attempts": gen["attempts"], "dropped_for_context": gen.get("dropped_for_context", []),
                    "no_validator": {"claims_cited": sorted({label2pid.get(l, l) for c in (first or {}).get("claims", []) for l in c["sources"]}),
                                     "claims": (first or {}).get("claims", []), "problems": (first or {}).get("problems", [])},
                    "stage2_s": round(time.perf_counter() - t0, 2)})
        print(f"[{i}/{len(s1)}] {rec['id']} {gen['mode']} {out[-1]['stage2_s']}s", flush=True)
    tag = model.replace(":", "-").replace("/", "_")
    path = RUNS / f"e2e_stage2__{set_name}__{tag}.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in out) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    stage, set_name = sys.argv[1], sys.argv[2]
    if stage == "1":
        stage1(set_name)
    else:
        stage2(set_name, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "ollama")   # backend: ollama | transformers
