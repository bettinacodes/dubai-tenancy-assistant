"""Configuration 0: the LLM with NO retrieval. This measured baseline motivates the architecture.

The prompt is deliberately fair: it asks for citations and permits "I don't know", so failures
cannot be blamed on the prompt. Outputs are saved to data/eval/runs/ for scoring and manual review.
"""
from __future__ import annotations

import json
import sys

from tenancy_rag import config
from tenancy_rag.llm import chat

SYSTEM = ("You are an assistant that answers questions about residential tenancy law in Dubai, UAE. "
          "Answer accurately and concisely. Cite the specific law or decree and the article number(s) "
          "you rely on. If you do not know, say so.")

RUNS = config.EVAL / "runs"


def run(backend: str = "ollama", model: str | None = None) -> list[dict]:
    items = [json.loads(l) for l in (config.EVAL / "gold.jsonl").open(encoding="utf-8")]
    items += [json.loads(l) for l in (config.EVAL / "adversarial.jsonl").open(encoding="utf-8")]
    out = []
    for i, it in enumerate(items, 1):
        res = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["question"]}],
                   backend=backend, model=model)
        out.append({"id": it["id"], "question": it["question"], "answer": res.text, "model": res.model,
                    "latency_s": res.latency_s, "cached": res.cached})
        print(f"[{i}/{len(items)}] {it['id']} {res.latency_s:.1f}s{' (cached)' if res.cached else ''}", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    tag = (model or config.GEN_MODEL_OLLAMA).replace(":", "-").replace("/", "_")
    path = RUNS / f"config0_no_retrieval__{tag}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", path)
    return out


if __name__ == "__main__":
    run(model=sys.argv[1] if len(sys.argv) > 1 else None)
