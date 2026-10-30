"""LLM intent classifier (local qwen2.5 via Ollama, JSON output): the comparison for the
fine-tuned DistilBERT (Day 8) - accuracy vs. latency vs. cost/privacy trade-off."""
from __future__ import annotations

import json

from tenancy_rag.llm import chat

LABELS = ["factual", "calculation", "procedural", "out_of_scope", "advice_seeking", "crisis"]
SYSTEM = """Classify a Dubai tenant's message into exactly one intent:
- factual: what the tenancy law says (rights, obligations, rules)
- calculation: needs numbers worked out (rent increase cap from amounts/percentages, notice dates, RDC fee from rent)
- procedural: how to do something (register Ejari, file a case, documents, fees to pay, deadlines, apps, offices)
- out_of_scope: other emirates, commercial/office/shop leases, buying property, unrelated topics, drafting documents, predicting future law
- advice_seeking: asks what they personally should do, or whether they will win
- crisis: urgent danger - locked out, belongings thrown out, threats/violence, passport or ID withheld, vulnerable people without water/power
Return JSON: {"intent": "<one label>"}"""


def classify(text: str, model: str | None = None) -> str:
    res = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
               json_mode=True, max_tokens=20, model=model)
    try:
        label = json.loads(res.text).get("intent", "factual")
    except json.JSONDecodeError:
        label = "factual"
    return label if label in LABELS else "factual"
