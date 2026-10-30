"""Hybrid intent policy: transparent rules decide; the fine-tuned classifier may only ADD a
protective route (advice / crisis / out-of-scope) the rules missed, and only when highly confident.

Rationale: costs are asymmetric. Misrouting a factual question to advice or crisis
hides a legitimate answer; an advice question that reaches the generator is still caught by the
generator's advice guard. The classifier is miscalibrated on real questions (wrong at 0.9+), so it
gets a veto-free, high-threshold, add-only role.
"""
from __future__ import annotations

from functools import lru_cache

from tenancy_rag.understand import intent_rules

PROTECTIVE = ("advice_seeking", "crisis", "out_of_scope")
THRESHOLD = 0.95


@lru_cache(maxsize=1)
def _proba_fn():
    import torch
    from tenancy_rag.understand.distilbert_clf import LABELS, OUT, _loaded
    tok, model = _loaded(str(OUT / "selected"))

    def proba(text: str) -> tuple[str, float]:
        with torch.no_grad():
            p = torch.softmax(model(**tok([text], truncation=True, max_length=128, return_tensors="pt")).logits, -1)[0]
        k = int(p.argmax())
        return LABELS[k], float(p[k])
    return proba


def make_classifier(threshold: float = THRESHOLD):
    def classify(text: str) -> str:
        rule = intent_rules.classify(text)
        if rule in PROTECTIVE:
            return rule
        label, p = _proba_fn()(text)
        return label if (label in PROTECTIVE and p >= threshold) else rule
    return classify


classify = make_classifier()
