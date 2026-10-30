"""Query understanding and routing: the first stage of every request.

  raw query
    -> PII redaction (only the redacted text is ever logged)
    -> injection screen (strip instruction-like spans)
    -> crisis detection          -> route "crisis"        (escalation card first, no essay)
    -> scope rules + intent      -> route "out_of_scope"  (explain scope, escalate) / "decline_drafting"
    -> advice-seeking            -> route "advice"        (what the law says + options, no recommendation)
    -> calculation               -> route "calculation"   (deterministic tool + retrieval for the legal basis)
    -> otherwise                 -> route "answer"        (retrieval-augmented answer with citations)
Intent policy: hybrid - transparent rules decide; the fine-tuned DistilBERT may only
add a protective route (advice / crisis / out-of-scope) at >= 0.95 confidence. Falls back to rules alone
if the trained model is not present (models/ is not committed; notebook 04 trains it).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

from tenancy_rag.guardrails import crisis, injection, pii, scope
from tenancy_rag.understand import intent_rules, slots


@lru_cache(maxsize=1)
def default_intent_model():
    from tenancy_rag.understand.distilbert_clf import OUT
    if (OUT / "selected" / "config.json").exists():
        from tenancy_rag.understand.intent_hybrid import classify
        return classify
    warnings.warn("fine-tuned intent model not found - using rule-based intent only (run notebook 04)")
    return intent_rules.classify


@dataclass
class Routed:
    route: str
    intent: str
    query: str                      # cleaned query used downstream (injection spans removed)
    redacted: str                   # what may be logged
    flags: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)


def route(text: str, intent_model=None) -> Routed:
    redacted, pii_counts = pii.redact(text)
    inj = injection.screen(text)
    q = inj["cleaned"] if inj["injection"] else text
    flags = {"pii": pii_counts, "injection": inj["injection"]}

    cr = crisis.detect(text)
    if cr["crisis"]:
        return Routed("crisis", "crisis", q, redacted, {**flags, "crisis_signals": cr["signals"]})

    if inj["injection"] and len(q.split()) < 4:  # nothing left but the instruction
        return Routed("refuse_injection", "out_of_scope", q, redacted, flags)

    sc = scope.check(q)
    flags.update({"scope_reasons": sc["reasons"], "difc_caveat": sc["difc_caveat"]})
    intent = (intent_model or default_intent_model())(q)
    if "document_drafting" in sc["reasons"]:
        return Routed("decline_drafting", "out_of_scope", q, redacted, flags)
    if not sc["in_scope"] or intent == "out_of_scope":
        return Routed("out_of_scope", "out_of_scope", q, redacted, flags)
    if intent == "crisis":
        return Routed("crisis", intent, q, redacted, flags)
    if intent == "advice_seeking":
        return Routed("advice", intent, q, redacted, flags)
    sl = slots.extract(q)
    if intent == "calculation":
        return Routed("calculation", intent, q, redacted, flags, sl)
    return Routed("answer", intent, q, redacted, flags, sl)
