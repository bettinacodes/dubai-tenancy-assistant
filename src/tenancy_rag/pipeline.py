"""Musta'jir end-to-end pipeline: query -> routed, grounded, verified response.

    understand (router)  ->  crisis card | scope / drafting / injection refusal
                         ->  retrieve (dense + filters + rerank + citation lookup)
                         ->  calculators (whenever their inputs are present)
                         ->  generate (citation contract)  ->  validate  ->  retry / extractive fallback
                         ->  assemble: answer + citations + working + caveats + notices + escalation
Numbers come from code, authority from retrieval, only fluency from the model.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from tenancy_rag import config
from tenancy_rag.generate.answer import generate_answer
from tenancy_rag.guardrails import pii
from tenancy_rag.guardrails.crisis import escalation_card
from tenancy_rag.tools import calculators
from tenancy_rag.understand.router import route

NLLB_CODES = {"Arabic": "arb_Arab", "Hindi": "hin_Deva", "Urdu": "urd_Arab"}
# "what changed / what did the old law say" questions. Bare "before", "amend" and "change" are NOT cues: they matched
# 5 ordinary gold questions ("how much notice before...", "can I leave before...").
HISTORICAL_Q = re.compile(r"\b(used to|previously|previous (law|rules?|version|text)|old (law|rules?|index|version)|originally|2007|"
                          r"before (the )?(2008|amendment|change|law 33)|(was|been|were) (amended|changed)|amendments?|what changed|"
                          r"did .{0,60}\bchange|has (it|this|the law|anything) changed|changes? (to|in) the law|history)\b", re.I)
NOTICES = {
    "not_advice": "This is general information about the law, not legal advice for your situation.",
    "arabic_prevails": "The English texts are translations; the Arabic text of Dubai legislation prevails.",
}
SCOPE_MESSAGES = {
    "other_emirate": "I only cover residential tenancy law in Dubai. Each emirate has its own tenancy law, so I can't answer this reliably.",
    "commercial_lease": "I only cover residential tenancies. Commercial leases (shops, offices, warehouses) are outside what I have verified.",
    "property_purchase": "I only cover renting a home in Dubai, not buying property or mortgages.",
    "speculative_future": "I can't predict future changes to the law. My information is as at {as_at}.",
    "unrelated": "That's outside what I can help with. I answer questions about renting a home in Dubai.",
}


@lru_cache(maxsize=1)
def _parents() -> dict[str, dict]:
    return {json.loads(l)["parent_id"]: json.loads(l) for l in (config.PROCESSED / "parents.jsonl").open(encoding="utf-8")}


def detect_language(text: str) -> str:
    if re.search(r"[؀-ۿ]", text):
        return "Urdu" if re.search(r"[ٹڈڑںھہے]", text) else "Arabic"
    if re.search(r"[ऀ-ॿ]", text):
        return "Hindi"
    return "English"


def run_tools(slots: dict, query: str) -> dict | None:
    """Run a calculator whenever its inputs are present (independent of intent)."""
    try:
        if "current_rent" in slots and "market_avg" in slots:
            return calculators.rent_increase_cap(slots["current_rent"]["value"], slots["market_avg"]["value"],
                                                 requested_increase_pct=slots.get("requested_increase_pct")).to_dict()
        if "shortfall_pct" in slots:
            return calculators.rent_increase_cap(shortfall_pct=slots["shortfall_pct"],
                                                 requested_increase_pct=slots.get("requested_increase_pct")).to_dict()
        if "notice_date" in slots and "expiry_date" in slots:
            reason = slots.get("eviction_reason") or ("amend_terms" if re.search(r"\b(rent|increase|hike|terms?|raise)\b", query, re.I) else None)
            if reason:
                out = calculators.notice_period_check(reason, slots["notice_date"]["date"], slots["expiry_date"]["date"],
                                                      slots.get("served_via")).to_dict()
                if slots["notice_date"].get("assumed"):   # relative period ("vacate in 12 months") with no date given
                    out["caveats"].append(f"No notice date was given, so I assumed the notice was received today and "
                                          f"'{slots['notice_period']['span']}' runs from today.")
                return out
        if "annual_rent" in slots and re.search(r"\b(fee|fees|cost|file|filing|rdc|dispute|case)\b", query, re.I):
            return calculators.rdc_fee_estimate(slots["annual_rent"]["value"]).to_dict()
    except ValueError:
        return None
    return None


class Assistant:
    def __init__(self, backend: str = "ollama", model: str | None = None, retrieval_preset: str = "final",
                 check_entailment: bool = True, n_sources: int = config.RERANK_TOP_N):
        from tenancy_rag.retrieve.hybrid import HybridRetriever, preset
        self.backend, self.model, self.check_entailment, self.n_sources = backend, model, check_entailment, n_sources
        self.retriever = HybridRetriever(preset(retrieval_preset))

    # ---------------------------------------------------------------- sources
    def _ranked(self, query: str) -> list[str]:
        ranked: list[str] = []
        for c in self.retriever.search(query, k=30):
            for pid in c.get("parent_ids") or [c["parent_id"]]:
                if pid not in ranked:
                    ranked.append(pid)
        return ranked

    def _sources(self, query: str, tool: dict | None, historical: bool, alt_query: str | None = None) -> list[dict]:
        """Top parents for the query. For non-English questions `alt_query` is the NLLB English translation and
        the two rankings are fused with RRF (k=20): best hit@10 in both languages, MRR within 1 point of the best single ranking."""
        P = _parents()
        ranked = self._ranked(query)
        if alt_query:
            scores: dict[str, float] = {}
            for lst in (ranked, self._ranked(alt_query)):
                for i, pid in enumerate(lst, 1):
                    scores[pid] = scores.get(pid, 0.0) + 1.0 / (20 + i)
            ranked = sorted(scores, key=lambda p: -scores[p])
        pids = ranked[: self.n_sources]
        for src in (tool or {}).get("sources", []):               # the calculator's legal basis is always shown
            if src.get("parent_id") in P and src["parent_id"] not in pids:
                pids.append(src["parent_id"])
        if historical:                                             # superseded versions of what was retrieved
            for c in self.retriever.last_historical[:6]:
                if c["parent_id"] not in pids and len([p for p in pids if P[p]["status"] == "superseded"]) < 2:
                    pids.append(c["parent_id"])
        out = []
        for i, pid in enumerate(pids, 1):
            p = P[pid]
            out.append({"label": f"S{i}", "parent_id": pid, "instrument": p["instrument"], "locator": p["locator"],
                        "heading": p.get("article_heading"), "text": p["text"], "status": p["status"],
                        "authority_tier": p["authority_tier"], "flagged": bool(p.get("quality_flags")),
                        "url": p["source_url"], "superseded_by": p.get("superseded_by", [])})
        return out

    # ---------------------------------------------------------------- main entry
    def answer(self, query: str) -> dict:
        t0 = time.perf_counter()
        lang = detect_language(query)
        query_en = None
        if lang != "English":   # route, extract slots and verify claims in English; answer in the user's language
            from tenancy_rag.eval.translate_parity import translate
            query_en = translate([query], NLLB_CODES[lang], "eng_Latn")[0]
        r = route(query_en or query)
        base = {"route": r.route, "intent": r.intent, "query_logged": r.redacted, "flags": r.flags, "language": lang,
                "query_english": query_en,
                "notices": dict(NOTICES, corpus_as_at=f"Information as at {config.CORPUS_AS_AT}."),
                "escalation": escalation_card()}
        if r.route == "crisis":
            srcs = [s for s in self._sources(pii.redact(r.query)[0], None, False) if s["authority_tier"] in ("legislation", "decree")][:2]
            return {**base, "mode": "crisis_card", "answer": "This sounds urgent. Please contact the police or the Rental Disputes Center now - "
                    "these numbers are verified from official government pages.", "citations": self._cite(srcs),
                    "relevant_law": [{"instrument": s["instrument"], "locator": s["locator"], "text": s["text"][:600]} for s in srcs],
                    "tool": None, "caveats": [], "seconds": round(time.perf_counter() - t0, 2)}
        if r.route in ("out_of_scope", "decline_drafting", "refuse_injection"):
            return {**base, "mode": r.route, "answer": self._scope_message(r), "citations": [], "tool": None,
                    "caveats": [], "seconds": round(time.perf_counter() - t0, 2)}

        tool = run_tools(r.slots, r.query)
        historical = bool(HISTORICAL_Q.search(r.query))
        # Only redacted text goes further: the generator's prompt is written to the LLM disk cache and the query
        # embedding to the embedding cache, so names, phone numbers and IDs must not reach them.
        # Slots and tools above use the original text; redaction never touches amounts or dates.
        q_safe = pii.redact(query if query_en else r.query)[0]
        q_en_safe = pii.redact(query_en)[0] if query_en else None
        sources = self._sources(q_safe, tool, historical, alt_query=q_en_safe)
        question = f"{q_safe}\n(English translation: {q_en_safe})" if query_en else q_safe
        gen = generate_answer(question, sources, tool=tool, language=lang, advice=r.route == "advice",
                              historical=historical, backend=self.backend, model=self.model,
                              check_entailment=self.check_entailment)
        cited = [s for s in sources if s["label"] in gen["cited_labels"]]
        caveats = gen["caveats"] + (tool["caveats"] if tool else [])
        if r.flags.get("difc_caveat"):
            caveats.append("DIFC has its own leasing regime; check with the DIFC Courts or a lawyer.")
        if r.flags.get("injection"):
            caveats.append("Part of your message asked me to change how I work; I ignored that part.")
        caveats += self._change_notes(cited)
        return {**base, "mode": gen["mode"], "answer": gen["answer"], "citations": self._cite(cited), "tool": tool,
                "caveats": list(dict.fromkeys(caveats)), "sources_considered": [s["parent_id"] for s in sources],
                "validation": gen["attempts"], "seconds": round(time.perf_counter() - t0, 2)}

    @staticmethod
    def _cite(sources: list[dict]) -> list[dict]:
        return [{"label": s["label"], "instrument": s["instrument"], "locator": s["locator"], "url": s["url"],
                 "authority": s["authority_tier"], "status": s["status"]} for s in sources]

    @staticmethod
    def _change_notes(cited: list[dict]) -> list[str]:
        P, notes = _parents(), []
        for s in cited:
            for old in P[s["parent_id"]].get("supersedes", []):
                notes.append(f"{s['locator']} of {P[old]['instrument_short']} was replaced by Law No. 33 of 2008; the answer uses the amended text.")
        return notes

    @staticmethod
    def _scope_message(r) -> str:
        if r.route == "refuse_injection":
            return "I can't change how I work or reveal my instructions. Ask me anything about renting a home in Dubai."
        if r.route == "decline_drafting":
            return ("I don't draft legal documents. I can explain what the law requires - for example, the notice periods and "
                    "how a notice must be served - so you or a lawyer can prepare it.")
        reasons = r.flags.get("scope_reasons") or ["unrelated"]
        return " ".join(SCOPE_MESSAGES.get(x, SCOPE_MESSAGES["unrelated"]).format(as_at=config.CORPUS_AS_AT) for x in reasons)
