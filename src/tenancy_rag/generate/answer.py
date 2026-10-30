"""Grounded answer generation with verification, one regeneration, and deterministic fallbacks.

generate -> validate -> (if it fails) regenerate once with the validator's problems as feedback
-> (if it still fails) a deterministic fallback - an unverified paraphrase is never shown:
     * advice questions      -> options template (what exists, who can advise) - no recommendation
     * "what changed" (hist.) -> the 2007 and 2008 texts side by side (diff), not a paraphrase
     * otherwise             -> extractive: quote the most relevant clause of the binding provision
"""
from __future__ import annotations

import difflib
import re
import time

from tenancy_rag.generate.citation_validator import premise_windows, validate
from tenancy_rag.generate.prompts import build_messages
from tenancy_rag.llm import NUM_CTX, chat

BINDING = ("legislation", "decree")
MAX_OUTPUT_TOKENS = 700
DIRECTIVE = re.compile(r"\b(you should|you must (file|sue|go|take)|i (would )?recommend|i advise|my advice|you will (win|lose)|"
                       r"you'?ll (win|lose)|you are likely to (win|lose)|go ahead and (file|sue)|best (option|course) (is|for you))\b", re.I)


def fit_context(sources: list[dict], fixed_chars: int, chars_per_token: float = 3.5) -> tuple[list[dict], list[str]]:
    """Drop lowest-ranked sources until prompt + output budget fits the context window (instead of
    letting the LLM server silently truncate the prompt). Returns (kept, dropped parent ids)."""
    budget = NUM_CTX - MAX_OUTPUT_TOKENS - 200
    kept = list(sources)
    dropped = []
    while kept and (fixed_chars + sum(len(s["text"]) + 200 for s in kept)) / chars_per_token > budget:
        dropped.append(kept.pop()["parent_id"])
    return kept, dropped


def _best_passage(question: str, sources: list[dict], cited: list[str] | None = None) -> tuple[dict, str]:
    """Most relevant clause window (lexical overlap) of an in-force binding source. Sources the model cited in
    its failed attempts come first: the citation is usually right even when a claim fails verification, and
    quoting a different provision lost the binding article in 5/50 answers per model."""
    binding = [s for s in sources if s["authority_tier"] in BINDING and s["status"] == "in_force"]
    preferred = [s for l in (cited or []) for s in binding if s["label"] == l]
    pool = preferred[:1] or binding or sources
    q = set(re.findall(r"[a-z]{4,}", question.lower()))
    # FAQ records: quote the answer, never a bare "Q: ..." line (seen in the Arabic smoke test)
    cands = [(s, s["text"].split("\nA:", 1)[1].strip() if "\nA:" in s["text"] else w)
             for s in pool[:3] for w in premise_windows(s["text"]) if not w.startswith("Q:")]
    cands = cands or [(pool[0], pool[0]["text"])]
    return max(cands, key=lambda sw: len(q & set(re.findall(r"[a-z]{4,}", sw[1].lower()))))


def _advice_fallback(sources: list[dict]) -> dict:
    law = [s for s in sources if s["authority_tier"] in BINDING and s["status"] == "in_force"][:2]
    text = ("I can't tell you what to do or predict how a case would go - that depends on facts and evidence a lawyer or the "
            "Rental Disputes Center would need to assess. What exists officially:\n"
            "- Amicable settlement: the RDC's Mediation and Conciliation Directorate tries to settle disputes first, and half "
            "the filing fee is refunded if a settlement is reached.\n"
            "- Filing a case at the Rental Disputes Center (online through the RDC system or Dubai REST).\n"
            "- A qualified lawyer can advise on your specific situation.")
    if law:
        text += "\nThe provisions most relevant to your question: " + "; ".join(f"{s['instrument']}, {s['locator']}" for s in law) + "."
    return {"mode": "advice_template", "answer": text, "cited_labels": [s["label"] for s in law], "claims": [],
            "caveats": []}


def _historical_fallback(sources: list[dict]) -> dict | None:
    by_id = {s["parent_id"]: s for s in sources}
    for old in (s for s in sources if s["status"] == "superseded"):
        new = next((by_id[n] for n in old.get("superseded_by", []) if n in by_id), None)
        if not new:
            continue
        a, b = old["text"].split("\n"), new["text"].split("\n")
        removed = [l for l in difflib.ndiff(a, b) if l.startswith("- ")][:6]
        added = [l for l in difflib.ndiff(a, b) if l.startswith("+ ")][:6]
        text = (f"{new['locator']} was replaced by Law No. 33 of 2008. Lines of the original 2007 text that changed:\n"
                + "\n".join(f"  2007: {l[2:]}" for l in removed)
                + "\nThe current (amended) text reads:\n" + "\n".join(f"  now: {l[2:]}" for l in added))
        return {"mode": "historical_diff", "answer": text, "cited_labels": [new["label"], old["label"]], "claims": [],
                "caveats": ["The 2007 text is shown only to explain the change; it is no longer in force."]}
    return None


def generate_answer(question: str, sources: list[dict], *, tool: dict | None = None, language: str = "English",
                    advice: bool = False, historical: bool = False, backend: str = "ollama", model: str | None = None,
                    check_entailment: bool = True) -> dict:
    fixed = len(question) + 3000 + (len(" ".join(tool["working"])) + 400 if tool else 0)
    sources, dropped = fit_context(sources, fixed)
    attempts, problems = [], None
    for attempt in range(2):
        t0 = time.perf_counter()
        msgs = build_messages(question, sources, language, tool, advice, historical, retry_problems=problems)
        res = chat(msgs, backend=backend, model=model, json_mode=True, max_tokens=MAX_OUTPUT_TOKENS)
        v = validate(res.text, sources, question, tool, historical, check_entailment)
        if advice and v.parsed and DIRECTIVE.search(str(v.parsed.get("answer", ""))):
            v.ok = False
            v.problems.append("The answer tells the user what to do or predicts an outcome; explain the law and options only.")
        attempts.append({"attempt": attempt + 1, "ok": v.ok, "problems": v.problems, "claims": v.claims,
                         "llm_latency_s": res.latency_s, "prompt_tokens": res.usage.get("prompt_tokens"),
                         "total_s": round(time.perf_counter() - t0, 2), "cached": res.cached})
        if v.ok:
            d = v.parsed
            mode = "insufficient" if d.get("insufficient") and not d.get("claims") else \
                ("generated" if attempt == 0 else "generated_after_retry")
            labels = sorted({l for c in v.claims for l in c["sources"]}, key=lambda x: int(x[1:]))
            return {"mode": mode, "answer": d.get("answer", ""), "claims": v.claims, "cited_labels": labels,
                    "caveats": [c for c in d.get("caveats", []) if isinstance(c, str)], "attempts": attempts,
                    "dropped_for_context": dropped}
        problems = v.problems
    fb = (_advice_fallback(sources) if advice else None) or (_historical_fallback(sources) if historical else None)
    if fb is None:
        cited = [l for a in attempts for c in a["claims"] for l in c.get("sources", [])]
        cited = sorted(set(cited), key=lambda l: (-cited.count(l), int(l[1:]) if l[1:].isdigit() else 99))
        src, passage = _best_passage(question, sources, cited) if sources else (None, "")
        fb = {"mode": "extractive_fallback",
              "answer": ("I could not produce a summary that passed verification, so here is the relevant text of the law"
                         + (f" ({src['instrument']}, {src['locator']})" if src else "") + f":\n\n\"{passage}\""),
              "claims": [], "cited_labels": [src["label"]] if src else [],
              "caveats": ["This is a direct quotation; read it with the notes below or ask the Rental Disputes Center."]}
    return {**fb, "attempts": attempts, "dropped_for_context": dropped}
