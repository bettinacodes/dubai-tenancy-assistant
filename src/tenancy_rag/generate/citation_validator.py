"""Citation validator - the component that makes the system more than a demo.

Day 3 showed LLMs citing laws that do not exist in 46 of 46 citations (7B). Every generated
answer is therefore checked before it is shown:
  1. structure     - valid JSON, at least one claim, every claim cites >= 1 label
  2. existence     - every cited label was actually provided (no invented sources)
  3. currency      - no superseded source cited as current law (allowed only in historical mode)
  4. authority     - a claim may not rest only on guidance flagged as misstating the law
  5. entailment    - each claim must be ENTAILED by at least one of its cited sources
                     (NLI cross-encoder, max over source windows, threshold tau)
  6. numbers       - every number in the answer must appear in the sources, the calculator
                     result or the user's question (no invented figures)
Failures are returned as human-readable problems used for one regeneration attempt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from tenancy_rag import config

ENTAIL_THRESHOLD = 0.5
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")


@dataclass
class Validation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)   # each with entail score and pass flag
    parsed: dict | None = None


@lru_cache(maxsize=1)
def _nli():
    import torch
    from sentence_transformers import CrossEncoder
    device = config.NLI_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    m = CrossEncoder(config.NLI_MODEL, device=device, max_length=512)
    labels = [m.model.config.id2label[i].lower() for i in range(len(m.model.config.id2label))]
    return m, labels.index("entailment")


_ITEM = re.compile(r"^\s*([a-z]|\d{1,2}|[ivx]+)[.)]\s")
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
                 "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30",
                 "forty": "40", "sixty": "60", "ninety": "90", "hundred": "100"}


def premise_windows(text: str) -> list[str]:
    """Structure-aware premise windows. Enumerated legal text is read the way a
    lawyer reads it: lead-in sentence (line ending ':') + one clause + any closing 'For the purposes
    of ...' lines of the same block. Unstructured text also gets 2- and 3-line sliding windows.
    Scoring a claim against a whole article fails (512-token truncation, mixed clauses)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    wins, chapeau, block, structured = [], None, [], False

    def flush(notes):
        for it in block:
            wins.append(" ".join(x for x in (chapeau, it, *notes) if x))

    for l in lines:
        if l.lower().startswith("for the purposes of"):
            flush([l])
            block.clear()
            continue
        if l.endswith(":"):
            flush([])
            block.clear()
            chapeau = l
            continue
        if _ITEM.match(l) and chapeau:
            block.append(l)
            structured = True
            continue
        wins.append(l)
    flush([])
    if not structured:
        for k in (2, 3):
            wins += [" ".join(lines[i:i + k]) for i in range(len(lines) - k + 1)]
    out = []
    for w in dict.fromkeys(wins):            # long single lines: split to stay inside 512 tokens
        words = w.split()
        out += [" ".join(words[i:i + 220]) for i in range(0, len(words), 180)] if len(words) > 220 else [w]
    return out


def claim_numbers(text: str) -> set[str]:
    low = text.lower()
    nums = {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", low)}
    nums |= {d for w, d in _NUMBER_WORDS.items() if re.search(rf"\b{w}\b", low)}
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in nums}


def entailment(claim: str, premise_texts: list[str], return_window: bool = False):
    """Max entailment probability over premise windows that contain every number in the claim
    (numeracy guard: NLI models ignore number mismatches - '11-20% -> 10%' scored 0.96 against the
    clause that says 5%)."""
    import numpy as np
    model, ent = _nli()
    need = claim_numbers(claim)
    wins = [w for t in premise_texts for w in premise_windows(t)]
    wins = [w for w in wins if need <= claim_numbers(w)]
    if not wins:
        return (0.0, None) if return_window else 0.0
    probs = np.asarray(model.predict([(w, claim) for w in wins], apply_softmax=True, show_progress_bar=False))[:, ent]
    best = int(np.argmax(probs))
    return (float(probs[best]), wins[best]) if return_window else float(probs[best])


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip("0").rstrip(".") if "." in n else n.replace(",", "") for n in _NUM.findall(text)}


def parse(raw: str) -> dict | None:
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return d if isinstance(d, dict) else None


def upgrade_citations(claims: list[dict], sources: list[dict]) -> None:
    """Prefer binding law over guidance. A small generator often cites the FAQ
    answer that paraphrases a law even when the law itself is in context. For every verified claim
    that rests only on guidance, add the in-force legislation/decree source that ALSO entails it
    (same NLI + numeracy check). Citations are only ever added after verification, never guessed."""
    binding = [s for s in sources if s["authority_tier"] in ("legislation", "decree") and s["status"] == "in_force"]
    by_label = {s["label"]: s for s in sources}
    for c in claims:
        if not c["pass"] or not c["sources"]:
            continue
        if any(by_label[l]["authority_tier"] in ("legislation", "decree") for l in c["sources"] if l in by_label):
            continue
        for s in binding:
            if entailment(c["text"], [s["text"]]) >= ENTAIL_THRESHOLD:
                c["sources"] = [s["label"]] + c["sources"]
                c["upgraded_to"] = s["label"]
                break


def validate(raw: str, sources: list[dict], question: str, tool: dict | None = None,
             historical: bool = False, check_entailment: bool = True) -> Validation:
    d = parse(raw)
    if d is None:
        return Validation(False, ["Output was not valid JSON."])
    by_label = {s["label"]: s for s in sources}
    problems, checked = [], []
    if d.get("insufficient") and not d.get("claims"):
        return Validation(True, [], [], d)       # an honest "I can't answer from these sources" passes
    claims = d.get("claims") or []
    if not isinstance(claims, list) or not claims:
        problems.append("No claims with citations were given.")
    for c in claims if isinstance(claims, list) else []:
        text = str(c.get("text", "")).strip()
        labels = [l for l in (c.get("sources") or []) if isinstance(l, str)]
        rec = {"text": text, "sources": labels, "pass": True, "entailment": None, "reasons": []}
        if not labels:
            rec["reasons"].append("no citation")
        unknown = [l for l in labels if l not in by_label]
        if unknown:
            rec["reasons"].append(f"cites unknown source(s) {unknown}")
        known = [by_label[l] for l in labels if l in by_label]
        if not historical and any(s["status"] == "superseded" for s in known):
            rec["reasons"].append("cites superseded law as current")
        if known and all(s.get("flagged") for s in known):
            rec["reasons"].append("relies only on guidance known to misstate the law")
        if check_entailment and known and text:
            rec["entailment"] = round(entailment(text, [s["text"] for s in known]), 3)
            if rec["entailment"] < ENTAIL_THRESHOLD:
                rec["reasons"].append(f"not supported by the cited text (entailment {rec['entailment']})")
        rec["pass"] = not rec["reasons"]
        if not rec["pass"]:
            problems.append(f'Claim "{text[:120]}": ' + "; ".join(rec["reasons"]))
        checked.append(rec)
    if check_entailment:
        upgrade_citations(checked, sources)
    allowed = _numbers(question) | {n for s in sources for n in _numbers(s["text"] + " " + s.get("instrument", "") + " " + s.get("locator", ""))}
    if tool:
        allowed |= _numbers(" ".join(tool["working"]) + " " + json.dumps(tool["result"]))
    stray = sorted(n for n in _numbers(str(d.get("answer", ""))) if n not in allowed and len(n) > 1)
    if stray:
        problems.append(f"The answer contains numbers not found in the sources or calculator: {stray}")
    return Validation(not problems, problems, checked, d)
