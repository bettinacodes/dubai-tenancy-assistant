"""Rule-based intent classifier: the transparent baseline the fine-tuned DistilBERT must beat.

Order encodes priority: crisis > out_of_scope > advice_seeking > calculation > procedural > factual.
"""
from __future__ import annotations

import re

from tenancy_rag.guardrails import crisis, scope
from tenancy_rag.understand import slots

ADVICE = re.compile(r"\b(should (i|we)|shall i|do you (think|reckon)|will i win|would i win|(chances?|likely) (of|to) (win|get)|"
                    r"is it (worth|better|a good idea|wise)|what (should|would) (i|you) do|advi[sc]e me|my case (is )?strong|"
                    r"(best|good) strategy|recommend|worth (it|suing|complaining)|how much (compensation|can i get))\b", re.I)
PROCEDURAL = re.compile(r"\b(how (do|can|to|should) (i|we|you)|how to|where (do|can|to|is)|steps?|process|procedure|documents?|"
                        r"papers?|apply|register|file|filing|submit|track|status|cancel|renew\w*|certificate|app|online|"
                        r"working hours|open|appeal|hearing|payment writ|deposit (the )?rent|trustee)\b", re.I)
# note: "%" sits outside the \b...\b group - "\b%" can never match (no word boundary before "%")
CALC_WORDS = re.compile(r"%|\b(calculat\w*|work out|how much|maximum|max|cap|percent|increase|fee|days?|notice|enough|"
                        r"by what date|timing|on time|in time)\b", re.I)
UNRELATED = re.compile(r"\b(poem|joke|weather|recipe|football|movie|salary|dewa bill|best area|school)\b", re.I)


def classify(text: str) -> str:
    if crisis.detect(text)["crisis"]:
        return "crisis"
    sc = scope.check(text)
    if not sc["in_scope"] or UNRELATED.search(text):
        return "out_of_scope"
    if ADVICE.search(text):
        return "advice_seeking"
    s = slots.extract(text)
    has_numbers = ("current_rent" in s) or ("annual_rent" in s and re.search(r"\bfee\b", text, re.I)) \
        or "shortfall_pct" in s or ("notice_date" in s) or ("date" in s and re.search(r"notice|expir|end", text, re.I)) \
        or bool(re.search(r"\b\d+\s*days?\b", text, re.I))
    if has_numbers and CALC_WORDS.search(text):
        return "calculation"
    if PROCEDURAL.search(text):
        return "procedural"
    return "factual"
