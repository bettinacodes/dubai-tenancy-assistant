"""Crisis detection: situations where an essay about the law is the wrong response.

Lock-outs, threats/violence, removal of belongings and retention of identity documents always
route to the escalation card (verified contacts, data/escalation_contacts.json). A utility cut
alone is a legal question (Law 26/2007 Art. 34) - it becomes a crisis only with an urgency or
vulnerability marker (child, illness, heat, tonight, unsafe ...).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from tenancy_rag import config

ALWAYS = [
    r"\b(changed|change|changing)\b[^.?!]{0,20}\blocks?\b|\block(ed)? (me|us) out\b|\blocks? (is |was )?changed\b",
    r"\b(threw|thrown|throwing|tossed|dumped|removed|took|taken|throw)\b[^.?!]{0,25}\b(stuff|things|belongings|furniture|luggage|door)\b",
    r"\b(holding|keeping|kept|took|taken|retain\w*|seized|withholding)\b[^.?!]{0,20}\b(passport|emirates id)\b",
    r"\b(passport|emirates id)\b[^.?!]{0,30}\b(won'?t|will not|refus\w*)\b[^.?!]{0,20}\b(give|return)",
    r"\b(threat\w*|beat|hit|attack\w*|violen\w*|kill|harass\w*|assault\w*)\b",
    r"\b(nowhere|no where|no place) to (sleep|stay|go)\b",
    r"\b(kicked|forced|forcing|throwing|throw) (me|us) out\b[^.?!]{0,30}\b(today|tonight|now|right now)\b",
    r"\b(at|outside) my door\b[^.?!]{0,40}\b(men|people|guys)\b|\bmen\b[^.?!]{0,30}\b(forcing|kick)",
    r"\bcancel my visa\b",
]
UTILITY = r"\b(cut|cutting|switched off|turned off|disconnect\w*)\b[^.?!]{0,30}\b(water|electricity|power|ac|a/c|air ?con\w*|gas)\b"
URGENCY = r"\b(baby|child|children|kids?|pregnant|sick|ill|elderly|old (mother|father|parent)|disab\w*|medicine|4\d ?degree|heat|tonight|right now|unsafe|danger)\b"
_ALWAYS = [re.compile(p, re.I) for p in ALWAYS]


def detect(text: str) -> dict:
    hits = [p.pattern[:40] for p in _ALWAYS if p.search(text)]
    if re.search(UTILITY, text, re.I) and re.search(URGENCY, text, re.I):
        hits.append("utility_cut+vulnerability")
    return {"crisis": bool(hits), "signals": hits}


@lru_cache(maxsize=1)
def escalation_card() -> dict:
    """Verified contacts only (each quoted from a cached official page; see tests)."""
    c = json.loads((config.DATA / "escalation_contacts.json").read_text(encoding="utf-8"))
    return {"title": "If you are in danger or locked out, contact these now",
            "contacts": [{k: x[k] for k in ("label", "value", "when", "url")} for x in c["contacts"]],
            "verified_on": c["verified_on"]}
