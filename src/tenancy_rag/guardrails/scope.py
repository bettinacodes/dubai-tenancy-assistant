"""Scope rules: what the verified corpus can and cannot answer (residential tenancy, Dubai).

Rules fire on clear signals; ambiguous cases are left to the intent classifier. Out-of-scope is a
design boundary, not a refusal to help: the response explains the scope and escalates.
"""
from __future__ import annotations

import re

OTHER_EMIRATES = re.compile(r"\b(abu\s*dhabi|al\s*ain|sharjah|ajman|ras\s*al\s*khaimah|rak|fujairah|umm\s*al\s*quwain|uaq)\b", re.I)
COMMERCIAL = re.compile(r"\b(shop|office|warehouse|showroom|retail (unit|space)|commercial (lease|property|unit|premises|tenancy)|"
                        r"business premises|my (company|business|trading business)|labou?r camp)\b", re.I)
PURCHASE = re.compile(r"\b(buy(ing)?|purchase|mortgage|off[- ]plan|golden visa|title deed transfer)\b", re.I)
DRAFTING = re.compile(r"\b(write|draft|prepare|create|make|generate)\b[^.?!]{0,25}\b(notice|letter|contract|agreement|template|complaint)\b", re.I)
SPECULATIVE = re.compile(r"\b(will|going to)\b[^.?!]{0,40}\b(next year|in 20[3-9]\d|in the future|20(2[7-9]|[3-9]\d))\b", re.I)
DIFC = re.compile(r"\bDIFC\b|dubai international financial cent", re.I)


def check(text: str) -> dict:
    reasons = []
    if OTHER_EMIRATES.search(text):
        reasons.append("other_emirate")
    if COMMERCIAL.search(text):
        reasons.append("commercial_lease")
    if PURCHASE.search(text) and not re.search(r"\b(sell|sale|sold)\b", text, re.I):
        reasons.append("property_purchase")
    if DRAFTING.search(text):
        reasons.append("document_drafting")
    if SPECULATIVE.search(text):
        reasons.append("speculative_future")
    return {"in_scope": not reasons, "reasons": reasons, "difc_caveat": bool(DIFC.search(text))}
