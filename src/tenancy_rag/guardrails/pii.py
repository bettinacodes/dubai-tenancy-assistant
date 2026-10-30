"""PII redaction before anything is logged (queries are never stored raw).

Regex-based (Presidio was cut for time). Covers the identifiers tenants
actually type: phone numbers, e-mails, Emirates ID, IBAN, passport-like numbers, and names
introduced by "my name is ...". Free-text names elsewhere are a documented limitation.
"""
from __future__ import annotations

import re

PATTERNS = [
    ("EMIRATES_ID", re.compile(r"\b784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d\b")),
    ("IBAN", re.compile(r"\bAE\d{2}(?:\s?\d{4}){4}\s?\d{3}\b", re.I)),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+971|00971|0)[\s-]?(?:5[024568]|[2-9])[\s-]?\d{3}[\s-]?\d{4}(?!\d)")),
    ("PASSPORT", re.compile(r"\b(?:passport(?:\s+(?:no\.?|number))?\s*[:#]?\s*)([A-Z]{1,2}\d{6,8})\b", re.I)),
    ("NAME", re.compile(r"\b(?i:my name is|i am called|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted text, counts per PII type)."""
    counts: dict[str, int] = {}
    for label, pat in PATTERNS:
        def sub(m: re.Match, label=label) -> str:
            counts[label] = counts.get(label, 0) + 1
            if m.groups():  # keep the introducing words, replace only the captured value
                return m.group(0).replace(m.group(1), f"[{label}]")
            return f"[{label}]"
        text = pat.sub(sub, text)
    return text, counts
