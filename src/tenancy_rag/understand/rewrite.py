"""Query rewriting: normalise a non-English / code-mixed / very informal question into a clear
English retrieval query, keeping every number, date and named entity.

bge-m3 is multilingual, so retrieval may work without this step; Day 9's parity evaluation
measures retrieval with and without rewriting (it costs one LLM call per query). The answer is
still written in the user's own language - only the retrieval query is rewritten.
"""
from __future__ import annotations

import json
import re

from tenancy_rag.llm import chat

SYSTEM = ("Rewrite the user's message as one clear English question about Dubai residential tenancy, "
          "for searching the law. Translate if it is not in English. Keep every number, amount, date and name. "
          "Do not answer it. Return JSON: {\"query\": \"...\", \"language\": \"<ISO 639-1 code of the original>\"}")


def looks_english(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text)
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.95


def rewrite(text: str, force: bool = False) -> dict:
    """Returns {"query", "language", "rewritten": bool}. English input is passed through unless forced."""
    if not force and looks_english(text):
        return {"query": text, "language": "en", "rewritten": False}
    res = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], json_mode=True, max_tokens=120)
    try:
        d = json.loads(res.text)
        return {"query": d.get("query") or text, "language": d.get("language", "und"), "rewritten": True}
    except json.JSONDecodeError:
        return {"query": text, "language": "und", "rewritten": False}
