"""Structure-aware chunking with parent-document retrieval, plus a fixed-window baseline.

Structure-aware (the system): one parent = one legal article / FAQ pair / guidance section.
Parents up to MAX_SINGLE tokens become a single chunk. Longer parents are split into
~WINDOW-token windows with ~OVERLAP tokens of overlap, cut only at clause/sentence
boundaries, so no obligation is cut mid-sentence. Chunks are what we embed; the full
parent is what the generator sees (parent-document expansion at retrieval time).

Contextual prefix: "{instrument_short} - {chapter} - {locator} ({heading}): " is prepended to
the text that is embedded/indexed (stored separately so ablation 6 can switch it off).

Fixed-window baseline (ablation 5): each source document's parents are concatenated and cut
into FIXED-token windows with no regard to structure; each window records which parents it
overlaps so article-level gold labels can still be scored.

Token counts use the bge-m3 tokenizer (the embedding model's own tokenizer).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from tenancy_rag import config

MAX_SINGLE = 400
WINDOW = 250
OVERLAP = 50
FIXED = 512
FIXED_OVERLAP = 0


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def n_tokens(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


@lru_cache(maxsize=1)
def _sentencizer():
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def context_prefix(p: dict) -> str:
    """Compact code-style prefix: 'L26/2007 (am. L33/2008) - Eviction Cases - Art. 25: '."""
    parts = [p["instrument_short"]]
    if p.get("chapter"):
        parts.append(p["chapter"])
    loc = p.get("locator") or ""
    if p.get("article_heading") and not p["parent_id"].split("__")[-1].startswith("q-"):
        loc = f"{loc} ({p['article_heading']})"
    parts.append(loc)
    return " - ".join(x for x in parts if x) + ": "


def natural_prefix(p: dict) -> str:
    """Natural-language prefix for embedding models: full instrument name and headings, no codes.
    'Dubai Law No. 26 of 2007 ... (as amended by L33/2008), Eviction Cases, Article 25: '
    FAQ answers already contain their question, so they get only the source and section."""
    is_faq = p["parent_id"].split("__")[-1].startswith("q-")
    parts = [p["instrument"]]
    if p.get("chapter"):
        parts.append(p["chapter"])
    if p.get("article_no") is not None:
        parts.append(f"Article {p['article_no']}" + (f" ({p['article_heading']})" if p.get("article_heading") else ""))
    elif p.get("article_heading") and not is_faq:
        parts.append(p["article_heading"])
    return ", ".join(parts) + ": "


def _units(text: str) -> list[str]:
    """Split into clause lines; a line still longer than WINDOW is split into sentences."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if n_tokens(line) <= WINDOW:
            units.append(line)
        else:
            units.extend(s.text.strip() for s in _sentencizer()(line).sents if s.text.strip())
    return units


def _windows(units: list[str]) -> list[list[str]]:
    sizes = [n_tokens(u) for u in units]
    out, start = [], 0
    while start < len(units):
        end, total = start, 0
        while end < len(units) and (total + sizes[end] <= WINDOW or end == start):
            total += sizes[end]
            end += 1
        out.append(units[start:end])
        if end >= len(units):
            break
        # step back over trailing units worth >= OVERLAP tokens (but always make progress)
        back, ov = end, 0
        while back > start + 1 and ov < OVERLAP:
            back -= 1
            ov += sizes[back]
        start = back
    return out


def chunk_parent(p: dict) -> list[dict]:
    prefix, prefix_nl = context_prefix(p), natural_prefix(p)
    base = {k: p[k] for k in ("parent_id", "instrument_short", "authority_tier", "jurisdiction", "status",
                              "locator", "article_no", "source_url")}
    if n_tokens(p["text"]) <= MAX_SINGLE:
        pieces = [p["text"]]
    else:
        pieces = ["\n".join(w) for w in _windows(_units(p["text"]))]
    return [{**base, "chunk_id": f"{p['parent_id']}__c{i}", "prefix": prefix, "prefix_nl": prefix_nl, "text": t,
             "n_tokens": n_tokens(prefix + t)} for i, t in enumerate(pieces)]


def fixed_windows(parents: list[dict]) -> list[dict]:
    """Structure-blind baseline: per source document, concatenate and cut every FIXED tokens."""
    tok = _tokenizer()
    by_doc: dict[str, list[dict]] = {}
    for p in parents:
        by_doc.setdefault(p["instrument_short"].split(" (am.")[0], []).append(p)
    out = []
    for doc, ps in by_doc.items():
        text, spans = "", []
        for p in ps:
            start = len(text)
            text += p["text"] + "\n\n"
            spans.append((start, len(text), p["parent_id"]))
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        step = FIXED - FIXED_OVERLAP
        for i, s in enumerate(range(0, len(offs), step)):
            window = offs[s:s + FIXED]
            a, b = window[0][0], window[-1][1]
            covered = [pid for (x, y, pid) in spans if x < b and y > a]
            src = next(p for p in ps if p["parent_id"] == covered[0])
            out.append({"chunk_id": f"fixed512__{doc.replace('/', '-')}__w{i}", "parent_ids": covered,
                        "parent_id": covered[0], "instrument_short": doc,
                        "authority_tier": src["authority_tier"], "jurisdiction": src["jurisdiction"],
                        "status": "in_force" if any(
                            next(p for p in ps if p["parent_id"] == c)["status"] == "in_force" for c in covered)
                        else "superseded",
                        "prefix": "", "text": text[a:b], "n_tokens": len(window)})
    return out


def build_chunks() -> tuple[list[dict], list[dict]]:
    parents = [json.loads(l) for l in (config.PROCESSED / "parents.jsonl").open(encoding="utf-8")]
    chunks = [c for p in parents for c in chunk_parent(p)]
    fixed = fixed_windows(parents)
    for path, rows in ((config.CHUNKS, chunks), (config.PROCESSED / "chunks_fixed512.jsonl", fixed)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return chunks, fixed


if __name__ == "__main__":
    chunks, fixed = build_chunks()
    multi = {c["parent_id"] for c in chunks if not c["chunk_id"].endswith("__c0")}
    print(f"structure-aware chunks: {len(chunks)} (parents split: {len(multi)}); fixed-512 windows: {len(fixed)}")
    print("max chunk tokens:", max(c["n_tokens"] for c in chunks))
