"""Parse every cached source into parent records (one per article / FAQ pair / section / page).

Output: data/processed/parents.jsonl (+ parse_report.json with parse statistics).
"""
from __future__ import annotations

import json
import re
from collections import Counter

from tenancy_rag import config
from tenancy_rag.ingest.fetch import load_sources
from tenancy_rag.ingest.parse_guidance import parse_faq_html, parse_guidance_html, parse_guidance_pdf
from tenancy_rag.ingest.parse_legislation import link_amendments, parse_legislation
from tenancy_rag.preprocess.normalise import build_vocab

PARENTS = config.PROCESSED / "parents.jsonl"
PARSE_REPORT = config.PROCESSED / "parse_report.json"


def latest_raw(source_id: str):
    files = sorted((config.RAW / source_id).glob(f"{source_id}__*"))
    if not files:
        raise FileNotFoundError(f"no cached file for {source_id}; run the fetcher first")
    return files[-1]


FAQ_PREFERENCE = ["RDC-FAQ", "DLD-FAQ"]  # on duplicates keep the dispute body's own answer


def dedup_faq_answers(records: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """The DLD and RDC FAQs repeat many answers verbatim (sometimes under different question
    wording). Keep one record per identical answer, preferring RDC, and fold every question
    wording into it (alternative phrasings help retrieval). Returns (records, {dropped: kept})."""
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["instrument_short"] in FAQ_PREFERENCE:
            groups.setdefault(norm(r["text"].split("\nA:", 1)[-1]), []).append(r)
    dropped: dict[str, str] = {}
    for grp in groups.values():
        if len(grp) < 2:
            continue
        grp.sort(key=lambda r: (FAQ_PREFERENCE.index(r["instrument_short"]), int(r["locator"][1:])))
        keep, rest = grp[0], grp[1:]
        questions = [keep["article_heading"]] + [r["article_heading"] for r in rest]
        uniq = list(dict.fromkeys(q for q in questions if q))
        answer = keep["text"].split("\nA:", 1)[-1]
        keep["text"] = "\n".join(f"Q: {q}" for q in uniq) + "\nA:" + answer
        keep["also_published_as"] = [f"{r['instrument_short']} {r['locator']}" for r in rest]
        for r in rest:
            dropped[r["parent_id"]] = keep["parent_id"]
    return [r for r in records if r["parent_id"] not in dropped], dropped


KNOWN_CONFLICTS = config.DATA / "known_conflicts.json"


def attach_quality_flags(records: list[dict]) -> list[str]:
    """Mark guidance records that misstate the binding text (data/known_conflicts.json).
    They stay retrievable, but carry `quality_flags` so retrieval can down-weight them and the
    generator/validator prefer the conflicting binding provision."""
    by_id = {r["parent_id"]: r for r in records}
    flagged = []
    for c in json.loads(KNOWN_CONFLICTS.read_text(encoding="utf-8")):
        r = by_id.get(c["parent_id"])
        if r is None:
            raise KeyError(f"known_conflicts.json: unknown parent {c['parent_id']}")
        missing = [x for x in c["conflicts_with"] if x not in by_id]
        if missing:
            raise KeyError(f"known_conflicts.json: unknown binding provision(s) {missing}")
        r["quality_flags"] = [{"type": "conflicts_with_binding_text", "conflicts_with": c["conflicts_with"],
                               "issue": c["issue"]}]
        flagged.append(c["parent_id"])
    return flagged


def build_parents() -> list[dict]:
    sources = load_sources()
    records, report = [], {"sources": {}}

    # Pass 1: HTML sources (clean text; also the vocabulary for PDF ligature repair)
    for src in sources:
        parser = src["parser"]
        if parser == "legislation_html":
            recs, meta = parse_legislation(latest_raw(src["id"]), src)
            report["sources"][src["id"]] = {"records": len(recs), "arabic_prevails_notice": meta["arabic_prevails_notice"]}
        elif parser == "faq_html":
            recs = parse_faq_html(latest_raw(src["id"]), src)
            report["sources"][src["id"]] = {"records": len(recs), "sections": src.get("faq_sections")}
        elif parser == "guidance_html":
            recs = parse_guidance_html(latest_raw(src["id"]), src)
            report["sources"][src["id"]] = {"records": len(recs)}
        else:
            continue
        records += recs

    # Pass 2: PDFs, repaired against the vocabulary of the clean HTML text
    vocab = build_vocab(r["text"] for r in records)
    for src in sources:
        if src["parser"] == "guidance_pdf":
            recs, meta = parse_guidance_pdf(latest_raw(src["id"]), src, vocab)
            report["sources"][src["id"]] = {"records": len(recs), **meta}
            records += recs
        elif src["parser"] == "reference_only":
            report["sources"][src["id"]] = {"records": 0, "excluded": src.get("exclusion_reason")}

    link_amendments(records)
    records, dropped = dedup_faq_answers(records)
    flagged = attach_quality_flags(records)
    report["quality_flagged"] = flagged
    report["faq_duplicates_removed"] = len(dropped)
    report["faq_duplicate_map"] = dropped
    report["total_parents"] = len(records)
    report["by_tier"] = Counter(r["authority_tier"] for r in records)
    report["by_status"] = Counter(r["status"] for r in records)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    with PARENTS.open("w", encoding="utf-8") as f:
        for r in records:
            r["corpus_as_at"] = config.CORPUS_AS_AT
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    PARSE_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


if __name__ == "__main__":
    recs = build_parents()
    print(json.dumps(json.loads(PARSE_REPORT.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
