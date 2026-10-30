"""Heuristic scoring of free-text (unstructured) answers, used for the no-retrieval baseline.

The full system emits structured citations, so this regex scorer is only needed where the model
writes citations in prose. It extracts (instrument, article) mentions and checks them against the
gold primary provisions, flags instruments that do not exist in the tenancy framework, and checks
calculator answers for the expected numbers. Results are complemented by manual review.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

from tenancy_rag import config

# Instruments that really exist in (or around) the Dubai residential tenancy framework.
REAL_INSTRUMENTS = {("law", 26, 2007), ("law", 33, 2008), ("decree", 43, 2013), ("decree", 26, 2013),
                    ("law", 5, 1985), ("law", 1, 2009), ("decree", 2, 1993), ("law", 16, 2007),
                    ("law", 7, 2006), ("order", 1, 2004)}

INSTR = re.compile(r"\b(Law|Decree|Order)\s*(?:No\.?|Number)?\s*\(?\s*(\d{1,3})\s*\)?\s*(?:of|/)\s*(\d{4})", re.I)
SHORT = re.compile(r"\b(?:Law|Decree)?\s*\(?(\d{1,3})\)?\s*/\s*(\d{4})")
ARTICLE = re.compile(r"\bArt(?:icle)?s?\.?\s*\(?\s*(\d{1,3})\s*\)?", re.I)


def _primary_refs(parents: dict, pid: str) -> tuple[set, int | None]:
    """Acceptable instruments and the article number for a gold primary parent."""
    p = parents[pid]
    short = p["instrument_short"]
    accept = set()
    for kind, no, yr in re.findall(r"([LD])(\d+)/(\d{4})", short):
        accept.add(({"L": "law", "D": "decree"}[kind], int(no), int(yr)))
    return accept, p.get("article_no")


# Acronym -> accepted expansions. Anything else written next to the acronym is a misnamed institution.
INSTITUTIONS = {
    "RDC": re.compile(r"rent(al)?\s+disputes?\s+(settlement\s+)?(cent(er|re)|committee)", re.I),
    "RERA": re.compile(r"real\s+estate\s+regulatory\s+(agency|authority)", re.I),
    "REST": re.compile(r"dubai\s+rest", re.I),
}


def misnamed_institutions(answer: str) -> list[str]:
    """Find 'Expansion (ACRONYM)' / 'ACRONYM (Expansion)' pairs whose expansion is wrong."""
    bad = []
    for acr, ok in INSTITUTIONS.items():
        for m in re.finditer(rf"((?:[A-Z][\w'-]*\s+){{1,5}})\(\s*{acr}\s*\)|\b{acr}\b\s*\(([^)]{{5,60}})\)", answer):
            expansion = (m.group(1) or m.group(2) or "").strip()
            if expansion and not ok.search(expansion):
                bad.append(f"{acr} = {expansion}")
    return bad


def analyse(answer: str) -> dict:
    instruments = {(k.lower(), int(n), int(y)) for k, n, y in INSTR.findall(answer)}
    articles = {int(a) for a in ARTICLE.findall(answer)}
    return {"instruments": sorted(instruments), "articles": sorted(articles),
            "unknown_instruments": sorted(i for i in instruments if i not in REAL_INSTRUMENTS),
            "misnamed_institutions": misnamed_institutions(answer)}


def _num_present(answer: str, value: float) -> bool:
    v = int(round(value))
    forms = {f"{v}", f"{v:,}"}
    return any(re.search(rf"(?<![\d,.]){re.escape(f)}(?![\d])", answer) for f in forms)


def score(run_path, gold_path=None, parents_path=None) -> dict:
    parents = {json.loads(l)["parent_id"]: json.loads(l) for l in (parents_path or config.PROCESSED / "parents.jsonl").open(encoding="utf-8")}
    gold = {g["id"]: g for g in map(json.loads, (gold_path or config.EVAL / "gold.jsonl").open(encoding="utf-8"))}
    rows = []
    for r in map(json.loads, open(run_path, encoding="utf-8")):
        if r["id"] not in gold:
            continue
        g, a = gold[r["id"]], analyse(r["answer"])
        row = {"id": r["id"], "category": g["category"], **a,
               "cites_any_article": bool(a["articles"]), "cites_any_instrument": bool(a["instruments"])}
        if g["primary_ids"]:
            hit = False
            for pid in g["primary_ids"]:
                accept, art = _primary_refs(parents, pid)
                if art in a["articles"] and accept & set(map(tuple, a["instruments"])):
                    hit = True
            row["primary_citation_correct"] = hit
        if g.get("tool"):
            e = g["tool"]["expected"]
            if "max_increase_pct" in e:
                row["numeric_correct"] = (f"{int(e['max_increase_pct'])}%" in r["answer"].replace(" %", "%")
                                          and (e["max_increase_aed"] == 0 or _num_present(r["answer"], e["max_increase_aed"])))
            elif "filing_fee" in e:
                row["numeric_correct"] = _num_present(r["answer"], e["filing_fee"])
            elif "valid" in e:
                row["numeric_correct"] = None  # judged manually (validity reasoning)
        rows.append(row)

    with_primary = [x for x in rows if "primary_citation_correct" in x]
    numeric = [x for x in rows if x.get("numeric_correct") is not None]
    summary = {
        "n_gold": len(rows),
        "cites_any_article": sum(x["cites_any_article"] for x in rows),
        "cites_any_instrument": sum(x["cites_any_instrument"] for x in rows),
        "answers_citing_unknown_instrument": sum(bool(x["unknown_instruments"]) for x in rows),
        "unknown_instruments": Counter(str(i) for x in rows for i in x["unknown_instruments"]).most_common(),
        "answers_misnaming_an_institution": sum(bool(x["misnamed_institutions"]) for x in rows),
        "misnamed_examples": sorted({m for x in rows for m in x["misnamed_institutions"]}),
        "primary_citation_correct": f"{sum(x['primary_citation_correct'] for x in with_primary)}/{len(with_primary)}",
        "numeric_correct": f"{sum(x['numeric_correct'] for x in numeric)}/{len(numeric)}",
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    res = score(sys.argv[1])
    print(json.dumps(res["summary"], indent=2))
