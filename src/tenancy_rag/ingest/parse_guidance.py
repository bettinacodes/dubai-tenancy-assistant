"""Parsers for official guidance (non-legislative) sources.

Guidance has no article numbering, so each parser picks the natural citation unit:
  * faq_html       one record per question-answer pair (DLD and RDC FAQ accordions)
  * guidance_html  one record per headed section of a page (trafilatura main-text extraction)
  * guidance_pdf   one record per page, with chapter metadata, running headers stripped
                   and dropped ligatures repaired
Every record carries a human-readable `locator` used in citation chips ("Q14", "p. 8", ...).
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import trafilatura
from bs4 import BeautifulSoup

from tenancy_rag.preprocess.normalise import normalise, repair_dropped_ligatures


def _base(source: dict, *, parent_id: str, locator: str, chapter: str | None, heading: str | None,
          text: str, extra: dict | None = None) -> dict:
    rec = {
        "parent_id": parent_id,
        "instrument": source["instrument"],
        "instrument_short": source["instrument_short"],
        "authority_tier": source["authority_tier"],
        "jurisdiction": source["jurisdiction"],
        "domain": "residential",
        "chapter": chapter,
        "article_no": None,
        "locator": locator,
        "article_heading": heading,
        "text": text,
        "in_force_from": source.get("in_force_from"),
        "amended_by": [],
        "supersedes": [],
        "superseded_by": [],
        "status": "in_force",
        "source_url": source["url"],
        "lang": "en",
        "translation_authoritative": False,
    }
    rec.update(extra or {})
    return rec


def _clean(el) -> str:
    return normalise(re.sub(r"[ \t\r\f\v]+", " ", el.get_text("\n")).replace(" \n", "\n"))


def parse_faq_html(path: Path, source: dict) -> list[dict]:
    """Accordion FAQ pages. `faq_sections` (optional) keeps only the named sections."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    keep = set(source.get("faq_sections") or [])
    section, recs, n = None, [], 0
    for el in soup.select(".content-section-title, div.accordion-item"):
        if "content-section-title" in el.get("class", []):
            section = normalise(el.get_text(" "))
            continue
        if keep and section not in keep:
            continue
        q_el, a_el = el.select_one(".accordion-header"), el.select_one(".accordion-body")
        if not (q_el and a_el):
            continue
        q, a = normalise(re.sub(r"\s+", " ", q_el.get_text(" "))), _clean(a_el)
        if not a:
            continue
        n += 1
        recs.append(_base(source, parent_id=f"{source['id']}__q-{n}", locator=f"Q{n}", chapter=section,
                          heading=q, text=f"Q: {q}\nA: {a}"))
    return recs


_HEADING = re.compile(r"^[A-Z][^.:;!?]{3,90}$")


def parse_guidance_html(path: Path, source: dict) -> list[dict]:
    """Main-text extraction with trafilatura (drops nav/boilerplate), split at short heading lines."""
    text = trafilatura.extract(path.read_text(encoding="utf-8", errors="replace"), include_tables=True,
                               favor_precision=True) or ""
    lines = [normalise(l) for l in text.splitlines() if l.strip()]
    sections, heading, buf = [], None, []
    for line in lines:
        is_heading = _HEADING.match(line) and len(line.split()) <= 12 and not line.endswith(",")
        if is_heading and buf:
            sections.append((heading, buf))
            heading, buf = line, []
        elif is_heading and not buf:
            heading = line if heading is None else f"{heading} - {line}"
        else:
            buf.append(line)
    if buf:
        sections.append((heading, buf))
    # Table cells mis-detected as headings leave tiny sections ("All"): fold them into the previous one
    merged: list[tuple[str | None, list[str]]] = []
    for h, b in sections:
        if merged and len(" ".join(b)) < 60:
            merged[-1][1].extend(([h] if h else []) + b)
        else:
            merged.append((h, list(b)))
    sections = merged
    return [_base(source, parent_id=f"{source['id']}__s-{i}", locator=f"§{i}", chapter=None, heading=h,
                  text="\n".join(b))
            for i, (h, b) in enumerate(sections, start=1)]


def _page_ranges(spec: list[list[int]]) -> set[int]:
    return {p for a, b in spec for p in range(a, b + 1)}


def parse_guidance_pdf(path: Path, source: dict, vocab: set[str]) -> tuple[list[dict], dict]:
    """One record per kept page. `keep_pages`: [[a, b], ...] (1-based, inclusive);
    `chapters`: {"Chapter title": [a, b]}; `superseded_pages` + `superseded_by` mark outdated pages."""
    keep = _page_ranges(source["keep_pages"]) if source.get("keep_pages") else None
    chapters = {title: _page_ranges([rng]) for title, rng in (source.get("chapters") or {}).items()}
    outdated = _page_ranges(source.get("superseded_pages", []))
    with pdfplumber.open(path) as pdf:
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]

    # Running headers: first lines that recur on 3+ pages ("Tenancy Guide", "How to Register", ...)
    first_lines = [t.strip().splitlines()[0] for _, t in pages if t.strip()]
    running = {l for l in set(first_lines) if first_lines.count(l) >= 3}

    recs, all_repairs = [], {}
    for pno, raw in pages:
        if keep is not None and pno not in keep:
            continue
        lines = raw.strip().splitlines()
        if lines and lines[0] in running:
            lines = lines[1:]
        lines = [l for l in lines if not re.fullmatch(r"\s*\d{1,3}\s*", l)]  # bare page numbers
        # Reflow hard PDF line wraps: a break followed by a lower-case word continues the sentence
        text = normalise(re.sub(r"\n(?=[a-z(])", " ", "\n".join(lines)))
        text, repairs = repair_dropped_ligatures(text, vocab)
        all_repairs.update(repairs)
        if len(text) < 80:  # chapter title pages / blank pages
            continue
        chapter = next((t for t, rng in chapters.items() if pno in rng), None)
        extra = {"status": "superseded", "superseded_by": [source["superseded_by"]]} if pno in outdated else None
        recs.append(_base(source, parent_id=f"{source['id']}__p-{pno}", locator=f"p. {pno}", chapter=chapter,
                          heading=None, text=text, extra=extra))
    return recs, {"ligature_repairs": all_repairs, "running_headers": sorted(running)}
