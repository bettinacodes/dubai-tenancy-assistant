"""Article-level parser for Dubai Legislation Portal pages (Word-exported HTML).

Two layouts occur on the portal:
  * Laws:    <h1> chapter, <p class="Article..."> article heading, <p class="Article..."> "Article (n)"
  * Decrees: <h1> article heading, <h1> "Article (n)"
Headings are buffered until an "Article (n)" label arrives; the last buffered heading is
the article heading, and an earlier <h1> (law layout) is the chapter.

One record is produced per legal article: the natural retrieval and citation unit.
Amending laws (Law 33/2008) quote replacement text for articles of the original law; those
quoted articles are emitted as the *current* text of the original article, and the original
versions are marked superseded (see link_amendments).
"""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from tenancy_rag.preprocess.normalise import normalise

ARTICLE_LABEL = re.compile(r"^Article\s*\(\s*(\d+)\s*\)\s*$")
ARABIC_PREVAILS = re.compile(r"Arabic text (will )?prevail", re.I)


def read_html(path: Path) -> str:
    raw = path.read_bytes()
    m = re.search(rb"charset=([\w-]+)", raw[:2000])
    enc = m.group(1).decode() if m else "utf-8"
    return raw.decode(enc, errors="replace")


def _blocks(soup: BeautifulSoup):
    """Yield (kind, text) in document order; kind in {label, h, artclass, body}."""
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.find_parent(["p", "li"]):  # nested block; parent already emitted
            continue
        # get_text("") keeps words split across Word <span>s intact ("Centr"+"e")
        text = normalise(re.sub(r"\s+", " ", el.get_text("")))
        if not text:
            continue
        if ARTICLE_LABEL.match(text):
            yield "label", text
        elif el.name in ("h1", "h2", "h3"):
            yield "h", text
        elif " ".join(el.get("class", [])).startswith("Article"):
            yield "artclass", text
        else:
            yield "body", text


def _split_articles(path: Path):
    soup = BeautifulSoup(read_html(path), "html.parser")
    blocks = list(_blocks(soup))
    law_layout = any(kind == "artclass" for kind, _ in blocks)

    articles, preamble, headings = [], [], []
    chapter, current, arabic_prevails = None, None, False
    for kind, text in blocks:
        if ARABIC_PREVAILS.search(text):
            arabic_prevails = True
            continue
        if kind in ("h", "artclass"):
            headings.append((kind, text))
            continue
        if kind == "label":
            heading = None
            if law_layout:
                for k, t in headings:
                    if k == "h":
                        chapter = t
                    else:
                        heading = t
            elif headings:
                heading = headings[-1][1]
            headings = []
            current = {"article_no": int(ARTICLE_LABEL.match(text).group(1)),
                       "article_heading": heading, "chapter": chapter, "paras": []}
            articles.append(current)
            continue
        # body paragraph: any headings buffered without a following label are body text
        paras = [t for _, t in headings] + [text]
        headings = []
        if current is None:
            preamble.extend(paras)
        else:
            current["paras"].extend(paras)
    return articles, {"preamble": "\n".join(preamble), "arabic_prevails_notice": arabic_prevails}


def _record(source: dict, a: dict, *, parent_id: str, instrument: str, short: str, extra: dict | None = None) -> dict:
    rec = {
        "parent_id": parent_id,
        "instrument": instrument,
        "instrument_short": short,
        "authority_tier": source["authority_tier"],
        "jurisdiction": source["jurisdiction"],
        "domain": "residential",
        "chapter": a["chapter"],
        "article_no": a["article_no"],
        "locator": f"Art. {a['article_no']}",
        "article_heading": a["article_heading"],
        "text": "\n".join(a["paras"]),
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


def parse_legislation(path: Path, source: dict) -> tuple[list[dict], dict]:
    """Return (article records, document metadata)."""
    articles, meta = _split_articles(path)
    amends = source.get("amends")
    if not amends:
        recs = [_record(source, a, parent_id=f"{source['id']}__art-{a['article_no']}",
                        instrument=source["instrument"], short=source["instrument_short"])
                for a in articles]
        return recs, meta

    # Amending law: [own Art 1] + [quoted replacement articles of the original law] + [own final articles]
    expected = source["amended_articles"]
    n_own_tail = source.get("own_tail_articles", 1)
    own_head, quoted, own_tail = articles[:1], articles[1:len(articles) - n_own_tail], articles[len(articles) - n_own_tail:]
    got = [a["article_no"] for a in quoted]
    if got != expected:
        raise ValueError(f"{source['id']}: quoted articles {got} != expected {expected}")

    recs = [_record(source, a, parent_id=f"{source['id']}__art-{i}", instrument=source["instrument"],
                    short=source["instrument_short"])
            for i, a in enumerate(own_head + own_tail, start=1)]
    for a in quoted:
        recs.append(_record(
            source, a,
            parent_id=f"{amends}__art-{a['article_no']}__as-amended-{source['instrument_short'].replace('/', '-')}",
            instrument=source["amends_instrument"] + f" (as amended by {source['instrument_short']})",
            short=f"{source['amends_short']} (am. {source['instrument_short']})",
            extra={"supersedes": [f"{amends}__art-{a['article_no']}"], "amends_article": a["article_no"]},
        ))
    return recs, meta


def link_amendments(records: list[dict]) -> list[dict]:
    """Mark original articles replaced by an amending law as superseded (kept, never cited as current)."""
    by_id = {r["parent_id"]: r for r in records}
    for r in records:
        for old_id in r.get("supersedes", []):
            old = by_id.get(old_id)
            if old is None:
                raise KeyError(f"{r['parent_id']} supersedes unknown {old_id}")
            r["chapter"] = r["chapter"] or old["chapter"]
            r["article_heading"] = r["article_heading"] or old["article_heading"]
            old["status"] = "superseded"
            old["superseded_by"].append(r["parent_id"])
            old["amended_by"].append(r["instrument_short"])
    return records
