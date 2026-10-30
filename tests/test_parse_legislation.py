"""Parser QA on the cached corpus: completeness, contiguity and amendment linkage."""
import pytest

from tenancy_rag import config
from tenancy_rag.ingest.fetch import load_sources
from tenancy_rag.ingest.parse_legislation import link_amendments, parse_legislation

SOURCES = {s["id"]: s for s in load_sources()}


def _parse(source_id):
    src = SOURCES[source_id]
    path = next((config.RAW / source_id).glob("*.html"))
    return parse_legislation(path, src)


@pytest.fixture(scope="module")
def corpus():
    recs = []
    for sid, src in SOURCES.items():
        if src["parser"] == "legislation_html":
            recs += _parse(sid)[0]
    return link_amendments(recs)


def test_law_26_2007_has_all_37_articles_contiguous():
    recs, meta = _parse("dubai-law-26-2007")
    assert [r["article_no"] for r in recs] == list(range(1, 38))
    assert meta["arabic_prevails_notice"]


@pytest.mark.parametrize("sid,n", [("dubai-decree-43-2013", 4), ("dubai-decree-26-2013", 31)])
def test_decrees_complete(sid, n):
    recs, _ = _parse(sid)
    assert [r["article_no"] for r in recs] == list(range(1, n + 1))


def test_no_empty_articles(corpus):
    assert all(r["text"].strip() for r in corpus)


def test_parent_ids_unique(corpus):
    ids = [r["parent_id"] for r in corpus]
    assert len(ids) == len(set(ids))


def test_amended_articles_superseded_and_linked(corpus):
    by_id = {r["parent_id"]: r for r in corpus}
    for n in SOURCES["dubai-law-33-2008"]["amended_articles"]:
        old = by_id[f"dubai-law-26-2007__art-{n}"]
        assert old["status"] == "superseded"
        new = by_id[old["superseded_by"][0]]
        assert new["status"] == "in_force" and new["article_no"] == n


def test_amended_art_25_has_12_month_notarised_notice(corpus):
    art = next(r for r in corpus if r["parent_id"].startswith("dubai-law-26-2007__art-25__as-amended"))
    assert "twelve (12) months" in art["text"] and "Notary Public" in art["text"]


def test_decree_43_bands_text():
    art1 = _parse("dubai-decree-43-2013")[0][0]
    for phrase in ["ten percent (10%)", "five percent (5%)", "fifteen percent (15%)", "twenty percent (20%)"]:
        assert phrase in art1["text"]
