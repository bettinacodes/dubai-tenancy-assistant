import pytest

from tenancy_rag import config
from tenancy_rag.retrieve.citation_lookup import lookup

pytestmark = pytest.mark.skipif(not (config.PROCESSED / "parents.jsonl").exists(), reason="corpus not built")


def test_resolves_to_current_amended_text():
    assert lookup("What does Article 26 of Law 26/2007 say?") == ["dubai-law-26-2007__art-26__as-amended-L33-2008"]


def test_unamended_article_and_decree():
    assert lookup("Art. 20 of Law No. (26) of 2007") == ["dubai-law-26-2007__art-20"]
    assert lookup("Decree 43 of 2013 Article 2 scope") == ["dubai-decree-43-2013__art-2"]


def test_no_instrument_means_no_lookup():
    assert lookup("Art. 12 rent instalments") == []          # ambiguous: several instruments have an Art. 12
    assert lookup("my landlord wants 15% more rent") == []


def test_unknown_article_ignored():
    assert lookup("Article 99 of Law 26/2007") == []
