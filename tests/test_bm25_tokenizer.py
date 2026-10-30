from tenancy_rag.index.bm25_index import legal_tokenize


def test_instrument_references_normalised():
    assert "26/2007" in legal_tokenize("Law No. (26) of 2007")
    assert "26/2007" in legal_tokenize("Law 26/2007")
    assert "43/2013" in legal_tokenize("Decree No. 43 of 2013")


def test_percentages_money_and_clauses_kept_whole():
    toks = legal_tokenize("an increase of 10% under Article 25(1)(a), capped at AED 20,000")
    assert "10%" in toks and "20000" in toks and "aed" in toks
    assert "25(1)(a)" in toks and "25" in toks


def test_negations_and_legal_modals_survive_stopwording():
    toks = legal_tokenize("The landlord may not evict and must not disconnect services")
    assert "not" in toks and "may" in toks and "must" in toks
    assert "the" not in toks and "and" not in toks


def test_stemming_matches_inflections():
    assert legal_tokenize("eviction")[0] == legal_tokenize("evictions")[0]
    assert legal_tokenize("evicting", stem=False) == ["evicting"]
