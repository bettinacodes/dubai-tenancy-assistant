"""Personal data must not reach the generator or retrieval: both write to disk caches."""
from tenancy_rag import pipeline
from tenancy_rag.pipeline import Assistant


def test_generator_and_retrieval_only_see_redacted_text(monkeypatch):
    seen = {}

    def fake_sources(self, query, tool, historical, alt_query=None):
        seen["retrieval"] = query
        return []

    def fake_generate(question, sources, **kw):
        seen["generator"] = question
        return {"mode": "insufficient", "answer": "", "cited_labels": [], "caveats": [], "attempts": []}

    monkeypatch.setattr(Assistant, "_sources", fake_sources)
    monkeypatch.setattr(pipeline, "generate_answer", fake_generate)
    a = object.__new__(Assistant)          # skip loading the retriever
    a.backend, a.model, a.check_entailment, a.n_sources = "ollama", None, False, 6
    r = a.answer("My name is Omar Haddad, call me on 050 123 4567. Can my landlord raise my AED 80,000 rent by 10%?")
    for where in ("retrieval", "generator"):
        assert "Omar" not in seen[where] and "4567" not in seen[where], where
        assert "80,000" in seen[where] and "10%" in seen[where], where   # the legal content survives
    assert "Omar" not in r["query_logged"]
