"""The hybrid intent policy may only ADD protective routes, and only at high confidence."""
from tenancy_rag.understand import intent_hybrid as H


def _with(monkeypatch, label, p):
    monkeypatch.setattr(H, "_proba_fn", lambda: (lambda text: (label, p)))


def test_rules_protective_route_is_never_overridden(monkeypatch):
    _with(monkeypatch, "factual", 0.99)
    assert H.make_classifier(0.95)("they changed the locks and my kids are outside") == "crisis"


def test_classifier_adds_protective_route_only_when_confident(monkeypatch):
    q = "What is the deal with my landlord?"          # rules: factual
    _with(monkeypatch, "advice_seeking", 0.97)
    assert H.make_classifier(0.95)(q) == "advice_seeking"
    _with(monkeypatch, "advice_seeking", 0.90)
    assert H.make_classifier(0.95)(q) == "factual"


def test_classifier_cannot_change_non_protective_labels(monkeypatch):
    _with(monkeypatch, "procedural", 0.99)
    assert H.make_classifier(0.95)("What is the deal with my landlord?") == "factual"
