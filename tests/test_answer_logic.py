"""Generation control flow with a stubbed LLM (no model calls): retry, guards and fallbacks."""
import json

import pytest

from tenancy_rag import config
from tenancy_rag.generate import answer as A
from tenancy_rag.llm import LLMResult

PARENTS = config.PROCESSED / "parents.jsonl"
pytestmark = pytest.mark.skipif(not PARENTS.exists(), reason="corpus not built")


@pytest.fixture(scope="module")
def P():
    return {json.loads(l)["parent_id"]: json.loads(l) for l in PARENTS.open(encoding="utf-8")}


def src(P, label, pid):
    p = P[pid]
    return {"label": label, "parent_id": pid, "instrument": p["instrument"], "locator": p["locator"], "heading": p.get("article_heading"),
            "text": p["text"], "status": p["status"], "authority_tier": p["authority_tier"], "flagged": bool(p.get("quality_flags")),
            "url": p["source_url"], "superseded_by": p.get("superseded_by", [])}


def stub(monkeypatch, outputs):
    calls = []

    def fake_chat(msgs, **kw):
        calls.append(msgs)
        return LLMResult(text=json.dumps(outputs[min(len(calls) - 1, len(outputs) - 1)]), model="stub", backend="stub", latency_s=0.0)
    monkeypatch.setattr(A, "chat", fake_chat)
    return calls


GOOD = {"answer": "The landlord may take a security deposit and must refund it when the lease ends.",
        "claims": [{"text": "The landlord must refund the security deposit when the lease ends.", "sources": ["S1"]}], "caveats": []}
BAD = {"answer": "The landlord must take a deposit.",
       "claims": [{"text": "The law requires the landlord to take a security deposit.", "sources": ["S1"]}], "caveats": []}


def test_valid_first_answer_is_returned(P, monkeypatch):
    calls = stub(monkeypatch, [GOOD])
    r = A.generate_answer("Deposit?", [src(P, "S1", "dubai-law-26-2007__art-20")])
    assert r["mode"] == "generated" and r["cited_labels"] == ["S1"] and len(calls) == 1


def test_retry_gets_feedback_then_succeeds(P, monkeypatch):
    calls = stub(monkeypatch, [BAD, GOOD])
    r = A.generate_answer("Deposit?", [src(P, "S1", "dubai-law-26-2007__art-20")])
    assert r["mode"] == "generated_after_retry"
    assert "failed verification" in calls[1][0]["content"]          # validator problems fed back


def test_extractive_fallback_quotes_the_law(P, monkeypatch):
    stub(monkeypatch, [BAD, BAD])
    r = A.generate_answer("Is a security deposit required?", [src(P, "S1", "dubai-law-26-2007__art-20")])
    assert r["mode"] == "extractive_fallback" and "may obtain from the Tenant a security deposit" in r["answer"]


def test_extractive_fallback_prefers_the_cited_binding_source(P, monkeypatch):
    # S1 overlaps the question better lexically, but the model cited S2: quote S2 (the citation is usually right)
    bad_s2 = {**BAD, "claims": [{"text": "The law requires the landlord to take a security deposit.", "sources": ["S2"]}]}
    stub(monkeypatch, [bad_s2, bad_s2])
    r = A.generate_answer("Is a security deposit required by the landlord?",
                          [src(P, "S1", "dubai-law-26-2007__art-20"), src(P, "S2", "dubai-law-26-2007__art-21")])
    assert r["mode"] == "extractive_fallback" and r["cited_labels"] == ["S2"]


def test_advice_guard_blocks_directives_and_uses_template(P, monkeypatch):
    directive = {**GOOD, "answer": "You should file a case at the RDC; you will win."}
    stub(monkeypatch, [directive, directive])
    r = A.generate_answer("Should I sue?", [src(P, "S1", "dubai-law-26-2007__art-20")], advice=True)
    assert r["mode"] == "advice_template" and "can't tell you what to do" in r["answer"]


def test_historical_fallback_diffs_2007_and_2008(P, monkeypatch):
    stub(monkeypatch, [BAD, BAD])
    srcs = [src(P, "S1", "dubai-law-26-2007__art-25__as-amended-L33-2008"), src(P, "S2", "dubai-law-26-2007__art-25")]
    r = A.generate_answer("What changed?", srcs, historical=True)
    assert r["mode"] == "historical_diff" and "2007:" in r["answer"] and "now:" in r["answer"]


def test_faq_fallback_never_quotes_a_bare_question(P, monkeypatch):
    stub(monkeypatch, [BAD, BAD])
    r = A.generate_answer("deposit back?", [src(P, "S1", "rdc-faq__q-50")])
    assert "Payment Writ" in r["answer"] and '"Q:' not in r["answer"]


def test_context_guard_drops_lowest_ranked(P):
    big = [src(P, f"S{i}", "dubai-law-26-2007__art-25__as-amended-L33-2008") for i in range(1, 9)]
    kept, dropped = A.fit_context(big, fixed_chars=3000)
    assert len(kept) < 8 and len(dropped) == 8 - len(kept) and kept[0]["label"] == "S1"
