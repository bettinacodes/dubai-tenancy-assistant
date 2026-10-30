"""Validator tests. The NLI cases are a regression set (true and false
claims against real articles), including the number-binding case plain NLI gets wrong."""
import json

import pytest

from tenancy_rag import config
from tenancy_rag.generate.citation_validator import claim_numbers, premise_windows, validate

PARENTS = config.PROCESSED / "parents.jsonl"
pytestmark = pytest.mark.skipif(not PARENTS.exists(), reason="corpus not built")


@pytest.fixture(scope="module")
def P():
    return {json.loads(l)["parent_id"]: json.loads(l) for l in PARENTS.open(encoding="utf-8")}


def test_windows_follow_legal_structure(P):
    w = premise_windows(P["dubai-decree-43-2013__art-1"]["text"])
    assert len(w) == 5 and all(x.startswith("When renewing") for x in w)      # lead-in + each clause a-e
    a25 = premise_windows(P["dubai-law-26-2007__art-25__as-amended-L33-2008"]["text"])
    sale = next(x for x in a25 if "wishes to sell" in x)
    assert "twelve (12) months" in sale                                       # closing 'For the purposes' line attached


def test_claim_numbers_reads_digits_and_words():
    assert claim_numbers("twelve months, AED 20,000 and 3.5%") == {"12", "20000", "3.5"}


NLI_CASES = [
    ("A landlord who wants to sell the property must give the tenant at least 12 months' notice before eviction.", "dubai-law-26-2007__art-25__as-amended-L33-2008", True),
    ("A landlord who wants to sell can evict the tenant with 3 months' notice.", "dubai-law-26-2007__art-25__as-amended-L33-2008", False),
    ("If the rent is 11% to 20% below the average, the maximum increase is 5%.", "dubai-decree-43-2013__art-1", True),
    ("If the rent is 11% to 20% below the average, the maximum increase is 10%.", "dubai-decree-43-2013__art-1", False),
    ("If the rent is 31% to 40% below the average, the maximum increase is 20%.", "dubai-decree-43-2013__art-1", False),
    ("The law requires the landlord to take a security deposit.", "dubai-law-26-2007__art-20", False),
    ("The RDC filing fee is 5% of the annual rent.", "rdc-faq__q-14", False),
]


@pytest.mark.parametrize("claim,pid,truth", NLI_CASES)
def test_entailment_regression(P, claim, pid, truth):
    from tenancy_rag.generate.citation_validator import entailment
    assert (entailment(claim, [P[pid]["text"]]) >= 0.5) is truth


def _src(P, label, pid):
    p = P[pid]
    return {"label": label, "parent_id": pid, "text": p["text"], "status": p["status"],
            "flagged": bool(p.get("quality_flags")), "authority_tier": p["authority_tier"]}


def test_structural_checks_without_nli(P):
    srcs = [_src(P, "S1", "dubai-law-26-2007__art-20"), _src(P, "S2", "dubai-law-26-2007__art-25"),
            _src(P, "S3", "rdc-faq__q-85")]
    raw = json.dumps({"answer": "The deposit is refundable within 14 days.",
                      "claims": [{"text": "x", "sources": ["S9"]},           # invented label
                                 {"text": "y", "sources": ["S2"]},           # superseded
                                 {"text": "z", "sources": ["S3"]},           # flagged-only
                                 {"text": "w", "sources": []}]})              # no citation
    v = validate(raw, srcs, "When do I get my deposit back?", check_entailment=False)
    reasons = [r for c in v.claims for r in c["reasons"]]
    assert not v.ok
    assert any("unknown source" in r for r in reasons) and any("superseded" in r for r in reasons)
    assert any("misstate" in r for r in reasons) and any("no citation" in r for r in reasons)
    assert any("numbers not found" in p for p in v.problems)                  # '14' is invented


def test_honest_insufficient_answer_passes(P):
    v = validate(json.dumps({"answer": "The sources do not cover this.", "claims": [], "insufficient": True}),
                 [_src(P, "S1", "dubai-law-26-2007__art-20")], "q", check_entailment=False)
    assert v.ok


def test_invalid_json_fails(P):
    assert not validate("not json", [], "q").ok
