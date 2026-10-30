"""QA for the hand-built evaluation sets: every gold answer must be checkable against the corpus.

* every cited parent id exists in the built corpus
* every quote is a verbatim (whitespace-normalised) substring of the cited parent's text
* primary (must-cite) ids are in force and are a subset of relevant ids
* forbidden citations in the adversarial set really are superseded
* calculator expectations agree with an independent re-computation from the verified constants
"""
import json
import re
from collections import Counter
from datetime import date

import pytest

from tenancy_rag import config
from tenancy_rag.tools import constants as K

PARENTS = config.PROCESSED / "parents.jsonl"
pytestmark = pytest.mark.skipif(not PARENTS.exists(), reason="corpus not built")

LABELS = {"answerable", "partially_answerable"}
BEHAVIOURS = {"abstain_out_of_scope", "answer_current_law_only", "explain_change", "answer_with_change_note",
              "correct_false_premise", "inform_no_advice", "resist_injection", "crisis_escalation",
              "decline_drafting", "answer"}


def _load(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


ws = lambda t: re.sub(r"\s+", " ", t).strip()


@pytest.fixture(scope="module")
def parents():
    return {p["parent_id"]: p for p in _load(PARENTS)}


@pytest.fixture(scope="module")
def gold():
    return _load(config.EVAL / "gold.jsonl")


@pytest.fixture(scope="module")
def adversarial():
    return _load(config.EVAL / "adversarial.jsonl")


def test_gold_shape(gold):
    assert len(gold) == 50
    assert len({g["id"] for g in gold}) == 50
    assert Counter(g["split"] for g in gold) == {"test": 35, "dev": 15}
    assert all(g["label"] in LABELS for g in gold)


def test_gold_ids_exist_and_primary_in_force(gold, parents):
    for g in gold:
        assert set(g["primary_ids"]) <= set(g["relevant_ids"]), g["id"]
        for pid in g["relevant_ids"]:
            assert pid in parents, (g["id"], pid)
        for pid in g["primary_ids"]:
            assert parents[pid]["status"] == "in_force", (g["id"], pid)


def test_gold_quotes_are_verbatim(gold, parents):
    for g in gold:
        assert g["quotes"], g["id"]
        for q in g["quotes"]:
            assert q["parent_id"] in g["relevant_ids"], (g["id"], q["parent_id"])
            assert ws(q["text"]) in ws(parents[q["parent_id"]]["text"]), (g["id"], q["text"][:60])


def test_adversarial_shape(adversarial, parents):
    assert len(adversarial) == 20
    for a in adversarial:
        assert a["expected_behaviour"] in BEHAVIOURS, a["id"]
        for pid in a["relevant_ids"]:
            assert pid in parents, (a["id"], pid)
        for pid in a.get("forbidden_citations", []):
            assert parents[pid]["status"] == "superseded", (a["id"], pid)


# --- independent re-computation of calculator expectations (the tools themselves come on Day 8) ---

def _band(current, avg):
    shortfall = (avg - current) / avg * 100
    for upper, pct in K.RENT_INCREASE_BANDS:
        if shortfall <= upper:
            return pct
    raise AssertionError


def test_tool_expectations_consistent(gold):
    for g in gold:
        t = g.get("tool")
        if not t:
            continue
        a, e = t["args"], t["expected"]
        if t["name"] == "rent_increase_cap":
            pct = _band(a["current_rent"], a["market_avg"])
            assert pct == e["max_increase_pct"], g["id"]
            assert round(a["current_rent"] * pct / 100) == e["max_increase_aed"], g["id"]
        elif t["name"] == "rdc_fee_estimate":
            fee = min(max(a["annual_rent"] * K.RDC_FEE_RATE, K.RDC_RENT_CASE_MIN), K.RDC_RENT_CASE_MAX)
            assert round(fee) == e["filing_fee"], g["id"]
            assert sum(K.RDC_FIXED_FEES.values()) == e["fixed_fees"], g["id"]
        elif t["name"] == "notice_period_check":
            days = (date.fromisoformat(a["expiry_date"]) - date.fromisoformat(a["notice_date"])).days
            if a["reason"] == "amend_terms":
                assert (days >= K.AMEND_TERMS_NOTICE_DAYS) == e["valid"], g["id"]
                assert days == e["days_given"]
            else:
                service_ok = a.get("served_via") in K.EVICTION_NOTICE_SERVICE
                assert service_ok == e["service_ok"], g["id"]


def test_no_flagged_record_is_gold_evidence(gold, parents):
    """Guidance that misstates the law (data/known_conflicts.json) must never be gold evidence."""
    flagged = {pid for pid, p in parents.items() if p.get("quality_flags")}
    for g in gold:
        assert not (set(g["primary_ids"]) & flagged), g["id"]
        for q in g["quotes"]:
            assert q["parent_id"] not in flagged, (g["id"], q["parent_id"])
