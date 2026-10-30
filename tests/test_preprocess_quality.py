"""Pre-processing quality checks on the built corpus."""
import json
import re

import pytest
from wordfreq import zipf_frequency

from tenancy_rag import config

PARENTS = config.PROCESSED / "parents.jsonl"
pytestmark = pytest.mark.skipif(not PARENTS.exists(), reason="corpus not built")


@pytest.fixture(scope="module")
def parents():
    return [json.loads(l) for l in PARENTS.open(encoding="utf-8")]


def test_no_residual_dropped_ligatures(parents):
    """No word in PDF-derived text is a rare token that becomes a common word by re-inserting
    the 'i'/'l' a ligature dropped (e.g. 'benefciary')."""
    residual = set()
    for p in parents:
        if p["instrument_short"] != "DLD-TG":
            continue
        for w in set(re.findall(r"[A-Za-z]+", p["text"])):
            if "f" not in w.lower() or zipf_frequency(w, "en") >= 2.0:
                continue
            cands = [w[:i + 1] + c + w[i + 1:] for i, ch in enumerate(w) if ch.lower() == "f" for c in "il"]
            if any(zipf_frequency(c, "en") >= 3.0 for c in cands):
                residual.add(w)
    assert not residual, sorted(residual)


def test_quality_flags_point_at_binding_text(parents):
    by_id = {p["parent_id"]: p for p in parents}
    flagged = [p for p in parents if p.get("quality_flags")]
    assert len(flagged) == len(json.loads((config.DATA / "known_conflicts.json").read_text(encoding="utf-8")))
    for p in flagged:
        assert p["authority_tier"] in ("regulatory_guidance", "service_guidance")
        for f in p["quality_flags"]:
            for target in f["conflicts_with"]:
                assert by_id[target]["authority_tier"] in ("legislation", "decree")
                assert by_id[target]["status"] == "in_force"


def test_no_running_headers_or_bare_page_numbers(parents):
    for p in parents:
        if p["instrument_short"] == "DLD-TG":
            first = p["text"].splitlines()[0]
            assert first not in ("Tenancy Guide", "How to Register", "Tenancy Regulating Legislations")
            assert not re.search(r"^\s*\d{1,3}\s*$", p["text"], re.M)
