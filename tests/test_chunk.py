"""Chunker QA on the built corpus (run `python -m tenancy_rag build-corpus` first)."""
import json
from collections import defaultdict

import pytest

from tenancy_rag import config
from tenancy_rag.preprocess.chunk import MAX_SINGLE, OVERLAP

PARENTS = config.PROCESSED / "parents.jsonl"
FIXED = config.PROCESSED / "chunks_fixed512.jsonl"

pytestmark = pytest.mark.skipif(not config.CHUNKS.exists(), reason="corpus not built")


def _load(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


@pytest.fixture(scope="module")
def parents():
    return {p["parent_id"]: p for p in _load(PARENTS)}


@pytest.fixture(scope="module")
def chunks_by_parent():
    out = defaultdict(list)
    for c in _load(config.CHUNKS):
        out[c["parent_id"]].append(c)
    return out


def test_every_parent_has_chunks(parents, chunks_by_parent):
    assert set(parents) == set(chunks_by_parent)


def test_chunk_ids_unique(chunks_by_parent):
    ids = [c["chunk_id"] for cs in chunks_by_parent.values() for c in cs]
    assert len(ids) == len(set(ids))


def test_split_chunks_cut_only_at_unit_boundaries(parents, chunks_by_parent):
    """Every line of a split chunk is a whole line (or whole sentence) of its parent."""
    for pid, cs in chunks_by_parent.items():
        if len(cs) == 1:
            assert cs[0]["text"] == parents[pid]["text"]
            continue
        for c in cs:
            for line in c["text"].split("\n"):
                assert line in parents[pid]["text"], (pid, line[:60])


def test_split_chunks_cover_parent_and_overlap(parents, chunks_by_parent):
    for pid, cs in chunks_by_parent.items():
        if len(cs) == 1:
            continue
        lines = {l.strip() for l in parents[pid]["text"].split("\n") if l.strip()}
        covered = {l for c in cs for l in c["text"].split("\n")}
        assert all(any(l in cov or cov in l for cov in covered) for l in lines), pid
        for a, b in zip(cs, cs[1:]):  # consecutive windows share content (overlap)
            assert set(a["text"].split("\n")) & set(b["text"].split("\n")), (a["chunk_id"], b["chunk_id"])


def test_only_long_parents_are_split(parents, chunks_by_parent):
    from tenancy_rag.preprocess.chunk import n_tokens
    for pid, cs in chunks_by_parent.items():
        if len(cs) > 1:
            assert n_tokens(parents[pid]["text"]) > MAX_SINGLE


def test_prefix_names_instrument_and_locator(parents, chunks_by_parent):
    for pid, cs in chunks_by_parent.items():
        p = parents[pid]
        assert cs[0]["prefix"].startswith(p["instrument_short"])
        assert p["locator"] in cs[0]["prefix"]


def test_superseded_status_propagates(parents, chunks_by_parent):
    for pid, cs in chunks_by_parent.items():
        assert all(c["status"] == parents[pid]["status"] for c in cs)


def test_fixed_windows_cover_every_parent(parents):
    covered = {pid for w in _load(FIXED) for pid in w["parent_ids"]}
    assert covered == set(parents)


def test_no_duplicate_faq_answers(parents):
    import re
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    answers = [norm(p["text"].split("\nA:", 1)[-1]) for p in parents.values()
               if p["instrument_short"] in ("DLD-FAQ", "RDC-FAQ")]
    assert len(answers) == len(set(answers))
