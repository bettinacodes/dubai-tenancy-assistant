import pytest

from tenancy_rag import config
from tenancy_rag.retrieve.fusion import rrf


def test_rrf_rewards_agreement_and_is_rank_based():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "w"}]
    fused = [c["chunk_id"] for c in rrf([a, b], k=60)]
    assert fused[0] == "y"                          # ranked by both lists
    assert set(fused) == {"x", "y", "z", "w"}
    assert rrf([a], k=60)[0]["score"] == pytest.approx(1 / 61)


@pytest.mark.skipif(not (config.PROCESSED / "parents.jsonl").exists(), reason="corpus not built")
def test_superseded_split_and_flag_demotion():
    from tenancy_rag.retrieve.filters import demote_flagged, split_superseded
    cands = [{"chunk_id": "1", "parent_id": "dubai-law-26-2007__art-25"},                       # superseded
             {"chunk_id": "2", "parent_id": "dld-faq__q-82"},                                   # flagged guidance
             {"chunk_id": "3", "parent_id": "dubai-law-26-2007__art-25__as-amended-L33-2008"}]  # current
    current, historical = split_superseded(cands)
    assert [c["chunk_id"] for c in historical] == ["1"]
    assert [c["chunk_id"] for c in demote_flagged(current)] == ["3", "2"]


def test_weighted_rrf_downweights_second_list():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "x"}]
    assert rrf([a, b], weights=[1.0, 0.2])[0]["chunk_id"] == "x"
    assert rrf([a, b], weights=[0.2, 1.0])[0]["chunk_id"] == "y"
