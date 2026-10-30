import math

import pytest

from tenancy_rag.eval.retrieval_metrics import aggregate, parents_from_chunks, query_metrics


def test_parents_from_chunks_dedups_and_expands_windows():
    chunks = [{"parent_id": "a"}, {"parent_id": "a"}, {"parent_id": "b", "parent_ids": ["b", "c"]}, {"parent_id": "d"}]
    assert parents_from_chunks(chunks) == ["a", "b", "c", "d"]


def test_hand_computed_example():
    ranked = ["x", "p", "y", "r", "z"]
    m = query_metrics(ranked, relevant={"p", "r"}, primary={"p"})
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert m["primary@1"] == 0 and m["primary@3"] == 1
    assert m["mrr@10"] == 0.5
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg@10"] == pytest.approx(dcg / idcg)


def test_perfect_and_empty_rankings():
    perfect = query_metrics(["p", "r"], {"p", "r"}, {"p"})
    assert perfect["mrr@10"] == 1 and perfect["ndcg@10"] == pytest.approx(1.0)
    miss = query_metrics(["x", "y"], {"p"}, set())
    assert miss["hit@10"] == 0 and miss["mrr@10"] == 0 and "primary@1" not in miss


def test_aggregate_skips_missing_primary():
    agg = aggregate([{"primary@1": 1.0, "mrr@10": 1.0}, {"mrr@10": 0.0}])
    assert agg["primary@1"] == 1.0 and agg["mrr@10"] == 0.5 and agg["n"] == 2
