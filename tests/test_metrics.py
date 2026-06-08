"""Deterministic-metric tests. Assertions are exact because the metrics have no
randomness and no model — ported from the eval-driven-llm retrieval smoke tests.
"""

import pytest

from eval_sanity import metrics as m


def test_perfect_ranking():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert m.precision_at_1(retrieved, relevant) == 1.0
    assert m.recall_at_k(retrieved, relevant, 2) == 1.0
    assert m.reciprocal_rank(retrieved, relevant) == 1.0


def test_first_relevant_at_rank_2():
    retrieved = ["x", "a", "b"]
    relevant = {"a"}
    assert m.precision_at_1(retrieved, relevant) == 0.0
    assert m.reciprocal_rank(retrieved, relevant) == 0.5


def test_recall_partial():
    retrieved = ["a", "x", "y"]
    relevant = {"a", "b"}
    assert m.recall_at_k(retrieved, relevant, 3) == 0.5


def test_ndcg_monotonic():
    relevant = {"a"}
    top = m.ndcg_at_k(["a", "x"], relevant, 2)
    low = m.ndcg_at_k(["x", "a"], relevant, 2)
    assert top > low


def test_aggregate_runs():
    runs = [(["a", "b"], {"a"}), (["x", "a"], {"a"})]
    out = m.aggregate(runs, ks=(2,))
    assert out["p@1"] == 0.5
    assert out["mrr"] == 0.75  # (1.0 + 0.5) / 2


def test_aggregate_reports_both_recall_and_hit():
    out = m.aggregate([(["a", "x"], {"a"})], ks=(2,))
    assert "recall@2" in out and "hit@2" in out


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        m.recall_at_k(["a", "a"], {"a"}, 2)


def test_relevant_must_be_set():
    with pytest.raises(TypeError):
        m.recall_at_k(["a"], ["a"], 1)  # type: ignore[arg-type]


# --- hit@k: the honest, ceiling-free signal --------------------------------


def test_hit_at_k_one_relevant_in_topk():
    # 3 relevant docs but only one in the top-2: proportion recall is capped at
    # 1/3, while hit@k is a clean 1.0.
    retrieved = ["a", "x"]
    relevant = {"a", "b", "c"}
    assert m.hit_at_k(retrieved, relevant, 2) == 1.0
    assert m.recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)


def test_hit_at_k_miss():
    assert m.hit_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_hit_at_k_empty_relevant_is_zero():
    assert m.hit_at_k(["a"], set(), 2) == 0.0


# --- context recall / grounded-but-wrong (judge-free halves) ---------------


def test_context_recall_docs_deterministic():
    assert m.context_recall_docs(["A", "X"], {"A", "B"}) == 0.5
    assert m.context_recall_docs(["A", "B"], {"A", "B"}) == 1.0


def test_context_recall_docs_no_relevant_is_zero():
    assert m.context_recall_docs(["A"], set()) == 0.0


def test_grounded_but_wrong_flag():
    assert m.grounded_but_wrong_flag(0.95, 0.2) is True   # faithful, bad retrieval
    assert m.grounded_but_wrong_flag(0.95, 0.9) is False  # faithful, good retrieval
    assert m.grounded_but_wrong_flag(0.3, 0.2) is False   # not faithful
