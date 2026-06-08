"""Diagnostic tests: the structural-ceiling logic that is eval-sanity's reason
to exist. Exact assertions, since the math is deterministic."""

import pytest

from eval_sanity import RetrievalSample, oracle_ceiling, sanity_report


# --- oracle_ceiling --------------------------------------------------------


def test_oracle_ceiling_single_answer_is_one():
    # Every query has exactly 1 relevant doc; min(k, 1)/1 == 1.0 for any k>=1.
    oc = oracle_ceiling([{"a"}, {"b"}, {"c"}], k=5)
    assert oc.mean_ceiling == 1.0
    assert oc.capped_queries == 0
    assert oc.capped_fraction == 0.0


def test_oracle_ceiling_caps_when_n_rel_exceeds_k():
    # 4 relevant docs, k=2 -> ceiling 2/4 = 0.5; flagged as capped.
    oc = oracle_ceiling([{"a", "b", "c", "d"}], k=2)
    assert oc.mean_ceiling == 0.5
    assert oc.capped_queries == 1
    assert oc.n_queries == 1


def test_oracle_ceiling_mixed():
    # one capped (3 rel, k=2 -> 2/3), one fine (1 rel -> 1.0)
    oc = oracle_ceiling([{"a", "b", "c"}, {"z"}], k=2)
    assert oc.mean_ceiling == pytest.approx((2 / 3 + 1.0) / 2)
    assert oc.capped_queries == 1


def test_oracle_ceiling_skips_empty_relevant():
    oc = oracle_ceiling([set(), {"a"}], k=3)
    assert oc.n_queries == 1
    assert oc.mean_ceiling == 1.0


def test_oracle_ceiling_rejects_bad_k():
    with pytest.raises(ValueError):
        oracle_ceiling([{"a"}], k=0)


# --- sanity_report ---------------------------------------------------------


def _multi_answer_dataset():
    # 3 queries, each with 4 relevant docs, retriever finds exactly 1 of them in
    # the top-2. Proportion recall is mechanically stuck at 1/4; hit@2 is 1.0.
    return [
        RetrievalSample(
            query=f"q{i}",
            retrieved_doc_ids=[f"q{i}_rel0", f"q{i}_noise"],
            relevant_doc_ids={f"q{i}_rel{j}" for j in range(4)},
        )
        for i in range(3)
    ]


def test_sanity_report_surfaces_divergence():
    rep = sanity_report(_multi_answer_dataset(), k=2, threshold=0.8)
    assert rep.n_queries == 3
    assert rep.mean_proportion_recall == pytest.approx(0.25)
    assert rep.mean_hit_at_k == 1.0
    assert rep.divergence == pytest.approx(0.75)


def test_sanity_report_flags_unreachable_threshold():
    # 4 relevant docs, k=2 -> ceiling 0.5 < threshold 0.8 -> unreachable.
    rep = sanity_report(_multi_answer_dataset(), k=2, threshold=0.8)
    assert rep.unreachable_queries == 3
    assert rep.unreachable_fraction == 1.0
    assert "UNREACHABLE" in rep.headline


def test_sanity_report_threshold_reachable_when_k_large():
    # k=4 holds all 4 relevant docs -> ceiling 1.0 -> threshold reachable.
    rep = sanity_report(_multi_answer_dataset(), k=4, threshold=0.8)
    assert rep.unreachable_queries == 0
    assert "reachable for every query" in rep.headline


def test_sanity_report_single_answer_no_artifact():
    samples = [
        RetrievalSample(query="q", retrieved_doc_ids=["a", "x"], relevant_doc_ids={"a"})
    ]
    rep = sanity_report(samples, k=2, threshold=0.8)
    # hit and proportion recall agree; nothing unreachable.
    assert rep.mean_proportion_recall == 1.0
    assert rep.mean_hit_at_k == 1.0
    assert rep.divergence == 0.0
    assert rep.unreachable_queries == 0


def test_sanity_report_skips_queries_without_relevant_docs():
    samples = [
        RetrievalSample(query="empty", retrieved_doc_ids=["a"], relevant_doc_ids=set()),
        RetrievalSample(query="real", retrieved_doc_ids=["a"], relevant_doc_ids={"a"}),
    ]
    rep = sanity_report(samples, k=1)
    assert rep.n_queries == 1


def test_sanity_report_empty_dataset():
    rep = sanity_report([], k=2)
    assert rep.n_queries == 0
    assert rep.unreachable_queries == 0


def test_sanity_report_validates_threshold():
    with pytest.raises(ValueError):
        sanity_report(_multi_answer_dataset(), k=2, threshold=1.5)


def test_report_format_is_printable():
    rep = sanity_report(_multi_answer_dataset(), k=2, threshold=0.8)
    text = rep.format()
    assert "eval-sanity report" in text
    assert str(rep) == text
