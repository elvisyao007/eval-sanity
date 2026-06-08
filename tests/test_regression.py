"""Regression-detection tests: one per decision branch.

Scenarios are built with large, deliberate statistical separation so a reduced
bootstrap (resamples=2000) is more than enough to resolve them — the point is to
exercise each quadrant and the alarm logic, not to stress the CI precision.
"""

import json

import pytest

from eval_sanity import RetrievalSample, detect_regression, from_eval_json
from eval_sanity.regression import (
    GENERATION_ONLY,
    SILENT_REGRESSION,
    STABLE,
    VISIBLE_REGRESSION,
)

K = 5
RES = 2000  # fast; scenarios are separated far beyond CI resolution


def _hit_sample(qid: str, found: bool) -> RetrievalSample:
    """One query with a single relevant doc; `found` puts it at rank 1 (recall
    and hit = 1.0) or leaves it out entirely (both 0.0)."""
    relevant = {f"{qid}_rel"}
    retrieved = [f"{qid}_rel", f"{qid}_x"] if found else [f"{qid}_x", f"{qid}_y"]
    return RetrievalSample(query=qid, retrieved_doc_ids=retrieved, relevant_doc_ids=relevant)


def _runs(n, baseline_found, current_found):
    """Build baseline/current sample lists from per-query found-flags."""
    base = [_hit_sample(f"q{i}", baseline_found(i)) for i in range(n)]
    cur = [_hit_sample(f"q{i}", current_found(i)) for i in range(n)]
    return base, cur


def _gen(n, score):
    return {f"q{i}": (score(i) if callable(score) else score) for i in range(n)}


# --- quadrant 1: SILENT REGRESSION (alarm) ---------------------------------


def test_silent_regression_alarms():
    n = 40
    # baseline finds every doc; current loses half. Generation score identical.
    base, cur = _runs(n, lambda i: True, lambda i: i % 2 == 1)
    gen_b = _gen(n, 0.9)
    gen_c = _gen(n, 0.9)  # unchanged
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.quadrant == SILENT_REGRESSION
    assert rep.alarm is True
    assert rep.recall.direction == "down"
    assert rep.gen.direction == "flat"
    assert "SILENT REGRESSION" in rep.verdict
    assert "dashboard" in rep.verdict
    # ~20 queries lost recall, worst-first
    assert len(rep.danger_list) == 20
    assert rep.danger_list[0].recall_delta == -1.0


# --- quadrant 2: VISIBLE REGRESSION (retrieval down AND gen down) -----------


def test_visible_regression_no_alarm():
    n = 40
    base, cur = _runs(n, lambda i: True, lambda i: i % 2 == 1)
    gen_b = _gen(n, 0.9)
    gen_c = _gen(n, 0.6)  # generation also drops, significantly
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.quadrant == VISIBLE_REGRESSION
    assert rep.alarm is False
    assert rep.recall.direction == "down"
    assert rep.gen.direction == "down"


# --- quadrant 3: GENERATION ONLY (retrieval flat, gen moves) ---------------


def test_generation_only_no_alarm():
    n = 40
    base, cur = _runs(n, lambda i: True, lambda i: True)  # retrieval identical
    gen_b = _gen(n, 0.9)
    gen_c = _gen(n, 0.6)
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.quadrant == GENERATION_ONLY
    assert rep.alarm is False
    assert rep.recall.direction == "flat"
    assert rep.gen.direction == "down"


# --- quadrant 4: STABLE (nothing moves) ------------------------------------


def test_stable_no_alarm():
    n = 40
    base, cur = _runs(n, lambda i: True, lambda i: True)
    gen_b = _gen(n, 0.9)
    gen_c = _gen(n, 0.9)
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.quadrant == STABLE
    assert rep.alarm is False
    assert rep.recall.direction == "flat"
    assert rep.gen.direction == "flat"


# --- noise must NOT alarm ---------------------------------------------------


def test_small_retrieval_noise_does_not_alarm():
    n = 40
    # only 2 of 40 queries flip -> mean delta -0.05, CI spans 0 -> not significant
    base, cur = _runs(n, lambda i: True, lambda i: i not in (3, 17))
    gen_b = _gen(n, 0.9)
    gen_c = _gen(n, 0.9)
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.recall.direction == "flat"  # within noise
    assert rep.alarm is False
    assert rep.quadrant == STABLE


# --- query alignment --------------------------------------------------------


def test_unaligned_queries_reported_and_excluded():
    base = [_hit_sample(f"q{i}", True) for i in range(5)]      # q0..q4
    cur = [_hit_sample(f"q{i}", True) for i in range(2, 7)]    # q2..q6
    gen_b = {f"q{i}": 0.9 for i in range(5)}
    gen_c = {f"q{i}": 0.9 for i in range(2, 7)}
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.only_in_baseline == ["q0", "q1"]
    assert rep.only_in_current == ["q5", "q6"]
    assert rep.n_aligned == 3  # q2, q3, q4


def test_missing_gen_score_excludes_query():
    base = [_hit_sample(f"q{i}", True) for i in range(3)]
    cur = [_hit_sample(f"q{i}", True) for i in range(3)]
    gen_b = {"q0": 0.9, "q1": 0.9, "q2": 0.9}
    gen_c = {"q0": 0.9, "q1": 0.9}  # q2 has no current gen score
    rep = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES)

    assert rep.missing_gen == ["q2"]
    assert rep.n_aligned == 2


# --- reproducibility --------------------------------------------------------


def test_bootstrap_is_reproducible():
    n = 40
    base, cur = _runs(n, lambda i: True, lambda i: i % 2 == 1)
    gen_b, gen_c = _gen(n, 0.9), _gen(n, 0.9)
    a = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES, seed=42)
    b = detect_regression(base, cur, gen_b, gen_c, K, resamples=RES, seed=42)
    assert (a.recall.ci_low, a.recall.ci_high) == (b.recall.ci_low, b.recall.ci_high)
    assert (a.gen.ci_low, a.gen.ci_high) == (b.gen.ci_low, b.gen.ci_high)


# --- validation -------------------------------------------------------------


def test_rejects_bad_k():
    with pytest.raises(ValueError):
        detect_regression([], [], {}, {}, 0)


def test_rejects_bad_ci():
    with pytest.raises(ValueError):
        detect_regression([], [], {}, {}, K, ci=1.5)


# --- from_eval_json shell ---------------------------------------------------


def _write(path, found_flags, gen):
    per_query = {}
    for i, found in enumerate(found_flags):
        qid = f"q{i}"
        per_query[qid] = {
            "retrieved_doc_ids": [f"{qid}_rel", f"{qid}_x"]
            if found
            else [f"{qid}_x", f"{qid}_y"],
            "relevant_doc_ids": [f"{qid}_rel"],
            "faithfulness": gen,
        }
    path.write_text(json.dumps({"per_query": per_query}), encoding="utf-8")


def test_from_eval_json_silent_regression(tmp_path):
    n = 40
    bp = tmp_path / "baseline.json"
    cp = tmp_path / "current.json"
    _write(bp, [True] * n, 0.9)
    _write(cp, [i % 2 == 1 for i in range(n)], 0.9)

    rep = from_eval_json(bp, cp, K, resamples=RES)
    assert rep.quadrant == SILENT_REGRESSION
    assert rep.alarm is True
    assert rep.gen_label == "faithfulness"


def test_from_eval_json_requires_per_query(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nope": {}}), encoding="utf-8")
    good = tmp_path / "good.json"
    _write(good, [True], 0.9)
    with pytest.raises(ValueError):
        from_eval_json(bad, good, K, resamples=RES)
