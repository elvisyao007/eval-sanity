"""Detect *silent* retrieval regressions between two eval runs.

v0.1 audits a single run's metrics. This module compares two runs (a baseline
and a current) and catches the failure mode dashboards hide: retrieval quality
drops by a statistically real amount while the generation-side score
(faithfulness, supplied by the caller) does not move. A faithfulness-only
dashboard shows green; the retriever has quietly degraded.

The judgement is never a bare threshold — every delta carries a 95% confidence
interval from a *paired bootstrap* (resample queries with replacement, keep the
baseline/current pairing intact, take the 2.5/97.5 percentiles of the mean
delta). A change counts only when its CI excludes 0. Bootstrap parameters follow
eval-driven-llm ADR-0007: seed 42, 10 000 resamples — re-runs give identical
bounds.

Boundary preserved from v0.1: eval-sanity never computes faithfulness or calls a
model. Generation scores are passed in by the caller as ``{query_id: float}``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import hit_at_k, recall_at_k
from .sample import RetrievalSample

# ADR-0007 pinned bootstrap parameters.
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_CI = 0.95


@dataclass(frozen=True)
class MetricDelta:
    """A single metric's change from baseline to current, with a bootstrap CI.

    ``delta`` is ``current_mean - baseline_mean`` (negative = a drop). The CI is
    the 95% paired-bootstrap interval on that mean delta. ``significant`` is true
    iff the interval excludes 0.
    """

    name: str
    baseline_mean: float
    current_mean: float
    delta: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        """True when the 95% CI excludes 0 (the change is not just noise)."""
        return self.ci_high < 0.0 or self.ci_low > 0.0

    @property
    def direction(self) -> str:
        """``"down"`` (sig. drop), ``"up"`` (sig. rise), or ``"flat"`` (noise)."""
        if self.ci_high < 0.0:
            return "down"
        if self.ci_low > 0.0:
            return "up"
        return "flat"

    def ci_str(self) -> str:
        return f"[CI {self.ci_low:+.3f},{self.ci_high:+.3f}]"

    def __str__(self) -> str:
        return f"{self.name} {self.delta:+.3f} {self.ci_str()}"


@dataclass(frozen=True)
class PerQueryDelta:
    """One aligned query's baseline/current values for retrieval and generation."""

    query_id: str
    recall_baseline: float
    recall_current: float
    hit_baseline: float
    hit_current: float
    gen_baseline: float
    gen_current: float

    @property
    def recall_delta(self) -> float:
        return self.recall_current - self.recall_baseline

    @property
    def hit_delta(self) -> float:
        return self.hit_current - self.hit_baseline

    @property
    def gen_delta(self) -> float:
        return self.gen_current - self.gen_baseline


# The four quadrants. Only SILENT_REGRESSION raises the alarm.
SILENT_REGRESSION = "silent_regression"
VISIBLE_REGRESSION = "visible_regression"  # retrieval down AND generation moved
GENERATION_ONLY = "generation_only"        # retrieval stable, generation moved
STABLE = "stable"                          # neither moved significantly


@dataclass(frozen=True)
class RegressionReport:
    """Verdict on whether a silent retrieval regression occurred between runs.

    Attributes:
        k: Retrieval cutoff the metrics were computed at.
        gen_label: Name of the generation-side score (e.g. ``"faithfulness"``).
        n_aligned: Queries present in both runs and both score maps — the only
            queries the statistics use.
        recall: :class:`MetricDelta` for proportion recall@k.
        hit: :class:`MetricDelta` for hit@k.
        gen: :class:`MetricDelta` for the generation score.
        quadrant: One of the four quadrant constants above.
        alarm: True only for ``SILENT_REGRESSION``.
        per_query: Every aligned query's deltas.
        danger_list: Queries whose recall dropped, worst (most negative recall
            delta) first.
        only_in_baseline / only_in_current: Query ids that failed to align.
        missing_gen: Aligned-by-retrieval ids dropped for lacking a generation
            score in one of the maps.
    """

    k: int
    gen_label: str
    n_aligned: int
    recall: MetricDelta
    hit: MetricDelta
    gen: MetricDelta
    quadrant: str
    alarm: bool
    per_query: list[PerQueryDelta] = field(default_factory=list)
    danger_list: list[PerQueryDelta] = field(default_factory=list)
    only_in_baseline: list[str] = field(default_factory=list)
    only_in_current: list[str] = field(default_factory=list)
    missing_gen: list[str] = field(default_factory=list)

    @property
    def quadrant_counts(self) -> dict[str, int]:
        """Per-query counts across the four sign-based quadrants of
        (retrieval delta, generation delta)."""
        counts = {
            "retrieval_down & gen_down": 0,
            "retrieval_down & gen_steady": 0,
            "retrieval_steady & gen_down": 0,
            "retrieval_steady & gen_steady": 0,
        }
        for q in self.per_query:
            r_down = q.recall_delta < 0.0
            g_down = q.gen_delta < 0.0
            if r_down and g_down:
                counts["retrieval_down & gen_down"] += 1
            elif r_down:
                counts["retrieval_down & gen_steady"] += 1
            elif g_down:
                counts["retrieval_steady & gen_down"] += 1
            else:
                counts["retrieval_steady & gen_steady"] += 1
        return counts

    @property
    def verdict(self) -> str:
        """One-sentence headline for the run."""
        r = self.recall
        g = self.gen
        if self.quadrant == SILENT_REGRESSION:
            return (
                f"SILENT REGRESSION detected: recall@{self.k} {r.delta:+.3f} "
                f"{r.ci_str()}, {self.gen_label} unchanged {g.ci_str()} "
                f"— your dashboard won't show this."
            )
        if self.quadrant == VISIBLE_REGRESSION:
            return (
                f"Visible regression: recall@{self.k} {r.delta:+.3f} {r.ci_str()} "
                f"AND {self.gen_label} {g.delta:+.3f} {g.ci_str()} both moved "
                f"significantly — a faithfulness dashboard would catch this one."
            )
        if self.quadrant == GENERATION_ONLY:
            return (
                f"Generation-only change: {self.gen_label} {g.delta:+.3f} "
                f"{g.ci_str()} moved while recall@{self.k} held steady "
                f"{r.ci_str()}. Retrieval is not the cause."
            )
        return (
            f"No significant change: recall@{self.k} {r.ci_str()} and "
            f"{self.gen_label} {g.ci_str()} are both within noise."
        )

    def format(self, top: int = 10) -> str:
        """Full human-readable report. ``top`` caps the danger list shown."""
        lines = [
            f"eval-sanity regression report  (k={self.k}, "
            f"n_aligned={self.n_aligned}, resamples={DEFAULT_RESAMPLES}, "
            f"seed={DEFAULT_SEED})",
            "=" * 70,
            f"  recall@{self.k} : {self.recall.baseline_mean:.3f} -> "
            f"{self.recall.current_mean:.3f}  ({self.recall.delta:+.3f}) "
            f"{self.recall.ci_str()}  [{self.recall.direction}]",
            f"  hit@{self.k}    : {self.hit.baseline_mean:.3f} -> "
            f"{self.hit.current_mean:.3f}  ({self.hit.delta:+.3f}) "
            f"{self.hit.ci_str()}  [{self.hit.direction}]",
            f"  {self.gen_label:<9}: {self.gen.baseline_mean:.3f} -> "
            f"{self.gen.current_mean:.3f}  ({self.gen.delta:+.3f}) "
            f"{self.gen.ci_str()}  [{self.gen.direction}]",
            "-" * 70,
            "  per-query quadrants (by sign of delta):",
        ]
        for label, n in self.quadrant_counts.items():
            lines.append(f"    {label:<32}: {n}")
        if self.only_in_baseline or self.only_in_current or self.missing_gen:
            lines.append("-" * 70)
            lines.append(
                f"  unaligned: {len(self.only_in_baseline)} only-in-baseline, "
                f"{len(self.only_in_current)} only-in-current, "
                f"{len(self.missing_gen)} missing generation score (excluded)"
            )
        if self.danger_list:
            lines.append("-" * 70)
            shown = self.danger_list[:top]
            lines.append(
                f"  danger list — {len(self.danger_list)} queries lost recall "
                f"(worst {len(shown)} shown):"
            )
            for q in shown:
                lines.append(
                    f"    {q.query_id:<20} recall {q.recall_delta:+.3f}  "
                    f"hit {q.hit_delta:+.3f}  {self.gen_label} {q.gen_delta:+.3f}"
                )
        lines.append("=" * 70)
        lines.append(f"  {'*** ALARM *** ' if self.alarm else ''}{self.verdict}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def _percentile_indices(resamples: int, ci: float) -> tuple[int, int]:
    alpha = 1.0 - ci
    lo = int((alpha / 2.0) * resamples)
    hi = int((1.0 - alpha / 2.0) * resamples) - 1
    hi = max(lo, min(hi, resamples - 1))
    return lo, hi


def _paired_bootstrap(
    deltas_by_metric: dict[str, list[float]],
    resamples: int,
    seed: int,
    ci: float,
) -> dict[str, tuple[float, float, float]]:
    """Joint paired bootstrap over aligned per-query deltas.

    Every metric shares the *same* resampled query indices each iteration, so the
    pairing across metrics (and across baseline/current) is preserved. Pure
    stdlib ``random`` for reproducibility and zero dependencies.

    Returns ``{metric: (point_delta, ci_low, ci_high)}``.
    """
    names = list(deltas_by_metric)
    n = len(next(iter(deltas_by_metric.values()))) if names else 0
    out: dict[str, tuple[float, float, float]] = {}
    if n == 0:
        return {name: (0.0, 0.0, 0.0) for name in names}

    points = {name: sum(d) / n for name, d in deltas_by_metric.items()}
    boot: dict[str, list[float]] = {name: [] for name in names}
    rng = random.Random(seed)
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        for name, deltas in deltas_by_metric.items():
            boot[name].append(sum(deltas[i] for i in idx) / n)

    lo_i, hi_i = _percentile_indices(resamples, ci)
    for name in names:
        s = sorted(boot[name])
        out[name] = (points[name], s[lo_i], s[hi_i])
    return out


def _classify(recall: MetricDelta, hit: MetricDelta, gen: MetricDelta) -> tuple[str, bool]:
    """Map the three metric deltas to a quadrant; alarm only on silent regression.

    Retrieval is judged regressed when *either* recall@k or hit@k significantly
    drops (in a paired comparison the structural recall ceiling is identical on
    both runs, so a recall delta is meaningful too). Generation is "moved" when
    its CI excludes 0 in either direction.
    """
    retrieval_down = recall.direction == "down" or hit.direction == "down"
    gen_moved = gen.direction != "flat"
    if retrieval_down and not gen_moved:
        return SILENT_REGRESSION, True
    if retrieval_down and gen_moved:
        return VISIBLE_REGRESSION, False
    if not retrieval_down and gen_moved:
        return GENERATION_ONLY, False
    return STABLE, False


def detect_regression(
    baseline: list[RetrievalSample],
    current: list[RetrievalSample],
    gen_scores_baseline: dict[str, float],
    gen_scores_current: dict[str, float],
    k: int,
    *,
    gen_label: str = "faithfulness",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> RegressionReport:
    """Compare two eval runs and flag a silent retrieval regression.

    Queries are aligned by :attr:`RetrievalSample.query` (it doubles as the join
    id, matching the keys of the generation-score maps). A query is included in
    the statistics only if it appears in both runs *and* both score maps;
    everything else is reported and excluded. For each aligned query the function
    computes proportion recall@k, hit@k, and reads the generation score, forms
    per-query deltas (current - baseline), and runs a single joint paired
    bootstrap to get a 95% CI on each metric's mean delta.

    The alarm fires only for the silent-regression quadrant: retrieval
    significantly down (recall@k or hit@k CI below 0) while the generation score
    is statistically unchanged (CI spans 0).

    Args:
        baseline: Retrieval samples from the earlier run.
        current: Retrieval samples from the later run.
        gen_scores_baseline: ``{query_id: score}`` for the earlier run. The score
            is any generation-side signal in [0, 1] the caller computed
            (eval-sanity does not compute it).
        gen_scores_current: ``{query_id: score}`` for the later run.
        k: Retrieval cutoff.
        gen_label: Display name for the generation score.
        resamples: Bootstrap resamples (ADR-0007 default 10 000).
        seed: Bootstrap seed (ADR-0007 default 42) — fixed for reproducibility.
        ci: Confidence level (default 0.95).

    Returns:
        A :class:`RegressionReport`.

    Raises:
        ValueError: if ``k < 1`` or ``ci`` is not in (0, 1).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")

    base_by_id = {s.query: s for s in baseline}
    cur_by_id = {s.query: s for s in current}
    base_ids = set(base_by_id)
    cur_ids = set(cur_by_id)

    only_in_baseline = sorted(base_ids - cur_ids)
    only_in_current = sorted(cur_ids - base_ids)
    in_both = base_ids & cur_ids

    aligned: list[str] = []
    missing_gen: list[str] = []
    for qid in sorted(in_both):
        if qid in gen_scores_baseline and qid in gen_scores_current:
            aligned.append(qid)
        else:
            missing_gen.append(qid)

    per_query: list[PerQueryDelta] = []
    recall_d: list[float] = []
    hit_d: list[float] = []
    gen_d: list[float] = []
    for qid in aligned:
        b, c = base_by_id[qid], cur_by_id[qid]
        rb = recall_at_k(b.retrieved_doc_ids, b.relevant_doc_ids, k)
        rc = recall_at_k(c.retrieved_doc_ids, c.relevant_doc_ids, k)
        hb = hit_at_k(b.retrieved_doc_ids, b.relevant_doc_ids, k)
        hc = hit_at_k(c.retrieved_doc_ids, c.relevant_doc_ids, k)
        gb = gen_scores_baseline[qid]
        gc = gen_scores_current[qid]
        per_query.append(
            PerQueryDelta(qid, rb, rc, hb, hc, gb, gc)
        )
        recall_d.append(rc - rb)
        hit_d.append(hc - hb)
        gen_d.append(gc - gb)

    n = len(aligned)
    boot = _paired_bootstrap(
        {"recall": recall_d, "hit": hit_d, "gen": gen_d}, resamples, seed, ci
    )

    def _mk(name_out: str, key: str, b_vals: list[float], c_vals: list[float]) -> MetricDelta:
        point, lo, hi = boot[key]
        bmean = sum(b_vals) / n if n else 0.0
        cmean = sum(c_vals) / n if n else 0.0
        return MetricDelta(name_out, bmean, cmean, point, lo, hi)

    recall = _mk(
        f"recall@{k}", "recall",
        [q.recall_baseline for q in per_query], [q.recall_current for q in per_query],
    )
    hit = _mk(
        f"hit@{k}", "hit",
        [q.hit_baseline for q in per_query], [q.hit_current for q in per_query],
    )
    gen = _mk(
        gen_label, "gen",
        [q.gen_baseline for q in per_query], [q.gen_current for q in per_query],
    )

    quadrant, alarm = _classify(recall, hit, gen)
    danger = sorted(
        [q for q in per_query if q.recall_delta < 0.0], key=lambda q: q.recall_delta
    )

    return RegressionReport(
        k=k,
        gen_label=gen_label,
        n_aligned=n,
        recall=recall,
        hit=hit,
        gen=gen,
        quadrant=quadrant,
        alarm=alarm,
        per_query=per_query,
        danger_list=danger,
        only_in_baseline=only_in_baseline,
        only_in_current=only_in_current,
        missing_gen=missing_gen,
    )


def from_eval_json(
    baseline_path: str | Path,
    current_path: str | Path,
    k: int,
    *,
    gen_field: str = "faithfulness",
    retrieved_field: str = "retrieved_doc_ids",
    relevant_field: str = "relevant_doc_ids",
    gen_label: str | None = None,
    **kwargs,
) -> RegressionReport:
    """Thin parser shell: load two eval-output JSON files and detect regression.

    Expected schema (per file) — a ``per_query`` object keyed by query id::

        {
          "per_query": {
            "QA-1004": {
              "retrieved_doc_ids": ["d9", "d2", ...],   # ranked, best first
              "relevant_doc_ids":  ["d2", "d7", ...],   # golden set
              "faithfulness": 0.91                       # any gen score in [0,1]
            },
            ...
          }
        }

    The field names are configurable via ``retrieved_field`` / ``relevant_field``
    / ``gen_field`` to fit your pipeline's output. A query missing the generation
    field is parsed for retrieval but its generation score is omitted, so it is
    excluded from the aligned statistics (and reported under ``missing_gen``).

    eval-sanity does not compute the generation score — it only reads whatever
    your run already wrote. All remaining keyword args pass through to
    :func:`detect_regression` (``resamples``, ``seed``, ``ci``).

    Returns:
        A :class:`RegressionReport`.
    """

    def _load(path: str | Path) -> tuple[list[RetrievalSample], dict[str, float]]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        per_query = data.get("per_query")
        if not isinstance(per_query, dict):
            raise ValueError(
                f"{path}: expected a top-level 'per_query' object keyed by query id"
            )
        samples: list[RetrievalSample] = []
        gen_scores: dict[str, float] = {}
        for qid, entry in per_query.items():
            samples.append(
                RetrievalSample(
                    query=qid,
                    retrieved_doc_ids=list(entry.get(retrieved_field, [])),
                    relevant_doc_ids=set(entry.get(relevant_field, [])),
                )
            )
            if gen_field in entry and entry[gen_field] is not None:
                gen_scores[qid] = float(entry[gen_field])
        return samples, gen_scores

    base_samples, base_gen = _load(baseline_path)
    cur_samples, cur_gen = _load(current_path)
    return detect_regression(
        base_samples,
        cur_samples,
        base_gen,
        cur_gen,
        k,
        gen_label=gen_label or gen_field,
        **kwargs,
    )
