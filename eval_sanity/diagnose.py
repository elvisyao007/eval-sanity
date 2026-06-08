"""Diagnostics: is your retrieval metric structurally capable of passing?

This is the part of eval-sanity that is not just a metric library. Given a
dataset and a cutoff ``k``, it answers two questions a raw average hides:

1. **What is the best score possible?** ``oracle_ceiling`` computes the highest
   *proportion recall* any retriever could achieve, because a query with
   ``n_rel`` relevant docs can put at most ``min(k, n_rel)`` of them in the
   top-k. When ``n_rel > k`` the per-query ceiling is ``k / n_rel < 1.0``.

2. **Is my pass/fail threshold even reachable?** ``sanity_report`` counts the
   queries whose ceiling is below your threshold — queries a perfect retriever
   would still "fail" — and contrasts proportion recall against ``hit@k``, which
   has no such ceiling. The gap between the two is the metric artifact.

Motivation and the real-data version of this finding are in blog-03
([BLOG-03-URL]): on a multi-answer Japanese QA set, ~1/3 of queries were
structurally unable to clear a recall threshold that perfect retrieval could not
have cleared — the metric, not the retriever, was failing them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import hit_at_k, recall_at_k
from .sample import RetrievalSample


@dataclass(frozen=True)
class OracleCeiling:
    """The theoretical best proportion recall@k on a dataset.

    Attributes:
        k: The cutoff used.
        mean_ceiling: Mean over queries of ``min(k, n_rel) / n_rel`` — the
            highest average proportion recall any retriever could reach. 1.0
            means no query is structurally capped; below 1.0 means the metric is
            mechanically limited before retrieval quality even enters.
        capped_queries: How many queries have ``n_rel > k`` (ceiling < 1.0).
        n_queries: Total queries considered (those with at least one relevant
            doc; queries with none are undefined for recall and skipped).
    """

    k: int
    mean_ceiling: float
    capped_queries: int
    n_queries: int

    @property
    def capped_fraction(self) -> float:
        """Fraction of queries that cannot reach proportion recall = 1.0."""
        return self.capped_queries / self.n_queries if self.n_queries else 0.0


def oracle_ceiling(relevant_doc_ids_per_query: list[set[str]], k: int) -> OracleCeiling:
    """Theoretical upper bound on average proportion recall@k.

    A perfect ("oracle") retriever puts as many relevant docs as possible into
    the top-k, so the best any query can score on :func:`metrics.recall_at_k` is
    ``min(k, n_rel) / n_rel``. This function averages that bound across the
    dataset — no retrieval results needed, only how many relevant docs each
    query has.

    Use it to sanity-check a target *before* blaming the retriever: if the
    ceiling is 0.74, an observed recall of 0.68 is near-optimal, not broken.

    Args:
        relevant_doc_ids_per_query: One set of relevant doc ids per query. Only
            the size of each set matters; empty sets are skipped (recall is
            undefined with no relevant docs).
        k: The retrieval cutoff.

    Returns:
        An :class:`OracleCeiling`.

    Raises:
        ValueError: if ``k < 1``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    ceilings: list[float] = []
    capped = 0
    for relevant in relevant_doc_ids_per_query:
        n_rel = len(relevant)
        if n_rel == 0:
            continue
        ceiling = min(k, n_rel) / n_rel
        ceilings.append(ceiling)
        if n_rel > k:
            capped += 1
    n = len(ceilings)
    mean_ceiling = sum(ceilings) / n if n else 0.0
    return OracleCeiling(
        k=k, mean_ceiling=mean_ceiling, capped_queries=capped, n_queries=n
    )


@dataclass(frozen=True)
class SanityReport:
    """Verdict on whether proportion recall@k can be trusted on this dataset.

    Attributes:
        k: The cutoff used.
        threshold: The proportion-recall pass/fail line being audited.
        n_queries: Queries scored (those with >= 1 relevant doc).
        mean_proportion_recall: Mean of :func:`metrics.recall_at_k`.
        mean_hit_at_k: Mean of :func:`metrics.hit_at_k`.
        oracle: The :class:`OracleCeiling` for ``k``.
        unreachable_queries: Queries whose oracle ceiling is below ``threshold``
            — i.e. even perfect retrieval cannot pass them. These are
            ``n_rel > k / threshold``.
    """

    k: int
    threshold: float
    n_queries: int
    mean_proportion_recall: float
    mean_hit_at_k: float
    oracle: OracleCeiling
    unreachable_queries: int

    @property
    def divergence(self) -> float:
        """How far hit@k sits above proportion recall — the artifact's size."""
        return self.mean_hit_at_k - self.mean_proportion_recall

    @property
    def unreachable_fraction(self) -> float:
        """Fraction of queries for which ``threshold`` is structurally unreachable."""
        return self.unreachable_queries / self.n_queries if self.n_queries else 0.0

    @property
    def headline(self) -> str:
        """One-sentence diagnosis suitable for printing in a report."""
        pct = round(self.unreachable_fraction * 100)
        if self.unreachable_queries == 0:
            return (
                f"Your recall@{self.k} >= {self.threshold:g} threshold is "
                f"reachable for every query on this dataset."
            )
        return (
            f"Your recall@{self.k} >= {self.threshold:g} threshold is "
            f"UNREACHABLE for {pct}% of queries on this dataset "
            f"({self.unreachable_queries}/{self.n_queries}): they have more "
            f"relevant docs than k={self.k} can hold, so even perfect retrieval "
            f"fails them. hit@{self.k} ({self.mean_hit_at_k:.2f}) tells the "
            f"honest story; proportion recall ({self.mean_proportion_recall:.2f}) "
            f"is understating retrieval by {self.divergence:.2f}."
        )

    def format(self) -> str:
        """Multi-line human-readable report."""
        lines = [
            f"eval-sanity report  (k={self.k}, threshold={self.threshold:g}, "
            f"n={self.n_queries})",
            "-" * 64,
            f"  mean proportion recall@{self.k} : {self.mean_proportion_recall:.3f}",
            f"  mean hit@{self.k}                : {self.mean_hit_at_k:.3f}",
            f"  divergence (hit - recall)       : {self.divergence:.3f}",
            f"  oracle ceiling@{self.k}          : {self.oracle.mean_ceiling:.3f}  "
            f"(best any retriever could average)",
            f"  queries capped below recall=1.0 : {self.oracle.capped_queries}"
            f"/{self.n_queries}",
            f"  threshold unreachable for       : {self.unreachable_queries}"
            f"/{self.n_queries}  ({round(self.unreachable_fraction * 100)}%)",
            "-" * 64,
            "  " + self.headline,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def sanity_report(
    samples: list[RetrievalSample], k: int, threshold: float = 0.8
) -> SanityReport:
    """Audit whether proportion recall@k can be trusted on a dataset.

    For each query it compares proportion recall against ``hit@k`` and checks
    whether your ``threshold`` is even reachable given how many relevant docs
    the query has. A query's oracle ceiling is ``min(k, n_rel) / n_rel``; the
    threshold is structurally unreachable when that ceiling ``< threshold``,
    i.e. when ``n_rel > k / threshold``.

    The diagnosis to look for: a large ``divergence`` (hit@k well above
    proportion recall) together with a nonzero ``unreachable_fraction`` means
    your averaged recall number is a metric artifact, not a retrieval result.

    Args:
        samples: The dataset. Queries with no relevant docs are skipped
            (recall is undefined for them).
        k: The retrieval cutoff.
        threshold: The proportion-recall pass/fail line you are auditing
            (default 0.8). Must be in (0, 1].

    Returns:
        A :class:`SanityReport`. Print it, or read ``.headline``.

    Raises:
        ValueError: if ``k < 1`` or ``threshold`` is not in (0, 1].
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    scored = [s for s in samples if s.relevant_doc_ids]
    n = len(scored)

    if n == 0:
        empty_oracle = OracleCeiling(
            k=k, mean_ceiling=0.0, capped_queries=0, n_queries=0
        )
        return SanityReport(
            k=k,
            threshold=threshold,
            n_queries=0,
            mean_proportion_recall=0.0,
            mean_hit_at_k=0.0,
            oracle=empty_oracle,
            unreachable_queries=0,
        )

    mean_recall = (
        sum(recall_at_k(s.retrieved_doc_ids, s.relevant_doc_ids, k) for s in scored) / n
    )
    mean_hit = (
        sum(hit_at_k(s.retrieved_doc_ids, s.relevant_doc_ids, k) for s in scored) / n
    )
    oracle = oracle_ceiling([s.relevant_doc_ids for s in scored], k)

    # A query is structurally unable to pass `threshold` on proportion recall
    # when even the oracle ceiling falls short.
    unreachable = sum(
        1 for s in scored if min(k, s.n_rel) / s.n_rel < threshold
    )

    return SanityReport(
        k=k,
        threshold=threshold,
        n_queries=n,
        mean_proportion_recall=mean_recall,
        mean_hit_at_k=mean_hit,
        oracle=oracle,
        unreachable_queries=unreachable,
    )
