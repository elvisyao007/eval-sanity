"""Deterministic retrieval metrics.

No LLM, no randomness: given a ranked list of retrieved document ids and the set
of relevant ids from a frozen golden set, every score here is fully
reproducible. These are the functions eval-sanity audits — the point of the tool
is that *which* of these you average changes the story your dashboard tells.

``recall_at_k`` here is *proportion recall*: of all relevant docs, what fraction
landed in the top-k. ``hit_at_k`` is the binary "did at least one relevant doc
land in the top-k". On single-answer datasets they agree; on multi-answer
datasets they can diverge sharply, and that divergence is the artifact
``diagnose.sanity_report`` surfaces.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check(retrieved: Sequence[str], relevant: set[str]) -> None:
    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set of doc ids")
    if len(set(retrieved)) != len(retrieved):
        # Duplicates in a ranked list silently corrupt rank-based metrics.
        raise ValueError("retrieved list contains duplicate ids")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Proportion recall: fraction of relevant docs found within the top-k.

    Note the structural ceiling: with ``n_rel`` relevant docs, the best possible
    value is ``min(k, n_rel) / n_rel``. When ``n_rel > k`` this is below 1.0 no
    matter how good the retriever is. That ceiling is exactly what
    ``diagnose.oracle_ceiling`` computes.
    """
    _check(retrieved, relevant)
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if *any* relevant doc appears in the top-k, else 0.0.

    Unlike :func:`recall_at_k`, this has no structural ceiling from multi-answer
    queries — a single relevant doc in the top-k scores it. That is what makes
    it the honest signal when relevant-doc counts vary across queries.
    """
    _check(retrieved, relevant)
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def precision_at_1(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1.0 if the top-ranked doc is relevant, else 0.0."""
    _check(retrieved, relevant)
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] in relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant doc (0 if none in list)."""
    _check(retrieved, relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k. Ideal DCG assumes all relevant docs ranked first."""
    _check(retrieved, relevant)
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(
    runs: list[tuple[Sequence[str], set[str]]],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Mean metrics across a golden set.

    ``runs`` is a list of ``(retrieved_ids, relevant_ids)`` per query. Returns
    macro-averaged metrics — fully deterministic given the inputs. Includes both
    ``recall@k`` (proportion) and ``hit@k`` so the divergence is visible.
    """
    if not runs:
        return {}
    n = len(runs)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in runs) / n
        out[f"hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in runs) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, rel, k) for r, rel in runs) / n
    out["mrr"] = sum(reciprocal_rank(r, rel) for r, rel in runs) / n
    out["p@1"] = sum(precision_at_1(r, rel) for r, rel in runs) / n
    return out


def context_recall_docs(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: set[str]
) -> float:
    """Ground-truth-anchored recall over the whole retrieved set (no cutoff).

    Fraction of the golden relevant documents that actually made it into the
    generation context window. This is the anchor a faithfulness/judge score is
    meant to be read against: low context recall + high faithfulness = the
    system is confidently grounded in the WRONG documents (the grounded-but-
    wrong failure mode). The judge half lives in v2; this deterministic anchor
    does not need a model.
    """
    if not relevant_doc_ids:
        return 0.0
    got = set(retrieved_doc_ids) & relevant_doc_ids
    return len(got) / len(relevant_doc_ids)


def grounded_but_wrong_flag(
    faith: float, ctx_recall: float, faith_hi: float = 0.8, recall_lo: float = 0.5
) -> bool:
    """True when an answer looks faithful but retrieval missed the truth.

    The failure mode a faithfulness-only eval would hide. ``faith`` is supplied
    by the caller (any grounding/faithfulness score in [0, 1]); eval-sanity does
    not compute it — it only encodes the rule that reads it against a
    ground-truth-anchored recall.
    """
    return faith >= faith_hi and ctx_recall < recall_lo
