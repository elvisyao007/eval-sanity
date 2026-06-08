"""eval-sanity: audit whether your RAG retrieval metrics can be trusted.

A small, dependency-free diagnostic tool. It does *not* run models or build an
eval pipeline — it takes the retrieved/relevant doc ids you already have and
tells you whether the metric you are averaging is structurally capable of saying
what you think it says.

The headline failure mode it catches: on multi-answer datasets, *proportion
recall* (relevant-found / relevant-total) is mechanically capped below 1.0 for
any query with more relevant docs than your cutoff k. Averaging it then reports
a low number that looks like a retrieval problem but is really a metric
artifact. ``hit@k`` does not have this defect. See ``diagnose.sanity_report``.
"""

from __future__ import annotations

from .metrics import (
    aggregate,
    context_recall_docs,
    grounded_but_wrong_flag,
    hit_at_k,
    ndcg_at_k,
    precision_at_1,
    recall_at_k,
    reciprocal_rank,
)
from .sample import RetrievalSample
from .diagnose import OracleCeiling, SanityReport, oracle_ceiling, sanity_report

__version__ = "0.1.0"

__all__ = [
    # data model
    "RetrievalSample",
    # deterministic metrics
    "recall_at_k",
    "precision_at_1",
    "reciprocal_rank",
    "ndcg_at_k",
    "hit_at_k",
    "aggregate",
    "context_recall_docs",
    "grounded_but_wrong_flag",
    # diagnostics
    "oracle_ceiling",
    "sanity_report",
    "OracleCeiling",
    "SanityReport",
    "__version__",
]
