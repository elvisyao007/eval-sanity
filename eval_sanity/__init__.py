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

v0.3 extends eval-sanity from retrieval metrics to agent trajectory audits:
``trajectory_report`` deterministically checks tool-call correctness, order
constraints, step efficiency, and task completion — no LLM required.
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
from .regression import (
    MetricDelta,
    PerQueryDelta,
    RegressionReport,
    detect_regression,
    from_eval_json,
)
from .trajectory import (
    # data model
    Trajectory,
    TrajectoryStep,
    # spec
    OrderConstraint,
    TrajectorySpec,
    # result types
    ToolCallCorrectness,
    OrderViolation,
    RedundantCall,
    StepEfficiency,
    TaskCompletion,
    TrajectoryReport,
    # evaluation
    trajectory_report,
    # regression detection
    TrajectorySetStats,
    TrajectoryRegressionReport,
    detect_trajectory_regression,
)

__version__ = "0.3.0"

__all__ = [
    # data model — retrieval
    "RetrievalSample",
    # deterministic retrieval metrics
    "recall_at_k",
    "precision_at_1",
    "reciprocal_rank",
    "ndcg_at_k",
    "hit_at_k",
    "aggregate",
    "context_recall_docs",
    "grounded_but_wrong_flag",
    # retrieval diagnostics
    "oracle_ceiling",
    "sanity_report",
    "OracleCeiling",
    "SanityReport",
    # retrieval regression detection (v0.2)
    "detect_regression",
    "from_eval_json",
    "RegressionReport",
    "MetricDelta",
    "PerQueryDelta",
    # trajectory evaluation (v0.3)
    "Trajectory",
    "TrajectoryStep",
    "OrderConstraint",
    "TrajectorySpec",
    "ToolCallCorrectness",
    "OrderViolation",
    "RedundantCall",
    "StepEfficiency",
    "TaskCompletion",
    "TrajectoryReport",
    "trajectory_report",
    "TrajectorySetStats",
    "TrajectoryRegressionReport",
    "detect_trajectory_regression",
    "__version__",
]
