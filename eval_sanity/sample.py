"""The one data model: a single query's retrieval result vs. ground truth.

This is intentionally generic — no judge fields, no answer text, no model
config. eval-sanity only needs the ranked list of ids your retriever returned
and the set of ids your golden set says are relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalSample:
    """One query: what was retrieved, and what was actually relevant.

    Args:
        query: The query string. Carried for reporting only; metrics ignore it.
        retrieved_doc_ids: Ranked list of doc ids the retriever returned, best
            first. Order matters for rank-based metrics; duplicates are rejected
            by the metric functions because they silently corrupt ranks.
        relevant_doc_ids: The set of doc ids the golden set marks relevant for
            this query. Its size (``n_rel``) is what drives the structural
            diagnostics — a query with many relevant docs cannot be fully
            "recalled" into a small top-k.
    """

    query: str
    retrieved_doc_ids: list[str] = field(default_factory=list)
    relevant_doc_ids: set[str] = field(default_factory=set)

    @property
    def n_rel(self) -> int:
        """Number of relevant docs for this query (the golden answer count)."""
        return len(self.relevant_doc_ids)
