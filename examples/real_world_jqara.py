"""
Real-world example: run sanity_report on the JQaRA retrieval results
that motivated eval-sanity's creation.

This example uses the aggregate numbers from eval-driven-llm
(github.com/elvisyao007/eval-driven-llm) to reproduce the oracle-ceiling
finding reported in blog-03.

It does NOT require the full JQaRA dataset. The inputs are the per-query
relevant-doc counts reconstructed from the published summary statistics:
  - 1667 queries
  - mean relevant docs per query ≈ 16  (multi-answer QA dataset)
  - k = 5

Run:
    python examples/real_world_jqara.py
"""

from eval_sanity import RetrievalSample, sanity_report, oracle_ceiling
import random

def make_jqara_approximation(n_queries: int = 1667, mean_rel: float = 16.0,
                              k: int = 5, seed: int = 42) -> list[RetrievalSample]:
    """
    Approximate JQaRA query distribution.
    relevant-doc counts are drawn from a Poisson(mean_rel) distribution,
    clipped to [1, 100] (JQaRA has 100 candidates per query).
    Retrieved docs are simulated as the oracle top-k (best-case retriever).
    """
    rng = random.Random(seed)
    samples = []
    for i in range(n_queries):
        n_rel = max(1, min(100, round(rng.gauss(mean_rel, 5))))
        relevant = [f"doc_{i}_{j}" for j in range(n_rel)]
        # oracle retriever: fills top-k with relevant docs when available
        retrieved = relevant[:k]
        samples.append(RetrievalSample(
            query_id=f"q{i}",
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        ))
    return samples

if __name__ == "__main__":
    k = 5
    threshold = 0.5
    samples = make_jqara_approximation(k=k)

    print("=== oracle ceiling (before retrieval) ===")
    ceiling = oracle_ceiling([s.relevant_doc_ids for s in samples], k=k)
    print(ceiling)

    print()
    print("=== sanity report (oracle retriever) ===")
    report = sanity_report(samples, k=k, threshold=threshold)
    print(report)
    print()
    print("Key finding: even a perfect retriever cannot clear the",
          f"recall@{k} >= {threshold} threshold for",
          f"{report.n_unreachable}/{report.n_queries} queries.")
    print("This is the structural artifact reported in blog-03:")
    print("https://dev.to/elvisyao007/the-33-grounded-but-wrong-answers-were-a-metric-artifact-how-id-based-context-recall-lies-on-ghg")
