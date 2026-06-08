# eval-sanity

**Audit whether your RAG retrieval metrics can be trusted on a given dataset — before you trust the number on your dashboard.**

eval-sanity is a tiny, zero-dependency diagnostic tool. It does **not** run models, build pipelines, or call a judge. It takes the retrieved/relevant document ids you already have and tells you whether the metric you are averaging is *structurally capable* of saying what you think it says.

## The problem it catches

The most common retrieval metric, **proportion recall@k** (`relevant-found / relevant-total`), has a mechanical ceiling: a query with `n_rel` relevant documents can put at most `min(k, n_rel)` of them in the top-k. When `n_rel > k`, the best possible recall is `k / n_rel < 1.0` — *no matter how good your retriever is*.

On a multi-answer dataset this means your averaged recall can look like a retrieval failure when it is really a **metric artifact**. `hit@k` (did *any* relevant doc land in the top-k?) does not have this defect. eval-sanity makes the gap between the two visible and tells you, in one sentence, what fraction of your dataset cannot pass your threshold even under perfect retrieval.

> This is the productized version of a finding from blog-03 ([BLOG-03-URL]): on a multi-answer Japanese QA set, ~1/3 of queries were structurally unable to clear a recall threshold that *perfect* retrieval could not have cleared.

## Install

```bash
pip install eval-sanity
```

Python ≥ 3.10. No runtime dependencies (stdlib only).

## 30 seconds: watch proportion recall lie while hit@k stays honest

```python
from eval_sanity import RetrievalSample, sanity_report

# 20 queries: 10 single-answer, 10 multi-answer (12 relevant docs each).
# The retriever always returns a correct doc at rank 1 — genuinely useful.
samples = [...]  # see examples/demo.py for the full synthetic set

report = sanity_report(samples, k=5, threshold=0.5)
print(report)
```

Output:

```
eval-sanity report  (k=5, threshold=0.5, n=20)
----------------------------------------------------------------
  mean proportion recall@5 : 0.583
  mean hit@5                : 1.000
  divergence (hit - recall)       : 0.417
  oracle ceiling@5          : 0.708  (best any retriever could average)
  queries capped below recall=1.0 : 10/20
  threshold unreachable for       : 10/20  (50%)
----------------------------------------------------------------
  Your recall@5 >= 0.5 threshold is UNREACHABLE for 50% of queries on this
  dataset (10/20): they have more relevant docs than k=5 can hold, so even
  perfect retrieval fails them. hit@5 (1.00) tells the honest story;
  proportion recall (0.58) is understating retrieval by 0.42.
```

A `recall@5` of **0.58** reads as "the retriever is missing 40% of the answers." eval-sanity shows the **oracle ceiling is only 0.71** — the best *any* retriever could average — and that for **half the dataset** even the lenient 0.5 threshold was never reachable. `hit@5 = 1.00` tells you the retriever actually found a correct doc for every single query.

Run it yourself:

```bash
python examples/demo.py
```

## API

```python
from eval_sanity import (
    RetrievalSample,      # query + retrieved_doc_ids + relevant_doc_ids
    sanity_report,        # the audit: divergence + unreachable-threshold verdict
    oracle_ceiling,       # theoretical best proportion recall@k for a dataset
    # deterministic metrics, if you want the raw numbers:
    recall_at_k, hit_at_k, precision_at_1, reciprocal_rank, ndcg_at_k, aggregate,
    context_recall_docs, grounded_but_wrong_flag,
)
```

### `oracle_ceiling(relevant_doc_ids_per_query, k) -> OracleCeiling`

Given how many relevant docs each query has, computes the highest average proportion recall@k *any* retriever could achieve. No retrieval results needed — use it to check whether a target is even attainable before blaming the retriever.

### `sanity_report(samples, k, threshold=0.8) -> SanityReport`

The full audit. Compares proportion recall against `hit@k`, computes the oracle ceiling, and counts queries whose ceiling falls below `threshold` (structurally unreachable: `n_rel > k / threshold`). Read `report.headline` for the one-liner or `print(report)` for the full block.

## What it is not (v0.1 scope)

- **Not an eval framework.** It scores nothing end-to-end and runs no models.
- **No judge / faithfulness / LLM calls.** The `faithfulness` half of the grounded-but-wrong check is intentionally left for v2; the deterministic anchor (`context_recall_docs`, `grounded_but_wrong_flag`) ships here.
- **No retriever, embedder, or dataset adapters.** Bring your own ids.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
