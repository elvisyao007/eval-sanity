# eval-sanity

![tests](https://github.com/elvisyao007/eval-sanity/actions/workflows/test.yml/badge.svg) [![PyPI](https://img.shields.io/pypi/v/eval-sanity)](https://pypi.org/project/eval-sanity/) [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Audit whether your RAG retrieval metrics can be trusted on a given dataset — before you trust the number on your dashboard.**

eval-sanity is a tiny, zero-dependency diagnostic tool. It does **not** run models, build pipelines, or call a judge. It takes the retrieved/relevant document ids you already have and tells you whether the metric you are averaging is *structurally capable* of saying what you think it says.

## The problem it catches

The most common retrieval metric, **proportion recall@k** (`relevant-found / relevant-total`), has a mechanical ceiling: a query with `n_rel` relevant documents can put at most `min(k, n_rel)` of them in the top-k. When `n_rel > k`, the best possible recall is `k / n_rel < 1.0` — *no matter how good your retriever is*.

On a multi-answer dataset this means your averaged recall can look like a retrieval failure when it is really a **metric artifact**. `hit@k` (did *any* relevant doc land in the top-k?) does not have this defect. eval-sanity makes the gap between the two visible and tells you, in one sentence, what fraction of your dataset cannot pass your threshold even under perfect retrieval.

> This is the productized version of a finding from blog-03 ([https://dev.to/elvisyao007/the-33-grounded-but-wrong-answers-were-a-metric-artifact-how-id-based-context-recall-lies-on-ghg]): on a multi-answer Japanese QA set, ~1/3 of queries were structurally unable to clear a recall threshold that *perfect* retrieval could not have cleared.

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

## Detecting silent regressions (v0.2)

A *silent regression* is the one your dashboard misses: retrieval quality drops by a real, measurable amount, but the generation-side score (faithfulness) barely moves — the model keeps writing fluent, grounded-sounding answers from the now-wrong context. A faithfulness-only dashboard stays green while the retriever rots.

`detect_regression` compares two eval runs and flags exactly this. Every metric delta carries a **95% confidence interval from a paired bootstrap** (seed 42, 10 000 resamples — re-runs are bit-for-bit identical), so the verdict is statistical, never a bare threshold. The alarm fires *only* when retrieval significantly drops **and** generation is statistically unchanged.

```python
from eval_sanity import RetrievalSample, detect_regression

# baseline / current: list[RetrievalSample] from two runs.
# Generation scores are passed in by you (eval-sanity never calls a judge).
report = detect_regression(
    baseline, current,
    gen_scores_baseline={"q0": 0.91, ...},   # e.g. faithfulness per query id
    gen_scores_current={"q0": 0.90, ...},
    k=5,
)
print(report)
```

Output on a synthetic silent-regression scenario (`examples/regression_demo.py`):

```
eval-sanity regression report  (k=5, n_aligned=60, resamples=10000, seed=42)
======================================================================
  recall@5 : 0.950 -> 0.667  (-0.283) [CI -0.417,-0.150]  [down]
  hit@5    : 0.950 -> 0.667  (-0.283) [CI -0.417,-0.150]  [down]
  faithfulness: 0.900 -> 0.900  (+0.000) [CI -0.005,+0.005]  [flat]
----------------------------------------------------------------------
  *** ALARM *** SILENT REGRESSION detected: recall@5 -0.283 [CI -0.417,-0.150],
  faithfulness unchanged [CI -0.005,+0.005] — your dashboard won't show this.
```

It classifies every comparison into four quadrants and alarms on only one:

| retrieval ↓ (CI) | generation moved (CI) | verdict | alarm |
|---|---|---|---|
| yes | no | **silent regression** | 🚨 |
| yes | yes | visible regression | — |
| no | yes | generation-only change | — |
| no | no | stable | — |

Have eval outputs on disk? `from_eval_json(baseline_path, current_path, k=5)` is a thin parser shell over `detect_regression` for JSON with a `per_query` map of `{retrieved_doc_ids, relevant_doc_ids, faithfulness}` (field names are configurable). Run the demo:

```bash
python examples/regression_demo.py
```

## API

```python
from eval_sanity import (
    RetrievalSample,      # query + retrieved_doc_ids + relevant_doc_ids
    sanity_report,        # the audit: divergence + unreachable-threshold verdict
    oracle_ceiling,       # theoretical best proportion recall@k for a dataset
    detect_regression,    # v0.2: silent-regression detection w/ paired-bootstrap CIs
    from_eval_json,       # v0.2: same, from two eval-output JSON files
    # deterministic metrics, if you want the raw numbers:
    recall_at_k, hit_at_k, precision_at_1, reciprocal_rank, ndcg_at_k, aggregate,
    context_recall_docs, grounded_but_wrong_flag,
)
```

### `oracle_ceiling(relevant_doc_ids_per_query, k) -> OracleCeiling`

Given how many relevant docs each query has, computes the highest average proportion recall@k *any* retriever could achieve. No retrieval results needed — use it to check whether a target is even attainable before blaming the retriever.

### `sanity_report(samples, k, threshold=0.8) -> SanityReport`

The full audit. Compares proportion recall against `hit@k`, computes the oracle ceiling, and counts queries whose ceiling falls below `threshold` (structurally unreachable: `n_rel > k / threshold`). Read `report.headline` for the one-liner or `print(report)` for the full block.

### `detect_regression(baseline, current, gen_scores_baseline, gen_scores_current, k) -> RegressionReport`

Compares two runs (`list[RetrievalSample]` each) plus caller-supplied generation scores (`{query_id: float}`), aligns queries by id, and runs a paired bootstrap to put a 95% CI on each metric's delta. Returns the quadrant, the alarm flag (silent regression only), a per-query danger list sorted by recall drop, and a one-line `verdict`. `from_eval_json(...)` wraps it for JSON inputs.

## What it is not

- **Not an eval framework.** It scores nothing end-to-end and runs no models.
- **No judge / faithfulness / LLM calls.** eval-sanity never computes a generation score — for both the v0.1 grounded-but-wrong check and the v0.2 regression detector, faithfulness is supplied by you. The deterministic anchors (`context_recall_docs`, `grounded_but_wrong_flag`) ship here.
- **No retriever, embedder, or dataset adapters.** Bring your own ids.
- **Zero runtime dependencies.** The paired bootstrap is pure-stdlib `random` + sorting; no numpy.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
