# Changelog

All notable changes are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-05

### Added
- `detect_regression`: silent-regression detection via paired bootstrap (95% CI,
  10 000 resamples, seed 42). Fires alarm only when retrieval drops significantly
  AND generation is statistically unchanged — the failure mode a faithfulness-only
  dashboard misses.
- `from_eval_json`: thin wrapper around `detect_regression` for JSON inputs with
  a `per_query` map of `{retrieved_doc_ids, relevant_doc_ids, faithfulness}`.
- Four-quadrant classification (silent regression / visible regression /
  generation-only change / stable).
- `examples/regression_demo.py`: reproducible synthetic silent-regression scenario.

### Changed
- None (backward-compatible addition).

---

## [0.1.0] — 2026-04

### Added
- `RetrievalSample`, `sanity_report`, `oracle_ceiling`: core audit — detects
  proportion-recall threshold artifacts and hit@k vs proportion-recall divergence.
- Deterministic retrieval metrics: `recall_at_k`, `hit_at_k`, `precision_at_1`,
  `reciprocal_rank`, `ndcg_at_k`, `aggregate`.
- `context_recall_docs`, `grounded_but_wrong_flag`: deterministic generation anchors.
- `examples/demo.py`: 30-second synthetic demo reproducing the JQaRA finding.
- Zero runtime dependencies (stdlib only). Python ≥ 3.10.
- Apache-2.0 license.

---
