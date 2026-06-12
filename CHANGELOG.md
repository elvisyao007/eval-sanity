# Changelog

All notable changes are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.3.0] — 2026-06

### Added
- `trajectory_report(trajectory, spec) -> TrajectoryReport`: deterministic
  agent trajectory audit against a declarative spec. Four checks, all
  without LLM calls:
  - **tool_call_correctness**: required tools present? forbidden tools avoided?
    Score is fraction of expected tools that appear.
  - **order_constraint_check**: precedence rules enforced — e.g. `write_back`
    must be preceded by a `validate` call whose result has `passed=True`.
    Returns a list of `OrderViolation` objects, one per infraction.
  - **step_efficiency**: total steps vs optional budget; redundant calls
    (same tool, same args) detected and reported.
  - **task_completion**: `final_result["status"]` matched against expected?
- `Trajectory`, `TrajectoryStep`: data-contract dataclasses matching the
  Phase A invoice-agent trace JSON schema. `Trajectory.from_json(path)`
  for one-line loading.
- `TrajectorySpec`, `OrderConstraint`: declarative spec types. Separate
  `expected_tools` (must appear), `forbidden_tools` (must not appear),
  `order_constraints`, `max_steps`, `expected_final_status`.
- `detect_trajectory_regression(baseline, candidate)`: compares two sets of
  `TrajectoryReport` objects and alarms when completion_rate drops, mean
  step count rises, or violation rate increases beyond configurable thresholds.
  Returns a `TrajectoryRegressionReport` with per-metric deltas and a
  verdict line. (Extends the "silent regression" concept from v0.2 to agent
  trajectories; threshold-based rather than bootstrap-CI because trajectory
  sets are typically small.)
- `examples/trajectory_demo.py`: self-contained demo with four bad-trace
  scenarios (skip validate, failed-validate write_back, redundant calls,
  wrong tool) and a regression scenario. Accepts `--traces-dir` to load
  real Phase A traces.
- `onprem-llm-stack/payloads/invoice-agent/eval_trajectories.py`: dogfooding
  script — loads all invoice-agent Phase A traces and audits them with
  `trajectory_report`, then runs `detect_trajectory_regression` on the two
  INV-001 runs.

### Changed
- `__version__` bumped to `"0.3.0"`.
- No changes to v0.1/v0.2 retrieval-metric or regression-detection API.

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
