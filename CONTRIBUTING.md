# Contributing to eval-sanity

Thank you for your interest in contributing. This document covers everything you need to get started.

## Development environment

```bash
git clone https://github.com/elvisyao007/eval-sanity.git
cd eval-sanity
pip install -e ".[dev]"
```

The `dev` extra installs pytest and ruff — both are only for development, never runtime dependencies.

## Running tests

```bash
pytest tests/ -v
```

All tests are deterministic (no model calls, no network, no randomness). Every PR must keep the full suite green on Python 3.10, 3.11, and 3.12.

## The zero-dependency rule

**eval-sanity has zero runtime dependencies and this is non-negotiable.**

The `dependencies = []` line in `pyproject.toml` must remain empty. No contribution may add a package to `dependencies`, even a small or well-known one. The stdlib is the only allowed import surface for production code under `eval_sanity/`.

If a contribution genuinely needs a third-party package (e.g. for documentation tooling), it goes in a new named extra under `[project.optional-dependencies]` and must never be imported in `eval_sanity/` itself.

## Code style

Lint is enforced in CI via [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check eval_sanity tests
```

To auto-fix formatting and import issues:

```bash
ruff check --fix eval_sanity tests
```

The ruff config lives in `pyproject.toml` (`[tool.ruff]`). Rules selected: `E` (pycodestyle) and `F` (pyflakes). Long lines (`E501`) are ignored — prefer clear code over wrapping.

## Submitting issues and pull requests

**Issues:**
- Use the bug report template for reproducible problems.
- Use the feature request template for new ideas.
- Include a minimal code snippet that demonstrates the issue.

**Pull requests:**
- Fork the repo, create a branch, open a PR against `main`.
- Fill out the PR template — especially the "Does this PR add a runtime dependency?" checkbox (must be **No**).
- Keep changes focused: one logical change per PR.
- Add or update tests in `tests/` for any new behaviour.
- Do **not** modify the `__all__` list in `eval_sanity/__init__.py` without updating tests.

## Core algorithm files

The files `eval_sanity/metrics.py`, `diagnose.py`, `regression.py`, `trajectory.py`, and `sample.py` contain the deterministic computation logic. Changes to these files require extra care:
- All assertions in `tests/` must remain passing (no test assertions may be weakened or removed to make a change pass).
- Determinism must be preserved: same inputs → bit-for-bit identical outputs across Python versions and runs.

## Versioning and CHANGELOG

eval-sanity follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

- Patch releases (`0.x.y`): bug fixes with no API change.
- Minor releases (`0.x.0`): new public API (backward-compatible).
- Major releases (`1.0.0`): breaking API changes (rare).

Update `CHANGELOG.md` and bump `__version__` in `eval_sanity/__init__.py` as part of any release PR.
