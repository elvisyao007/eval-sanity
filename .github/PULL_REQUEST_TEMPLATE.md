## Summary

<!-- One or two sentences describing what this PR does and why. -->

## Changes

<!-- Bullet list of what changed. -->

## Testing

<!-- How did you verify this? Which test files were added or updated? -->

- [ ] `pytest tests/ -v` passes locally on Python 3.10+
- [ ] `ruff check eval_sanity tests` passes with no errors

## Checklist

- [ ] **Does this PR add a runtime dependency?** Must be **No** — `dependencies = []` in `pyproject.toml` must stay empty. If a tool is needed for docs or dev only, it goes in an optional extra.
- [ ] No assertions in `tests/` were removed or weakened to make tests pass.
- [ ] `CHANGELOG.md` updated (if this is a user-visible change).
- [ ] `__version__` bumped in `eval_sanity/__init__.py` (if this is a release PR).
