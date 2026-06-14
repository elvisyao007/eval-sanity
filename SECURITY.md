# Security Policy

## Scope

eval-sanity is a **zero-dependency, stdlib-only Python library**. It:

- Does **not** execute language models or call external APIs.
- Does **not** make network requests of any kind.
- Does **not** read or write files except when the caller explicitly passes a path to `Trajectory.from_json()` or `from_eval_json()`.
- Has **no runtime dependencies** beyond the Python standard library.

The attack surface is therefore very small: eval-sanity processes data structures (lists of document IDs and scores) that callers supply. It does not parse untrusted network payloads.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a vulnerability

If you discover a security issue (e.g. a path-traversal bug in `from_json`, an integer overflow in a metric calculation, or an unsafe `eval`/`exec` call), please **do not open a public GitHub issue**.

Instead, email **elvisyao@proton.me** with:

1. A description of the vulnerability and its potential impact.
2. Steps to reproduce (minimal code snippet preferred).
3. Any suggested fix, if you have one.

You will receive an acknowledgement within 72 hours. We aim to release a patch within 14 days for confirmed issues.

## Out of scope

- Bugs that only affect callers who pass malicious data they themselves control (self-inflicted).
- Issues in optional development dependencies (pytest, ruff) that are never installed in production.
