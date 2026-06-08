"""eval-sanity v0.2 demo — catch a silent retrieval regression.

Runnable with plain `python examples/regression_demo.py`. No models, no network.

We synthesize two eval runs over the same 60 queries:

  * BASELINE: retrieval finds the relevant doc for ~95% of queries.
  * CURRENT:  a config change (say, a chunking tweak) quietly drops the relevant
              doc for a third of queries — but the generation model still writes
              fluent, grounded-sounding answers, so the *faithfulness* score
              barely moves.

A faithfulness dashboard shows green. eval-sanity's paired bootstrap shows the
retrieval drop is real (CI excludes 0) while faithfulness is statistically
unchanged (CI spans 0), and raises the silent-regression alarm.

Then we run a pure-noise scenario (a couple of queries flip) to show it does
*not* cry wolf.
"""

from eval_sanity import RetrievalSample, detect_regression

K = 5
N = 60


def sample(qid: str, found: bool) -> RetrievalSample:
    """One query, one relevant doc. `found` -> doc at rank 1; else absent."""
    rel = {f"{qid}_rel"}
    docs = [f"{qid}_rel", f"{qid}_a"] if found else [f"{qid}_a", f"{qid}_b"]
    return RetrievalSample(query=qid, retrieved_doc_ids=docs, relevant_doc_ids=rel)


def scenario_silent_regression():
    # Baseline: 57/60 found. Current: every 3rd query loses its relevant doc.
    baseline = [sample(f"q{i}", i % 20 != 0) for i in range(N)]
    current = [sample(f"q{i}", i % 3 != 0) for i in range(N)]
    # Faithfulness essentially unchanged: the generator stays fluent and grounded
    # in whatever it retrieved, wrong docs included. Tiny deterministic jitter.
    gen_baseline = {f"q{i}": 0.90 + (0.01 if i % 2 else -0.01) for i in range(N)}
    gen_current = {f"q{i}": 0.90 + (-0.01 if i % 2 else 0.01) for i in range(N)}
    return baseline, current, gen_baseline, gen_current


def scenario_noise():
    # Retrieval barely moves: 2 queries flip. Faithfulness flat.
    baseline = [sample(f"q{i}", True) for i in range(N)]
    current = [sample(f"q{i}", i not in (7, 31)) for i in range(N)]
    gen_baseline = {f"q{i}": 0.90 for i in range(N)}
    gen_current = {f"q{i}": 0.90 for i in range(N)}
    return baseline, current, gen_baseline, gen_current


def main() -> None:
    print("#" * 70)
    print("# Scenario A: silent retrieval regression (should ALARM)")
    print("#" * 70)
    b, c, gb, gc = scenario_silent_regression()
    rep = detect_regression(b, c, gb, gc, K)  # ADR-0007 defaults: 10k resamples, seed 42
    print(rep.format(top=5))

    print()
    print("#" * 70)
    print("# Scenario B: pure noise (should NOT alarm)")
    print("#" * 70)
    b, c, gb, gc = scenario_noise()
    rep = detect_regression(b, c, gb, gc, K)
    print(rep.format(top=5))


if __name__ == "__main__":
    main()
