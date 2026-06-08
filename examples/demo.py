"""eval-sanity 30-second demo — runnable with plain `python examples/demo.py`.

No models, no datasets to download, no network. We synthesize a small retrieval
set that mixes single-answer and *multi-answer* queries (the latter have several
correct documents) and let the report show how proportion recall understates a
retriever that hit@k says is doing fine.

The retriever below is deliberately *good* at the only thing that usually
matters — it surfaces a correct document near the top of every query. But for
the multi-answer queries there are more correct documents than the cutoff k can
hold, so proportion recall can never reward it fully. That gap is the metric
artifact: the average recall looks like a retrieval failure, but a perfect
retriever could not have scored higher.
"""

from eval_sanity import RetrievalSample, oracle_ceiling, sanity_report


def make_dataset() -> list[RetrievalSample]:
    """20 queries at k-relevant of either 1 (single-answer) or 12 (multi-answer).

    The retriever always puts a correct doc at rank 1, and for multi-answer
    queries gets a second correct doc into the top-5 — genuinely useful results.
    Twelve relevant docs is enough that with k=5 the proportion-recall ceiling
    falls below 0.5, so the artifact shows even against the lenient 0.5 threshold
    blog-03 uses.
    """
    samples = []
    for i in range(20):
        multi = i % 2 == 0  # half the dataset is multi-answer
        if multi:
            relevant = {f"q{i}_doc{j}" for j in range(12)}  # 12 correct answers
            retrieved = [
                f"q{i}_doc0",       # rank 1: relevant
                f"q{i}_noise_a",
                f"q{i}_doc1",       # rank 3: relevant
                f"q{i}_noise_b",
                f"q{i}_noise_c",
            ]
        else:
            relevant = {f"q{i}_doc0"}  # 1 correct answer
            retrieved = [
                f"q{i}_doc0",       # rank 1: relevant
                f"q{i}_noise_a",
                f"q{i}_noise_b",
                f"q{i}_noise_c",
                f"q{i}_noise_d",
            ]
        samples.append(
            RetrievalSample(
                query=f"query {i}",
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=relevant,
            )
        )
    return samples


def main() -> None:
    samples = make_dataset()
    K = 5
    THRESHOLD = 0.5

    print("Dataset: 20 queries — 10 single-answer (1 relevant doc),")
    print("         10 multi-answer (12 relevant docs each).")
    print("Retriever: always returns a correct doc at rank 1; for multi-answer")
    print("           queries a second correct doc lands in the top-5.\n")

    # The ceiling alone already tells you the target is partly impossible.
    oc = oracle_ceiling([s.relevant_doc_ids for s in samples], k=K)
    print(
        f"Oracle ceiling on proportion recall@{K}: {oc.mean_ceiling:.2f} "
        f"— the BEST any retriever could average.\n"
    )

    report = sanity_report(samples, k=K, threshold=THRESHOLD)
    print(report.format())
    print()
    print(">> Takeaway:", report.headline)


if __name__ == "__main__":
    main()
