# The reranker already scores every chunk from 0 to 1 for the question,
# so we reuse it as the "is the context good enough?" signal.
# This replaces a Gemini call, so it is free, instant and repeatable.

# Below this best score, the documents almost certainly do not contain
# the answer (in testing, unrelated chunks scored around 0.0001 - 0.02,
# real matches scored 0.5 - 1.0). Tune this number in the evaluation phase.
MIN_CONTEXT_SCORE = 0.05


def evaluate_context(
    query: str,
    results: list[dict],
) -> float:
    """
    Returns the best rerank score (0 to 1) among the retrieved chunks,
    or 0.0 when nothing was retrieved.
    """
    scores = [
        result["rerank_score"]
        for result in results
        if result.get("rerank_score") is not None
    ]

    if not scores:
        return 0.0

    return max(0.0, min(max(scores), 1.0))