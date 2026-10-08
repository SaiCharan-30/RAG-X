from backend.app.services.gemini_service import gemini_service


def evaluate_context(
    query: str,
    results: list[dict],
) -> float:
    if not results:
        print("DEBUG: No retrieval results.")
        return 0.0

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
You are evaluating retrieval quality for a
Retrieval-Augmented Generation system.

User query:
{query}

Retrieved context:
{context}

Rate how well the retrieved context can answer the query.

Return ONLY a decimal number between 0 and 1.

Examples:
0
0.25
0.5
0.75
1

Do not return words.
Do not return Markdown.
Do not explain your answer.
"""

    response = gemini_service.generate(prompt)

    print("DEBUG - Raw Gemini response:", repr(response))

    try:
        score = float(response.strip())
    except ValueError:
        print("DEBUG - Could not convert response to float.")
        return 0.0

    score = max(0.0, min(score, 1.0))

    print("DEBUG - Parsed context score:", score)

    return score