from backend.app.services.gemini_service import gemini_service


def check_grounding(
    query: str,
    answer: str,
    results: list[dict],
) -> float:
    if not answer or not results:
        return 0.0

    context = "\n\n".join(
        f"SOURCE {index}:\n{result['text']}"
        for index, result in enumerate(results, start=1)
    )

    prompt = f"""
You are a grounding evaluator for a
Retrieval-Augmented Generation system.

Determine whether the generated answer is supported
by the supplied retrieved context.

User question:
{query}

Generated answer:
{answer}

Retrieved context:
{context}

Rules:
1. Use ONLY the retrieved context.
2. Check whether the claims in the answer are supported.
3. Do not judge whether the answer is stylistically good.
4. Return ONLY a decimal number between 0 and 1.
5. 1 means fully supported.
6. 0 means completely unsupported.

Grounding score:
"""

    response = gemini_service.generate(prompt)

    try:
        score = float(response.strip())
    except ValueError:
        return 0.0

    return max(0.0, min(score, 1.0))