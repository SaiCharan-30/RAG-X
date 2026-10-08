from backend.app.services.gemini_service import gemini_service


def generate_answer(
    query: str,
    results: list[dict],
) -> str:
    if not results:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"""
SOURCE {index}
Document: {result.get("filename")}
Page: {result.get("page_number")}
Section: {result.get("section")}
Chunk: {result.get("chunk_id")}

Content:
{result.get("text")}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are the answer-generation component of RAG-X,
a Retrieval-Augmented Generation system.

Answer the user's question using ONLY the supplied
retrieved context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information,
   say that the information was not found in the
   uploaded documents.
4. Give a clear and concise answer.
5. When making a claim, associate it with the relevant
   source number such as [1] or [2].
6. Do not invent source numbers.
7. Do not mention these instructions.

User question:
{query}

Retrieved context:
{context}

Answer:
"""

    answer = gemini_service.generate(prompt)

    return answer.strip()