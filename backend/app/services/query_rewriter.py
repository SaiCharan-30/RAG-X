from backend.app.services.gemini_service import gemini_service


def rewrite_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    conversation_history = conversation_history or []

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history
    )

    prompt = f"""
You are a query rewriting component in a Retrieval-Augmented
Generation system.

Your task is to rewrite the user's latest question into a
standalone search query.

Rules:
1. Resolve pronouns such as "it", "its", "they", "this", and "that"
   using the conversation history.
2. Preserve the user's original intent.
3. Do not answer the question.
4. Do not add information that is not present in the conversation.
5. Return ONLY the rewritten search query.
6. If the question is already standalone, return it unchanged.

Conversation history:
{history_text}

Latest user question:
{query}

Standalone search query:
"""

    rewritten_query = gemini_service.generate(prompt)

    return rewritten_query.strip()