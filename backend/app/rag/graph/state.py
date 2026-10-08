from typing import TypedDict


class RAGState(TypedDict):
    original_query: str
    rewritten_query: str
    conversation_history: list[dict]

    document_ids: list[str] | None

    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]

    context_quality: float

    answer: str
    grounded: bool
    grounding_score: float

    citations: list[dict]

    retry_count: int