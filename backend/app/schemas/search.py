from pydantic import BaseModel, Field

from backend.app.schemas.chat import ChatRequest
from backend.app.schemas.document_mode import DocumentMode


class SearchRequest(ChatRequest):
    """
    Same fields and checks as a chat question (query, mode, document_ids),
    plus two search options. Reusing ChatRequest keeps the single / selected /
    all document rules in one place.
    """

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    rerank: bool = True


class SearchResult(BaseModel):
    # Final position in this response (1 = best).
    rank: int

    # Position before reranking (what hybrid search alone returned).
    retrieval_rank: int

    # Hybrid-search (rank fusion) score. Only useful for ordering,
    # it is NOT a percentage.
    retrieval_score: float

    # Reranker score (higher = more relevant). None when rerank is off.
    rerank_score: float | None = None

    document_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    section: str | None = None
    chunk_index: int | None = None
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: DocumentMode
    reranked: bool
    result_count: int
    results: list[SearchResult]