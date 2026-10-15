from backend.app.services.embeddings import embedding_service
from backend.app.services.vector_store import vector_store
from backend.app.services.reranker import reranker
from backend.app.services.sparse_embeddings import (
    sparse_embedding_service,
)

def semantic_search(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Perform semantic search and return clean retrieval results.
    """

    query_vector = embedding_service.embed_text(query)

    results = vector_store.search(
        query_vector=query_vector,
        limit=limit,
    )

    search_results = []

    for result in results:
        payload = result.payload or {}

        search_results.append(
            {
                "score": result.score,
                "document_id": payload.get("document_id"),
                "chunk_id": payload.get("chunk_id"),
                "filename": payload.get("filename"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "page_number": payload.get("page_number"),
                "section": payload.get("section"),
            }
        )

    return search_results

def hybrid_search(
    query: str,
    limit: int = 5,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Perform dense + sparse retrieval and combine
    the results using Qdrant's RRF fusion.
    """

    dense_vector = embedding_service.embed_text(query)

    sparse_indices, sparse_values = (
        sparse_embedding_service.embed_text(query)
    )

    results = vector_store.hybrid_search(
        dense_vector=dense_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        limit=limit,
        document_ids=document_ids,
    )

    search_results = []

    for result in results:
        payload = result.payload or {}

        search_results.append(
            {
                "score": result.score,
                "document_id": payload.get("document_id"),
                "chunk_id": payload.get("chunk_id"),
                "filename": payload.get("filename"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "page_number": payload.get("page_number"),
                "section": payload.get("section"),
            }
        )


    return search_results


def rerank_keep_search_order(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Return the first top_k candidates in the order hybrid search found them,
    each with its reranker score ("rerank_score").

    Why this order: on our evaluation questions the hybrid order found the
    right passage more often than the reranker's order (Hit@1 72% vs 55%).
    The reranker is still useful: its score drives the "is there an answer in
    the documents?" check.

    Speed: only the passages that are returned are sent to the reranker. It
    runs on the CPU and costs about 0.3 seconds per passage, and a passage's
    score does not depend on the other passages, so scoring the rest of the
    candidates would change nothing.

    Every result also gets "retrieval_rank" (its position in hybrid search).
    """
    if not candidates:
        return []

    for position, candidate in enumerate(candidates, start=1):
        candidate.setdefault("retrieval_rank", position)

    shortlist = sorted(
        candidates,
        key=lambda result: result["retrieval_rank"],
    )[:top_k]

    scored = reranker.rerank(
        query=query,
        results=shortlist,
        top_k=len(shortlist),
    )

    scored.sort(key=lambda result: result["retrieval_rank"])

    return scored


def hybrid_search_with_reranking(
    query: str,
    candidate_limit: int = 20,
    top_k: int = 5,
) -> list[dict]:
    candidates = hybrid_search(
        query=query,
        limit=candidate_limit,
    )

    return rerank_keep_search_order(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )

def conversational_search(
    query: str,
    conversation_history: list[dict] | None = None,
    candidate_limit: int = 20,
    top_k: int = 5,
) -> list[dict]:
    from backend.app.services.query_rewriter import rewrite_query

    rewritten_query = rewrite_query(
        query=query,
        conversation_history=conversation_history,
    )

    results = hybrid_search_with_reranking(
        query=rewritten_query,
        candidate_limit=candidate_limit,
        top_k=top_k,
    )

    for result in results:
        result["original_query"] = query
        result["rewritten_query"] = rewritten_query

    return results

def search_documents(
    query: str,
    document_ids: list[str] | None = None,
    top_k: int = 5,
    use_reranker: bool = True,
    candidate_limit: int = 20,
) -> list[dict]:
    """
    Retrieval used by the /search endpoint: hybrid search, then (optionally)
    reranking. Every result remembers its position before reranking
    ("retrieval_rank") so you can see what the reranker changed.
    With the reranker on, passages keep hybrid-search order and each one
    carries its "rerank_score" (see rerank_keep_search_order).
    """
    candidates = hybrid_search(
        query=query,
        limit=max(candidate_limit, top_k),
        document_ids=document_ids,
    )

    for position, candidate in enumerate(candidates, start=1):
        candidate["retrieval_rank"] = position

    if use_reranker:
        return rerank_keep_search_order(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    return candidates[:top_k]