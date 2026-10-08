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


def hybrid_search_with_reranking(
    query: str,
    candidate_limit: int = 20,
    top_k: int = 5,
) -> list[dict]:
    candidates = hybrid_search(
        query=query,
        limit=candidate_limit,
    )

    reranked_results = reranker.rerank(
        query=query,
        results=candidates,
        top_k=top_k,
    )

    return reranked_results

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