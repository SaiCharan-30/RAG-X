from uuid import uuid4

from qdrant_client.models import PointIdsList

from backend.app.services.embeddings import embedding_service
from backend.app.services.sparse_embeddings import (
    sparse_embedding_service,
)
from backend.app.services.vector_store import vector_store


DOCUMENT_A = "isolation-test-document-A"
DOCUMENT_B = "isolation-test-document-B"


def build_chunk(
    document_id: str,
    chunk_id: str,
    text: str,
    index: int,
) -> dict:
    return {
        "point_id": str(uuid4()),
        "document_id": document_id,
        "chunk_id": chunk_id,
        "filename": f"{document_id}.txt",
        "chunk_index": index,
        "text": text,
        "page_number": None,
        "section": None,
    }


chunks = [
    build_chunk(
        DOCUMENT_A,
        "chunk-A-1",
        "RAG-X uses hybrid retrieval with dense and sparse search.",
        0,
    ),
    build_chunk(
        DOCUMENT_A,
        "chunk-A-2",
        "RAG-X uses reranking to improve retrieved context.",
        1,
    ),
    build_chunk(
        DOCUMENT_B,
        "chunk-B-1",
        "PostgreSQL stores application and conversation data.",
        0,
    ),
    build_chunk(
        DOCUMENT_B,
        "chunk-B-2",
        "Qdrant stores vector representations for retrieval.",
        1,
    ),
]

point_ids = [chunk["point_id"] for chunk in chunks]

try:
    texts = [chunk["text"] for chunk in chunks]

    dense_vectors = embedding_service.embed_texts(texts)

    sparse_vectors = [
        sparse_embedding_service.embed_text(text)
        for text in texts
    ]

    vector_store.add_chunks(
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )

    query = "How does RAG-X retrieve information?"

    dense_vector = embedding_service.embed_text(query)

    sparse_indices, sparse_values = (
        sparse_embedding_service.embed_text(query)
    )

    results = vector_store.hybrid_search(
        dense_vector=dense_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        limit=10,
        document_ids=[DOCUMENT_A],
    )

    print("\n=== DOCUMENT ISOLATION TEST ===")
    print("Results:", len(results))

    for result in results:
        payload = result.payload or {}

        print(
            "Document:",
            payload.get("document_id"),
            "| Chunk:",
            payload.get("chunk_id"),
        )

        assert payload.get("document_id") == DOCUMENT_A

    print("\nIsolation check: PASSED")

finally:
    vector_store.client.delete(
        collection_name=vector_store.collection_name,
        points_selector=PointIdsList(
            points=point_ids,
        ),
    )

    print("Temporary test data deleted.")