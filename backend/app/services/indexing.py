from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from uuid import NAMESPACE_URL, uuid5
from sqlalchemy.orm import Session

from backend.app.models.chunk import DocumentChunk
from backend.app.services.document_parser import parse_document
from backend.app.services.chunker import create_chunks
from backend.app.services.embeddings import embedding_service
from backend.app.services.sparse_embeddings import sparse_embedding_service
from backend.app.services.vector_store import vector_store


def index_document(
    file_path: Path,
    document_id: str,
    db: Session,
) -> int:
    pages = parse_document(file_path)
    chunks = create_chunks(pages)

    if not chunks:
        return 0

    processed_chunks = []

    for index, chunk in enumerate(chunks):
        processed_chunks.append(
            {
                "chunk_id": f"{document_id}-chunk-{index}",
                "point_id": str(uuid5(NAMESPACE_URL, f"rag-x:{document_id}:{index}")),
                "document_id": document_id,
                "filename": file_path.name,
                "chunk_index": index,
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "section": chunk["section"],
                "created_at": datetime.now(timezone.utc),
            }
        )
    # Remove existing PostgreSQL chunks for this document.
    db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).delete(synchronize_session=False)

    db.commit()
    texts = [chunk["text"] for chunk in processed_chunks]

    dense_vectors = embedding_service.embed_texts(texts)

    sparse_vectors = [
        sparse_embedding_service.embed_text(text)
        for text in texts
    ]

    vector_store.create_collection()

    vector_store.add_chunks(
        processed_chunks,
        dense_vectors,
        sparse_vectors,
    )

    # Save chunk metadata in PostgreSQL.
    for chunk in processed_chunks:
        db_chunk = DocumentChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            page_number=chunk["page_number"],
            section=chunk["section"],
            created_at=chunk["created_at"],
        )

        db.add(db_chunk)

    db.commit()

    return len(processed_chunks)