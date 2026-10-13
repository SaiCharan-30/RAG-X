import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from uuid import NAMESPACE_URL, uuid5
from sqlalchemy.orm import Session

from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document
from backend.app.services.document_parser import parse_document
from backend.app.services.chunker import create_chunks
from backend.app.services.embeddings import embedding_service
from backend.app.services.sparse_embeddings import sparse_embedding_service
from backend.app.services.vector_store import vector_store


logger = logging.getLogger(__name__)


def index_document(
    file_path: Path,
    document_id: str,
    db: Session,
    original_filename: str | None = None,
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
                # The name people recognise (used later for citations).
                "filename": original_filename or file_path.name,
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


def _remove_partial_index(document_id: str, db: Session) -> None:
    """
    Undo a half-finished indexing run so that no orphan data is left
    in Qdrant or PostgreSQL. Never raises: the original error matters more.
    """
    try:
        vector_store.delete_by_document_id(document_id)
    except Exception:
        logger.warning(
            "Could not remove Qdrant vectors for %s after a failure.",
            document_id,
            exc_info=True,
        )

    try:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
        logger.warning(
            "Could not remove PostgreSQL chunks for %s after a failure.",
            document_id,
            exc_info=True,
        )


def index_uploaded_document(
    document: Document,
    db: Session,
) -> tuple[int, str | None]:
    """
    Index one uploaded document and keep its status up to date:

        uploaded -> processing -> indexed   (success)
                              \\-> failed    (any error, partial data removed)

    Returns (number_of_chunks, error_message). error_message is None on success.
    """
    document_id = document.document_id

    try:
        document.status = "processing"
        db.commit()

        chunk_count = index_document(
            file_path=Path(document.file_path),
            document_id=document_id,
            db=db,
            original_filename=document.original_filename,
        )

        if chunk_count == 0:
            raise ValueError(
                "No readable text could be extracted from this document."
            )

        document.status = "indexed"
        db.commit()

        return chunk_count, None

    except Exception as exc:
        db.rollback()

        logger.exception("Indexing failed for document %s", document_id)

        _remove_partial_index(document_id, db)

        logger.exception("Indexing failed for document %s", document_id)

        # Our own ValueError messages are written for users (for example
        # "No readable text..."). Anything else may contain file paths or
        # library internals, so show a generic message instead.
        if isinstance(exc, ValueError) and str(exc).strip():
            error_message = str(exc).strip()[:300]
        else:
            error_message = (
                "Could not read this file. It may be corrupted "
                "or in an unsupported format."
            )

        try:
            document.status = "failed"
            db.commit()
        except Exception:
            db.rollback()
            logger.warning(
                "Could not mark document %s as failed.",
                document_id,
                exc_info=True,
            )

        return 0, error_message
