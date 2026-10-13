from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.models.document import Document


def ensure_documents_ready(
    db: Session,
    document_ids: list[str] | None,
) -> None:
    """
    Used by /search and /chat when the user picks specific documents.
    404 if an id does not exist, 400 if a document is not indexed yet.
    Does nothing when document_ids is empty or None (all documents).
    """
    if not document_ids:
        return

    statuses = {
        document.document_id: document.status
        for document in db.query(Document)
        .filter(Document.document_id.in_(document_ids))
        .all()
    }

    unknown = [
        document_id
        for document_id in document_ids
        if document_id not in statuses
    ]

    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"Document(s) not found: {', '.join(unknown)}",
        )

    not_ready = [
        f"{document_id} (status: {statuses[document_id]})"
        for document_id in document_ids
        if statuses[document_id] != "indexed"
    ]

    if not_ready:
        raise HTTPException(
            status_code=400,
            detail=f"Document(s) not indexed yet: {', '.join(not_ready)}",
        )