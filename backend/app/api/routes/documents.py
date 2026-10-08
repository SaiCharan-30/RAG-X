from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.models.document import Document
from backend.app.schemas.document import DocumentUploadResponse

from backend.app.schemas.document import DocumentListItem
from backend.app.models.chunk import DocumentChunk
from backend.app.services.vector_store import vector_store
from backend.app.services.document_intelligence import summarize_document
from backend.app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentSummaryResponse,
    DocumentUploadResponse,
    SummaryType,
)

router = APIRouter()


UPLOAD_DIR = Path("data/uploads")


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".html",
}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # -----------------------------------------
    # 1. Validate filename
    # -----------------------------------------
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # -----------------------------------------
    # 2. Get file extension
    # -----------------------------------------
    extension = Path(file.filename).suffix.lower()

    # -----------------------------------------
    # 3. Validate file type
    # -----------------------------------------
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}",
        )

    # -----------------------------------------
    # 4. Generate unique document ID
    # -----------------------------------------
    document_id = str(uuid4())

    # -----------------------------------------
    # 5. Sanitize original filename
    # -----------------------------------------
    safe_filename = Path(file.filename).name

    # -----------------------------------------
    # 6. Generate stored filename
    # -----------------------------------------
    stored_filename = f"{document_id}{extension}"

    file_path = UPLOAD_DIR / stored_filename

    # -----------------------------------------
    # 7. Create upload directory
    # -----------------------------------------
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # 8. Read uploaded file
    # -----------------------------------------
    contents = await file.read()

    # -----------------------------------------
    # 9. Save physical file
    # -----------------------------------------
    file_path.write_bytes(contents)

    uploaded_at = datetime.now(timezone.utc)

    # -----------------------------------------
    # 10. Create database record
    # -----------------------------------------
    document = Document(
        document_id=document_id,
        original_filename=safe_filename,
        stored_filename=stored_filename,
        file_type=extension,
        size_bytes=len(contents),
        file_path=str(file_path),
        status="uploaded",
        uploaded_at=uploaded_at,
    )

    # -----------------------------------------
    # 11. Save metadata to PostgreSQL
    # -----------------------------------------
    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()

        # Remove file if database operation failed
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Failed to save document metadata.",
        )

    # -----------------------------------------
    # 12. Return API response
    # -----------------------------------------
    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        document_id=document_id,
        original_filename=safe_filename,
        stored_filename=stored_filename,
        file_type=extension,
        size_bytes=len(contents),
        uploaded_at=uploaded_at,
    )










@router.get(
    "",
    response_model=list[DocumentListItem],
)
def list_documents(
    db: Session = Depends(get_db),
):
    documents = (
        db.query(Document)
        .order_by(Document.uploaded_at.desc())
        .all()
    )

    return documents


@router.post(
    "/{document_id}/summary",
    response_model=DocumentSummaryResponse,
)
def summarize_document_endpoint(
    document_id: str,
    summary_type: SummaryType = "quick",
    db: Session = Depends(get_db),
):
    try:
        return summarize_document(
            document_id=document_id,
            db=db,
            summary_type=summary_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc



@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    document = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    file_path = Path(document.file_path)

    try:
        # 1. Remove vectors from Qdrant.
        vector_store.delete_by_document_id(document_id)

        # 2. Remove chunks from PostgreSQL.
        (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .delete(synchronize_session=False)
        )

        # 3. Remove document metadata from PostgreSQL.
        db.delete(document)
        db.commit()

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document data: {exc}",
        ) from exc

    # 4. Remove the original file.
    try:
        if file_path.exists():
            file_path.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document data was deleted, but the original file could not be removed: {exc}",
        ) from exc

    return DocumentDeleteResponse(
        message="Document and all associated data deleted successfully.",
        document_id=document_id,
    )