from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    test_version: str = "NEW_VERSION"
    message: str
    document_id: str
    original_filename: str
    stored_filename: str
    file_type: str
    size_bytes: int
    uploaded_at: datetime
    
    
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas.document import DocumentUploadResponse


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
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}",
        )

    document_id = str(uuid4())

    safe_filename = Path(file.filename).name

    stored_filename = f"{document_id}{extension}"

    file_path = UPLOAD_DIR / stored_filename

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contents = await file.read()

    file_path.write_bytes(contents)

    uploaded_at = datetime.now(timezone.utc)

    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        document_id=document_id,
        original_filename=safe_filename,
        stored_filename=stored_filename,
        file_type=extension,
        size_bytes=len(contents),
        uploaded_at=uploaded_at,
    )


class DocumentListItem(BaseModel):
    document_id: str
    original_filename: str
    file_type: str
    size_bytes: int
    status: str
    uploaded_at: datetime


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str    

SummaryType = Literal[
    "quick",
    "detailed",
    "key_points",
    "methodology",
    "findings",
    "limitations",
    "keywords",
]   

class DocumentSummaryResponse(BaseModel):
    document_id: str
    summary_type: SummaryType
    summary: str
    source_chunks: int









