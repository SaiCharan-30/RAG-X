from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    original_filename: str
    stored_filename: str
    file_type: str
    size_bytes: int
    uploaded_at: datetime


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
