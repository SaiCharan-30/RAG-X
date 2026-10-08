from datetime import datetime

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    created_at: datetime