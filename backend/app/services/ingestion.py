from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.services.chunker import create_chunks
from backend.app.services.document_parser import parse_document


def process_document(
    file_path: Path,
    document_id: str,
) -> list[dict]:
    """
    Parse a document and create metadata-aware chunks.
    """

    pages = parse_document(file_path)
    chunks = create_chunks(pages)

    processed_chunks = []

    for index, chunk in enumerate(chunks):
        processed_chunks.append(
            {
                # Application-level deterministic ID
                "chunk_id": f"{document_id}-chunk-{index}",

                # Qdrant point ID
                "point_id": str(uuid4()),

                "document_id": document_id,

                # Original filename
                "filename": file_path.name,

                "chunk_index": index,
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "section": chunk["section"],
                "created_at": datetime.now(timezone.utc),
            }
        )

    return processed_chunks