from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.validators import ensure_documents_ready
from backend.app.database.connection import get_db
from backend.app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from backend.app.services.search import search_documents

router = APIRouter()


@router.post(
    "",
    response_model=SearchResponse,
)
def search_endpoint(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    # This is a normal "def" (not "async def"), so FastAPI runs it in a
    # worker thread and the server stays responsive while the AI models work.

    # 1. Check the chosen documents (single / selected mode)
    ensure_documents_ready(db, request.document_ids)

    # 2. Retrieve (and optionally rerank)
    found = search_documents(
        query=request.query,
        document_ids=request.document_ids,
        top_k=request.top_k,
        use_reranker=request.rerank,
    )

    # 3. Build the response
    results = [
        SearchResult(
            rank=position,
            retrieval_rank=item["retrieval_rank"],
            retrieval_score=item["score"],
            rerank_score=item.get("rerank_score"),
            document_id=item.get("document_id"),
            chunk_id=item.get("chunk_id"),
            filename=item.get("filename"),
            page_number=item.get("page_number"),
            section=item.get("section"),
            chunk_index=item.get("chunk_index"),
            text=item.get("text") or "",
        )
        for position, item in enumerate(found, start=1)
    ]

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        reranked=request.rerank,
        result_count=len(results),
        results=results,
    )