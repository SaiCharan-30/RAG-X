from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    CreateConversationRequest,
    CreateConversationResponse,
)
from backend.app.services.conversation_service import (
    create_conversation,
)
from backend.app.services.rag_service import ask_question

router = APIRouter()


@router.post(
    "/conversations",
    response_model=CreateConversationResponse,
)
def create_chat_conversation(
    request: CreateConversationRequest,
    db: Session = Depends(get_db),
):
    conversation = create_conversation(
        db=db,
        title=request.title,
    )

    return CreateConversationResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
    )


@router.post(
    "/conversations/{conversation_id}/chat",
    response_model=ChatResponse,
)
def chat(
    conversation_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    try:
        result = ask_question(
            db=db,
            conversation_id=conversation_id,
            query=request.query,
            document_ids=request.document_ids,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    citations = [
        CitationResponse(**citation)
        for citation in result["citations"]
    ]

    return ChatResponse(
        conversation_id=conversation_id,
        original_query=result["original_query"],
        rewritten_query=result["rewritten_query"],
        answer=result["answer"],
        grounded=result["grounded"],
        grounding_score=result["grounding_score"],
        context_quality=result["context_quality"],
        citations=citations,
    )