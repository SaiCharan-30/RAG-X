from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.validators import ensure_documents_ready
from backend.app.database.connection import get_db
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    ConversationMessagesResponse,
    ConversationSummary,
    CreateConversationRequest,
    CreateConversationResponse,
    DeleteConversationResponse,
    MessageResponse,
)
from backend.app.services.conversation_service import (
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    list_conversations,
)
from backend.app.services.errors import AIServiceError
from backend.app.services.rag_service import ask_question

router = APIRouter()


def _get_conversation_or_404(
    db: Session,
    conversation_id: str,
):
    conversation = get_conversation(
        db=db,
        conversation_id=conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    return conversation


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


@router.get(
    "/conversations",
    response_model=list[ConversationSummary],
)
def list_chat_conversations(
    db: Session = Depends(get_db),
):
    return [
        ConversationSummary(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            message_count=message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
        for conversation, message_count in list_conversations(db=db)
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    conversation = _get_conversation_or_404(db, conversation_id)

    messages = get_messages(
        db=db,
        conversation_id=conversation_id,
    )

    return ConversationMessagesResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        messages=[
            MessageResponse(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                citations=message.citations,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteConversationResponse,
)
def delete_chat_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    deleted = delete_conversation(
        db=db,
        conversation_id=conversation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    return DeleteConversationResponse(
        message="Conversation deleted successfully.",
        conversation_id=conversation_id,
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
    _get_conversation_or_404(db, conversation_id)

    ensure_documents_ready(db, request.document_ids)

    try:
        result = ask_question(
            db=db,
            conversation_id=conversation_id,
            query=request.query,
            document_ids=request.document_ids,
        )

    except AIServiceError as exc:
        raise HTTPException(
            status_code=503,
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