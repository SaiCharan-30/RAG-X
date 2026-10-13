import logging

from sqlalchemy.orm import Session

from backend.app.rag.graph.workflow import rag_graph
from backend.app.services.conversation_service import (
    add_message,
    get_conversation_history,
)
from backend.app.services.errors import AIServiceError

logger = logging.getLogger(__name__)


def ask_question(
    db: Session,
    conversation_id: str,
    query: str,
    document_ids: list[str] | None = None,
) -> dict:
    # Get previous messages from PostgreSQL.
    conversation_history = get_conversation_history(
        db=db,
        conversation_id=conversation_id,
    )

    # Run the RAG workflow.
    state = {
        "original_query": query,
        "rewritten_query": "",
        "conversation_history": conversation_history,

        "document_ids": document_ids,

        "retrieved_chunks": [],
        "reranked_chunks": [],
        "context_quality": 0.0,
        "answer": "",
        "grounded": False,
        "grounding_score": 0.0,
        "citations": [],
        "retry_count": 0,
    }

    try:
        result = rag_graph.invoke(state)

    except Exception as exc:
        # Full technical details go to the server log only.
        logger.exception("RAG workflow failed")

        error_text = str(exc)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            raise AIServiceError(
                "The AI service has reached its usage limit. "
                "Please try again later."
            ) from exc

        raise AIServiceError(
            "The question could not be answered because an AI service "
            "failed. Please try again."
        ) from exc

    # Save both messages only after the answer succeeded, so a failed
    # question does not leave a half-finished conversation behind.
    add_message(
        db=db,
        conversation_id=conversation_id,
        role="user",
        content=query,
    )

    add_message(
        db=db,
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
        citations=result["citations"],
    )

    return result