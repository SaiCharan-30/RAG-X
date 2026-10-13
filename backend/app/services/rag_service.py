from sqlalchemy.orm import Session

from backend.app.rag.graph.workflow import rag_graph
from backend.app.services.conversation_service import (
    add_message,
    get_conversation_history,
)


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

    # Save the new user message.
    add_message(
        db=db,
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
        citations=result["citations"],
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

    result = rag_graph.invoke(state)

    # Save the assistant response.
    add_message(
        db=db,
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
    )

    return result