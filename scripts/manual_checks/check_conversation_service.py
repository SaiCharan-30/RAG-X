from backend.app.database.connection import SessionLocal
from backend.app.services.conversation_service import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation_history,
)


db = SessionLocal()

try:
    conversation = create_conversation(
        db=db,
        title="RAG-X Test Conversation",
    )

    print("\n=== CONVERSATION CREATED ===")
    print("Conversation ID:", conversation.conversation_id)
    print("Title:", conversation.title)

    add_message(
        db=db,
        conversation_id=conversation.conversation_id,
        role="user",
        content="What is the proposed architecture of RAG-X?",
    )

    add_message(
        db=db,
        conversation_id=conversation.conversation_id,
        role="assistant",
        content=(
            "RAG-X uses hybrid retrieval followed "
            "by reranking."
        ),
    )

    add_message(
        db=db,
        conversation_id=conversation.conversation_id,
        role="user",
        content="What are its advantages?",
    )

    history = get_conversation_history(
        db=db,
        conversation_id=conversation.conversation_id,
    )

    print("\n=== CONVERSATION HISTORY ===")

    for message in history:
        print(
            f"{message['role']}: "
            f"{message['content']}"
        )

    deleted = delete_conversation(
        db=db,
        conversation_id=conversation.conversation_id,
    )

    print("\n=== DELETE RESULT ===")
    print("Deleted:", deleted)

finally:
    db.close()