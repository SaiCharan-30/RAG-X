from backend.app.database.connection import SessionLocal
from backend.app.services.conversation_service import (
    create_conversation,
    delete_conversation,
)
from backend.app.services.rag_service import ask_question


db = SessionLocal()

try:
    conversation = create_conversation(
        db=db,
        title="RAG-X Memory Test",
    )

    conversation_id = conversation.conversation_id

    print("\n=== CONVERSATION ===")
    print("ID:", conversation_id)

    print("\n=== QUESTION 1 ===")

    result1 = ask_question(
        db=db,
        conversation_id=conversation_id,
        query="What is the proposed architecture of RAG-X?",
    )

    print("Original:", result1["original_query"])
    print("Rewritten:", result1["rewritten_query"])
    print("Answer:", result1["answer"])

    print("\n=== QUESTION 2 ===")

    result2 = ask_question(
        db=db,
        conversation_id=conversation_id,
        query="What are its advantages?",
    )

    print("Original:", result2["original_query"])
    print("Rewritten:", result2["rewritten_query"])
    print("Answer:", result2["answer"])

    print("\n=== MEMORY TEST ===")

    print(
        "Question 2 rewritten query:",
        result2["rewritten_query"],
    )

    print(
        "Conversation history used:",
        len(result2["conversation_history"]),
        "messages",
    )

finally:
    delete_conversation(
        db=db,
        conversation_id=conversation_id,
    )

    db.close()