from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models.conversation import Conversation
from backend.app.models.message import Message


def create_conversation(
    db: Session,
    title: str = "New Conversation",
) -> Conversation:
    now = datetime.now(timezone.utc)

    conversation = Conversation(
        conversation_id=str(uuid4()),
        title=title,
        created_at=now,
        updated_at=now,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


def get_conversation(
    db: Session,
    conversation_id: str,
) -> Conversation | None:
    return (
        db.query(Conversation)
        .filter(
            Conversation.conversation_id == conversation_id
        )
        .first()
    )


def list_conversations(
    db: Session,
) -> list[tuple[Conversation, int]]:
    """
    Every conversation with its message count, most recently active first.
    """
    return (
        db.query(Conversation, func.count(Message.id))
        .outerjoin(
            Message,
            Message.conversation_id == Conversation.conversation_id,
        )
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    citations: list[dict] | None = None,
) -> Message:
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.conversation_id == conversation_id
        )
        .first()
    )

    if conversation is None:
        raise ValueError(
            f"Conversation not found: {conversation_id}"
        )

    message = Message(
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        created_at=datetime.now(timezone.utc),
    )

    db.add(message)

    conversation.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(message)

    return message


def get_messages(
    db: Session,
    conversation_id: str,
) -> list[Message]:
    return (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at.asc())
        .all()
    )


def get_conversation_history(
    db: Session,
    conversation_id: str,
) -> list[dict]:
    messages = get_messages(
        db=db,
        conversation_id=conversation_id,
    )

    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]


def delete_conversation(
    db: Session,
    conversation_id: str,
) -> bool:
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.conversation_id == conversation_id
        )
        .first()
    )

    if conversation is None:
        return False

    db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).delete(
        synchronize_session=False
    )

    db.delete(conversation)
    db.commit()

    return True