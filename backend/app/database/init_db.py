from backend.app.database.base import Base
from backend.app.database.connection import engine

from backend.app.models.chunk import DocumentChunk
from backend.app.models.conversation import Conversation
from backend.app.models.document import Document
from backend.app.models.message import Message


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created successfully.")