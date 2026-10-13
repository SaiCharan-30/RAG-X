from sqlalchemy import text

from backend.app.database.base import Base
from backend.app.database.connection import engine

from backend.app.models.chunk import DocumentChunk
from backend.app.models.conversation import Conversation
from backend.app.models.document import Document
from backend.app.models.message import Message


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    # create_all() never changes a table that already exists, so a
    # database created before citations were saved needs this column
    # added once. "IF NOT EXISTS" makes it safe to run on every start.
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE messages "
                "ADD COLUMN IF NOT EXISTS citations JSON"
            )
        )


if __name__ == "__main__":
    init_db()
    print("Database tables created successfully.")