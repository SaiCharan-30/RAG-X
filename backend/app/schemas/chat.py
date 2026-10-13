from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from backend.app.schemas.document_mode import DocumentMode


class CreateConversationRequest(BaseModel):
    title: str = Field(
        default="New Conversation",
        min_length=1,
        max_length=255,
    )


class CreateConversationResponse(BaseModel):
    conversation_id: str
    title: str


class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=10000,
    )

    mode: DocumentMode = DocumentMode.ALL

    document_ids: list[str] | None = None

    @model_validator(mode="after")
    def validate_document_scope(self):
        if self.mode == DocumentMode.SINGLE:
            if not self.document_ids:
                raise ValueError(
                    "single mode requires exactly one document_id."
                )

            if len(self.document_ids) != 1:
                raise ValueError(
                    "single mode requires exactly one document_id."
                )

        elif self.mode == DocumentMode.SELECTED:
            if not self.document_ids:
                raise ValueError(
                    "selected mode requires at least one document_id."
                )

        elif self.mode == DocumentMode.ALL:
            self.document_ids = None

        return self


class CitationResponse(BaseModel):
    citation_id: int
    document_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    section: str | None = None
    chunk_index: int | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    original_query: str
    rewritten_query: str
    answer: str
    grounded: bool
    grounding_score: float
    context_quality: float
    citations: list[CitationResponse]


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    citations: list[CitationResponse] | None = None
    created_at: datetime


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    title: str
    messages: list[MessageResponse]


class DeleteConversationResponse(BaseModel):
    message: str
    conversation_id: str