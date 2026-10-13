import os

import requests

# Where the FastAPI backend is running. Change it without editing code by
# setting RAGX_BACKEND_URL before starting Streamlit, for example:
#   $env:RAGX_BACKEND_URL = "http://127.0.0.1:8000"
BACKEND_URL = os.getenv(
    "RAGX_BACKEND_URL",
    "http://127.0.0.1:8001",
).rstrip("/")


class ApiError(Exception):
    """A backend problem, with a message that is safe to show to the user."""


def _error_message(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None

    if isinstance(detail, str):
        return detail

    # FastAPI validation errors arrive as a list of {"msg": ...} items.
    if isinstance(detail, list):
        messages = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        return "; ".join(messages)

    return f"The backend returned an error ({response.status_code})."


def _request(
    method: str,
    path: str,
    timeout: int = 30,
    **kwargs,
):
    try:
        response = requests.request(
            method,
            f"{BACKEND_URL}{path}",
            timeout=timeout,
            **kwargs,
        )
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Cannot reach the backend at {BACKEND_URL}. "
            "Is the FastAPI server running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(
            "The backend took too long to respond. Please try again."
        ) from exc

    if not response.ok:
        raise ApiError(_error_message(response))

    return response.json()


# ---------------------------------------------------------------- health

def health_check() -> dict:
    return _request("GET", "/health", timeout=5)


# ------------------------------------------------------------- documents

def list_documents() -> list[dict]:
    return _request("GET", "/documents")


def upload_document(
    filename: str,
    content: bytes,
    mime_type: str | None = None,
) -> dict:
    # Indexing happens during the upload, so allow plenty of time.
    return _request(
        "POST",
        "/documents/upload",
        timeout=600,
        files={
            "file": (
                filename,
                content,
                mime_type or "application/octet-stream",
            )
        },
    )


def delete_document(document_id: str) -> dict:
    return _request("DELETE", f"/documents/{document_id}")


def summarize_document(
    document_id: str,
    summary_type: str = "quick",
) -> dict:
    return _request(
        "POST",
        f"/documents/{document_id}/summary",
        timeout=120,
        params={"summary_type": summary_type},
    )


# ---------------------------------------------------------------- search

def search(
    query: str,
    mode: str = "all",
    document_ids: list[str] | None = None,
    top_k: int = 5,
    rerank: bool = True,
) -> dict:
    return _request(
        "POST",
        "/search",
        timeout=120,
        json={
            "query": query,
            "mode": mode,
            "document_ids": document_ids,
            "top_k": top_k,
            "rerank": rerank,
        },
    )


# ------------------------------------------------------------------ chat

def list_conversations() -> list[dict]:
    return _request("GET", "/chat/conversations")


def create_conversation(title: str = "New Conversation") -> dict:
    return _request(
        "POST",
        "/chat/conversations",
        json={"title": title[:255] or "New Conversation"},
    )


def get_messages(conversation_id: str) -> dict:
    return _request(
        "GET",
        f"/chat/conversations/{conversation_id}/messages",
    )


def delete_conversation(conversation_id: str) -> dict:
    return _request(
        "DELETE",
        f"/chat/conversations/{conversation_id}",
    )


def chat(
    conversation_id: str,
    query: str,
    mode: str = "all",
    document_ids: list[str] | None = None,
) -> dict:
    return _request(
        "POST",
        f"/chat/conversations/{conversation_id}/chat",
        timeout=180,
        json={
            "query": query,
            "mode": mode,
            "document_ids": document_ids,
        },
    )