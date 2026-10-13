import streamlit as st

from utils import api_client
from utils.api_client import ApiError

STATUS_LABELS = {
    "indexed": "✅ Indexed",
    "failed": "❌ Failed",
    "processing": "⏳ Processing",
    "uploaded": "⏳ Uploaded",
}

MODE_OPTIONS = {
    "All documents": "all",
    "One document": "single",
    "Selected documents": "selected",
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def document_label(document: dict) -> str:
    # The short id keeps two files with the same name apart.
    return f"{document['original_filename']} ({document['document_id'][:8]})"


def document_scope_selector(key: str) -> tuple[str, list[str] | None, bool]:
    """
    Lets the user choose which documents to use.

    Returns (mode, document_ids, is_valid). is_valid is False when there is
    nothing to search, or the user chose "One" or "Selected" but has not
    picked a document yet.
    """
    try:
        documents = [
            document
            for document in api_client.list_documents()
            if document["status"] == "indexed"
        ]
    except ApiError as error:
        st.error(str(error))
        return "all", None, False

    if not documents:
        st.info("No indexed documents yet. Upload one on the Documents page.")
        return "all", None, False

    choice = st.radio(
        "Documents to use",
        list(MODE_OPTIONS),
        key=f"{key}_mode",
    )
    mode = MODE_OPTIONS[choice]

    labels = {
        document_label(document): document["document_id"]
        for document in documents
    }

    if mode == "all":
        st.caption(f"Using all {len(documents)} indexed document(s).")
        return "all", None, True

    if mode == "single":
        picked = st.selectbox(
            "Document",
            list(labels),
            key=f"{key}_single",
        )
        return "single", [labels[picked]], True

    picked = st.multiselect(
        "Documents",
        list(labels),
        key=f"{key}_selected",
    )

    if not picked:
        st.warning("Pick at least one document.")
        return "selected", None, False

    return "selected", [labels[name] for name in picked], True


def load_overview() -> dict:
    """Numbers for the home page and dashboard. Raises ApiError if the backend is down."""
    documents = api_client.list_documents()
    conversations = api_client.list_conversations()

    return {
        "documents": documents,
        "conversations": conversations,
        "indexed": sum(1 for d in documents if d["status"] == "indexed"),
        "failed": sum(1 for d in documents if d["status"] == "failed"),
        # Every question is one user message plus one assistant message.
        "questions": sum(c["message_count"] for c in conversations) // 2,
    }


def show_citations(citations: list[dict] | None) -> None:
    if not citations:
        return

    with st.expander(f"📎 Sources ({len(citations)})"):
        for citation in citations:
            place = []

            if citation.get("page_number") is not None:
                place.append(f"page {citation['page_number']}")

            if citation.get("section"):
                place.append(citation["section"])

            suffix = f" — {', '.join(place)}" if place else ""

            st.markdown(
                f"**[{citation['citation_id']}] "
                f"{citation.get('filename') or 'Unknown file'}**{suffix}"
            )

            if citation.get("snippet"):
                st.caption(citation["snippet"])