import streamlit as st

from utils import api_client
from utils.api_client import ApiError
from utils.components import document_scope_selector, show_citations


st.title("💬 Chat")

st.write(
    "Ask questions about your documents using conversational RAG."
)

st.divider()


# conversation_id is None while the user is on a brand-new, unsaved chat.
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def start_new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.chat_messages = []


def open_conversation(conversation_id: str) -> None:
    try:
        data = api_client.get_messages(conversation_id)
    except ApiError as error:
        st.session_state.sidebar_error = str(error)
        return

    st.session_state.conversation_id = conversation_id
    st.session_state.chat_messages = [
        {
            "role": message["role"],
            "content": message["content"],
            "citations": message["citations"],
        }
        for message in data["messages"]
    ]


def remove_conversation(conversation_id: str) -> None:
    try:
        api_client.delete_conversation(conversation_id)
    except ApiError as error:
        st.session_state.sidebar_error = str(error)
        return

    if st.session_state.conversation_id == conversation_id:
        start_new_chat()


# --------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Documents")
    mode, document_ids, scope_ok = document_scope_selector("chat")

    st.divider()
    st.subheader("Conversations")

    st.button(
        "➕ New chat",
        on_click=start_new_chat,
        width="stretch",
    )

    if "sidebar_error" in st.session_state:
        st.error(st.session_state.pop("sidebar_error"))

    try:
        conversations = [
            c for c in api_client.list_conversations()
            if c["message_count"] > 0
        ]
    except ApiError as error:
        st.error(str(error))
        conversations = []

    for conversation in conversations[:15]:
        is_open = (
            conversation["conversation_id"]
            == st.session_state.conversation_id
        )

        open_col, delete_col = st.columns([4, 1])

        open_col.button(
            conversation["title"][:40],
            key=f"open_{conversation['conversation_id']}",
            on_click=open_conversation,
            args=(conversation["conversation_id"],),
            type="primary" if is_open else "secondary",
            width="stretch",
        )

        delete_col.button(
            "🗑️",
            key=f"del_{conversation['conversation_id']}",
            on_click=remove_conversation,
            args=(conversation["conversation_id"],),
        )


# -------------------------------------------------------------- messages

def show_answer_notes(message: dict) -> None:
    # Only fresh answers carry these numbers; reopened chats do not.
    if message.get("no_answer"):
        st.caption("No relevant passages were found in the selected documents.")
    elif message.get("grounded") is False:
        st.warning(
            "This answer may not be fully supported by the documents. "
            "Please check the sources."
        )

    if message.get("rewritten_query"):
        st.caption(f"Searched for: {message['rewritten_query']}")


for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            show_answer_notes(message)
            show_citations(message.get("citations"))


# ----------------------------------------------------------------- input

prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    if not scope_ok:
        st.warning(
            "Choose the documents to use in the sidebar first "
            "(and make sure at least one is indexed)."
        )
        st.stop()

    with st.chat_message("user"):
        st.write(prompt)

    created_here = False

    try:
        if st.session_state.conversation_id is None:
            conversation = api_client.create_conversation(prompt[:50])
            st.session_state.conversation_id = conversation["conversation_id"]
            created_here = True

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents ..."):
                result = api_client.chat(
                    st.session_state.conversation_id,
                    prompt,
                    mode,
                    document_ids,
                )

    except ApiError as error:
        # Nothing was saved for this question, so do not keep an empty chat.
        if created_here:
            try:
                api_client.delete_conversation(
                    st.session_state.conversation_id
                )
            except ApiError:
                pass
            st.session_state.conversation_id = None

        st.error(str(error))
        st.stop()

    st.session_state.chat_messages.append(
        {"role": "user", "content": prompt, "citations": None}
    )
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": [c for c in result["citations"]],
            "grounded": result["grounded"],
            "no_answer": (
                not result["citations"]
                and result["context_quality"] < 0.05
            ),
            "rewritten_query": (
                result["rewritten_query"]
                if result["rewritten_query"] != result["original_query"]
                else None
            ),
        }
    )

    # Redraw so the new messages are shown by the loop above.
    st.rerun()