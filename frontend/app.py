import streamlit as st

from utils.api_client import ApiError
from utils.components import load_overview


st.set_page_config(
    page_title="RAG-X",
    page_icon="📚",
    layout="wide",
)


st.title("📚 RAG-X")

st.subheader("Advanced Multi-Document AI Knowledge Assistant")

st.write(
    "Welcome to RAG-X — an AI-powered knowledge assistant "
    "for searching, understanding, and interacting with your documents."
)

st.divider()


try:
    overview = load_overview()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Indexed documents", overview["indexed"])

    with col2:
        st.metric("Conversations", len(overview["conversations"]))

    with col3:
        st.metric("Questions asked", overview["questions"])

except ApiError as error:
    st.error(str(error))


st.divider()

st.header("🚀 Getting Started")

st.write(
    """
    Use the sidebar to navigate through RAG-X.

    **Documents** → Upload and manage your documents.

    **Search** → Search your knowledge base.

    **Chat** → Ask questions about your documents.

    **Evaluation** → Evaluate RAG performance.

    **Settings** → See how the app is configured.
    """
)