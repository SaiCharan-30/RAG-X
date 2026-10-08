import streamlit as st


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


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents", "0")

with col2:
    st.metric("Conversations", "0")

with col3:
    st.metric("Questions Asked", "0")


st.divider()

st.header("🚀 Getting Started")

st.write(
    """
    Use the sidebar to navigate through RAG-X.

    **Documents** → Upload and manage your documents.

    **Search** → Search your knowledge base.

    **Chat** → Ask questions about your documents.

    **Evaluation** → Evaluate RAG performance.

    **Settings** → Configure application settings.
    """
)