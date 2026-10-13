import streamlit as st

from utils import api_client
from utils.api_client import ApiError


st.title("⚙️ Settings")

st.write(
    "How RAG-X is currently configured. These values live in the "
    "backend, so they are shown here for reference only."
)

st.divider()


st.subheader("Application")

st.text_input(
    "Backend URL",
    value=api_client.BACKEND_URL,
    disabled=True,
    help="Set the RAGX_BACKEND_URL environment variable before starting "
    "Streamlit to change it.",
)

try:
    api_client.health_check()
    st.success("Backend is reachable and healthy.")
except ApiError as error:
    st.error(str(error))
except Exception:
    st.error("The backend responded, but not as expected.")


st.subheader("RAG pipeline")

st.markdown(
    """
    - **Chunking:** about 1000 characters with 150 overlap
    - **Retrieval:** hybrid search (dense bge-m3 + sparse BM42), top 20 candidates
    - **Reranking:** bge-reranker-base, best 5 passages are used
    - **No-answer rule:** if the best passage scores below 0.05 after one retry,
      the assistant says it could not find the answer instead of guessing
    - **Answers:** written by Gemini from the retrieved passages only,
      with citations for the sources it used
    """
)