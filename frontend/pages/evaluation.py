import streamlit as st


st.title("📈 Evaluation")

st.write(
    "Evaluate retrieval and generation quality of the RAG system."
)

st.divider()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Hit Rate", "—")

with col2:
    st.metric("MRR", "—")

with col3:
    st.metric("Faithfulness", "—")


st.info(
    "Evaluation metrics will be connected after the RAG pipeline is implemented."
)