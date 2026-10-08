import streamlit as st


st.title("⚙️ Settings")

st.write(
    "Configure RAG-X application settings."
)

st.divider()


st.subheader("Application")

st.text_input(
    "Backend URL",
    value="http://127.0.0.1:8000",
    disabled=True,
)


st.subheader("RAG Configuration")

st.selectbox(
    "Retrieval Strategy",
    [
        "Hybrid Search",
        "Dense Search",
        "Sparse Search",
    ],
)


st.slider(
    "Top K Documents",
    min_value=1,
    max_value=20,
    value=5,
)


st.info(
    "Advanced configuration will be connected to the backend later."
)