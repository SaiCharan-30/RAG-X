import streamlit as st


st.title("📊 Dashboard")

st.write("Overview of your RAG-X knowledge base.")

st.divider()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Documents", "0")

with col2:
    st.metric("Indexed Chunks", "0")

with col3:
    st.metric("Questions", "0")

with col4:
    st.metric("Conversations", "0")


st.divider()

st.subheader("Recent Activity")

st.info("No activity yet. Upload your first document to get started.")