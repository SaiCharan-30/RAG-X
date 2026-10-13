import pandas as pd
import streamlit as st

from utils.api_client import ApiError
from utils.components import load_overview, status_label


st.title("📊 Dashboard")

st.write("Overview of your RAG-X knowledge base.")

st.divider()


try:
    overview = load_overview()
except ApiError as error:
    st.error(str(error))
    st.stop()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Indexed documents", overview["indexed"])

with col2:
    st.metric("Failed uploads", overview["failed"])

with col3:
    st.metric("Questions", overview["questions"])

with col4:
    st.metric("Conversations", len(overview["conversations"]))


st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Documents")

    if overview["documents"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "File": d["original_filename"],
                        "Status": status_label(d["status"]),
                        "Size (KB)": round(d["size_bytes"] / 1024, 1),
                    }
                    for d in overview["documents"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No documents yet. Upload your first document to get started.")

with right:
    st.subheader("Recent conversations")

    conversations = [
        c for c in overview["conversations"] if c["message_count"] > 0
    ][:8]

    if conversations:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Title": c["title"],
                        "Messages": c["message_count"],
                        "Last active": c["updated_at"][:16].replace("T", " "),
                    }
                    for c in conversations
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No conversations yet. Ask your first question on the Chat page.")