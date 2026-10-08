import streamlit as st

from utils.api_client import list_documents, summarize_document


SUMMARY_TYPES = {
    "Quick Summary": "quick",
    "Detailed Summary": "detailed",
    "Key Points": "key_points",
    "Methodology": "methodology",
    "Findings": "findings",
    "Limitations": "limitations",
    "Keywords": "keywords",
}


st.title("Document Intelligence")
st.write(
    "Generate AI-powered summaries and insights from your indexed documents."
)

try:
    documents = list_documents()
except Exception as exc:
    st.error(f"Could not load documents: {exc}")
    st.stop()

if not documents:
    st.info("No documents available. Upload and index a document first.")
    st.stop()

document_options = {
    document["original_filename"]: document["document_id"]
    for document in documents
}

selected_filename = st.selectbox(
    "Select a document",
    options=list(document_options.keys()),
)

selected_document_id = document_options[selected_filename]

selected_summary_label = st.selectbox(
    "Select analysis type",
    options=list(SUMMARY_TYPES.keys()),
)

summary_type = SUMMARY_TYPES[selected_summary_label]

if st.button("Generate", type="primary"):
    with st.spinner("Analyzing document..."):
        try:
            result = summarize_document(
                document_id=selected_document_id,
                summary_type=summary_type,
            )

            st.subheader(selected_summary_label)

            st.write(result["summary"])

            st.caption(
                f"Source chunks: {result['source_chunks']}"
            )

        except Exception as exc:
            st.error(f"Failed to generate analysis: {exc}")