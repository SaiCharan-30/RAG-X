import streamlit as st

from utils import api_client
from utils.api_client import ApiError
from utils.components import document_label, status_label


st.title("📄 Documents")

st.write(
    "Upload, view, summarize, and manage your documents."
)

st.divider()


# ---------------------------------------------------------------- upload

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Results of the last upload are kept here so they survive the page refresh.
for result in st.session_state.pop("upload_results", []):
    if result["ok"]:
        st.success(result["text"])
    else:
        st.error(result["text"])


uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "html"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)


if uploaded_files and st.button("Upload and index", type="primary"):
    results = []

    for file in uploaded_files:
        with st.spinner(f"Indexing {file.name} ..."):
            try:
                response = api_client.upload_document(
                    file.name,
                    file.getvalue(),
                    file.type,
                )
            except ApiError as error:
                results.append(
                    {"ok": False, "text": f"{file.name}: {error}"}
                )
                continue

        if response["status"] == "indexed":
            results.append(
                {
                    "ok": True,
                    "text": (
                        f"{file.name}: indexed "
                        f"({response['chunks_indexed']} chunks)."
                    ),
                }
            )
        else:
            results.append(
                {
                    "ok": False,
                    "text": (
                        f"{file.name}: could not be indexed — "
                        f"{response.get('error') or 'unknown error'}"
                    ),
                }
            )

    st.session_state.upload_results = results
    st.session_state.uploader_key += 1
    st.rerun()


# ------------------------------------------------------------------ list

st.subheader("Your documents")

try:
    documents = api_client.list_documents()
except ApiError as error:
    st.error(str(error))
    st.stop()

if not documents:
    st.info("No documents uploaded yet.")

for document in documents:
    name_col, status_col, size_col, action_col = st.columns([4, 2, 1, 1])

    name_col.write(f"📄 {document['original_filename']}")
    status_col.write(status_label(document["status"]))
    size_col.write(f"{document['size_bytes'] / 1024:.1f} KB")

    with action_col.popover("🗑️"):
        st.write("Delete this document and its search index?")

        if st.button(
            "Yes, delete",
            key=f"delete_{document['document_id']}",
            type="primary",
        ):
            try:
                api_client.delete_document(document["document_id"])
            except ApiError as error:
                st.error(str(error))
            else:
                st.rerun()


# --------------------------------------------------------------- summary

indexed = [d for d in documents if d["status"] == "indexed"]

if indexed:
    st.divider()
    st.subheader("Summarize a document")

    labels = {document_label(d): d["document_id"] for d in indexed}

    col1, col2 = st.columns([3, 2])

    picked = col1.selectbox("Document", list(labels), key="summary_document")

    summary_type = col2.selectbox(
        "Summary type",
        [
            "quick",
            "detailed",
            "key_points",
            "methodology",
            "findings",
            "limitations",
            "keywords",
        ],
        key="summary_type",
    )

    if st.button("Generate summary"):
        with st.spinner("Summarizing ..."):
            try:
                summary = api_client.summarize_document(
                    labels[picked],
                    summary_type,
                )
            except ApiError as error:
                st.error(str(error))
            else:
                st.markdown(summary["summary"])
                st.caption(
                    f"Based on {summary['source_chunks']} chunk(s)."
                )