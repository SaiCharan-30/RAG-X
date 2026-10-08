import streamlit as st


st.title("📄 Documents")

st.write(
    "Upload, view, search, and manage your documents."
)

st.divider()


uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "html"],
    accept_multiple_files=True,
)


if uploaded_files:

    st.success(
        f"{len(uploaded_files)} document(s) selected."
    )

    for file in uploaded_files:
        st.write(
            f"📄 {file.name} — {file.size / 1024:.2f} KB"
        )

else:

    st.info("No documents uploaded yet.")