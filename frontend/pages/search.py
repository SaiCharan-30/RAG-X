import streamlit as st


st.title("🔎 Search")

st.write(
    "Search across your indexed documents."
)

st.divider()


query = st.text_input(
    "Enter your search query",
    placeholder="Example: What is the proposed architecture?",
)


if st.button("Search"):

    if not query.strip():

        st.warning("Please enter a search query.")

    else:

        st.info(
            "Search backend will be connected in the retrieval phase."
        )