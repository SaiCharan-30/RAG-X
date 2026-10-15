import streamlit as st

from utils import api_client
from utils.api_client import ApiError
from utils.components import document_scope_selector


st.title("🔎 Search")

st.write(
    "Search across your indexed documents. This shows the passages "
    "themselves, without writing an answer."
)

st.divider()


with st.sidebar:
    st.subheader("Documents")
    mode, document_ids, scope_ok = document_scope_selector("search")

    st.divider()
    st.subheader("Options")

    top_k = st.slider("Number of results", 1, 20, 5)
    rerank = st.checkbox(
        "Rerank results",
        value=True,
        help="Scores each result with a more accurate model and shows its "
        "relevance. Results keep the order hybrid search found them. "
        "Slower than plain search.",
    )


query = st.text_input(
    "Enter your search query",
    placeholder="Example: What is the proposed architecture?",
)


if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")

    elif not scope_ok:
        st.warning("Choose the documents to search in the sidebar first.")

    else:
        with st.spinner("Searching ..."):
            try:
                response = api_client.search(
                    query.strip(),
                    mode,
                    document_ids,
                    top_k,
                    rerank,
                )
            except ApiError as error:
                st.error(str(error))
                st.stop()

        if not response["results"]:
            st.info("No matching passages found.")

        for result in response["results"]:
            with st.container(border=True):
                title = f"**#{result['rank']} — {result['filename']}**"

                if result["page_number"] is not None:
                    title += f" · page {result['page_number']}"

                st.markdown(title)

                if result["rerank_score"] is not None:
                    st.caption(f"Relevance {result['rerank_score']:.0%}")
                else:
                    st.caption("Not reranked")

                st.write(result["text"])