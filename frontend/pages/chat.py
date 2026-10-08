import streamlit as st


st.title("💬 Chat")

st.write(
    "Ask questions about your documents using conversational RAG."
)

st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


prompt = st.chat_input(
    "Ask a question about your documents..."
)


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        response = (
            "RAG chat backend will be connected in a later phase."
        )

        st.write(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )