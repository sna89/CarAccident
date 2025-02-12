import streamlit as st

from components.rag import RAG


class Dialog:
    font_markdown = '<p style="font-family: David; font-size: 14; color: #555555;">{}</p>'
    rag = RAG()

    def __init__(self):
        pass

    @staticmethod
    def show_message(prompt):
        st.markdown(Dialog.font_markdown.format(prompt), unsafe_allow_html=True)

    @staticmethod
    @st.dialog("AI Chat", width="large")
    def show_dialog():
        Dialog.show_history()

        prompt = st.chat_input("Ask me a question?")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                Dialog.show_message(prompt)

            with st.chat_message("assistant"):
                response = Dialog.rag.chat(prompt)
                Dialog.show_message(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        if st.button("Clear Chat"):
            st.session_state.messages = []

        if st.button("Close Dialog"):
            st.rerun()

    @staticmethod
    def show_history():
        for message in st.session_state.messages[2:]:
            with st.chat_message(message["role"]):
                st.markdown(Dialog.font_markdown.format(message["content"]), unsafe_allow_html=True)