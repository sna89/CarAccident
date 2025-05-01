import streamlit as st
from components.rag.dialog import Dialog

def show_chat_view():
    # Header
    st.markdown('<h1 style="text-align: center; color: #2E86C1;">AI Road Expert Chat</h1>', unsafe_allow_html=True)
    
    # Create two columns for the layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(Dialog.font_markdown.format(message["content"]), unsafe_allow_html=True)

        # Chat input
        prompt = st.chat_input("Ask me a question about road safety and accident patterns?")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                Dialog.show_message(prompt)
            
            with st.chat_message("assistant"):
                response = Dialog.rag.chat(prompt)
                Dialog.show_message(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        # Buttons in the right column
        if st.button("← Back to Main Menu", key="back_to_main"):
            st.session_state.current_view = "main"
            st.rerun()
        if st.button("Clear Chat"):
            st.session_state.messages = []

if __name__ == "__main__":
    show_chat_view()    
