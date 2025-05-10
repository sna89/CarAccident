import streamlit as st
from components.qa.dialog import Dialog
from utils.session.session_handler import initialize_session

def show_chat_view():
    # Initialize session if needed
    if not st.session_state.get("initialized"):
        initialize_session()
    
    # Create dialog instance
    dialog = Dialog()
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #2E86C1;">AI Road Expert Chat</h1>', unsafe_allow_html=True)
    
    # Create two columns for the layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        dialog.show_history()

        # Chat input
        prompt = st.chat_input("Ask me a question about road safety and accident patterns?")
        if prompt:
            dialog.process_user_input(prompt)

    with col2:
        # Buttons in the right column
        if st.button("← Back to Main Menu", key="back_to_main"):
            st.session_state.current_view = "main"
            st.rerun()
        if st.button("Clear Chat"):
            dialog.clear_chat()

if __name__ == "__main__":
    show_chat_view()    
