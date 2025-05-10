import streamlit as st
from typing import List, Dict, Optional
from rag.rag_main import RAG

class Dialog:
    """Handles the chat dialog interface and state management."""
    
    def __init__(self):
        """Initialize the dialog with RAG and session state."""
        self.rag = RAG()
        self._initialize_session_state()
        
    def _initialize_session_state(self) -> None:
        """Initialize session state variables if they don't exist."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "is_dialog_open" not in st.session_state:
            st.session_state.is_dialog_open = False
            
    def show_message(self, content: str, role: str = "user") -> None:
        """Display a message in the chat interface."""
        try:
            with st.chat_message(role):
                st.markdown(
                    f'<p style="font-family: David; font-size: 14; color: #555555;">{content}</p>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error displaying message: {str(e)}")
            
    def process_user_input(self, prompt: str) -> None:
        """Process user input and generate response."""
        try:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            self.show_message(prompt, "user")
            
            # Generate and show response
            with st.spinner("Thinking..."):
                response = self.rag.chat(prompt)
                self.show_message(response, "assistant")
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            
    def clear_chat(self) -> None:
        """Clear the chat history."""
        st.session_state.messages = []
        st.rerun()
        
    def show_dialog(self) -> None:
        """Display the chat dialog interface."""
        with st.dialog("AI Chat", width="large"):
            # Chat input
            prompt = st.chat_input("Ask me a question?")
            if prompt:
                self.process_user_input(prompt)
                
            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat", help="Clear chat history"):
                    self.clear_chat()
            with col2:
                if st.button("Close Dialog", help="Close chat window"):
                    st.session_state.is_dialog_open = False
                    st.rerun()
                    
    def show_history(self) -> None:
        """Display chat history."""
        try:
            for message in st.session_state.messages:
                self.show_message(message["content"], message["role"])
        except Exception as e:
            st.error(f"Error displaying history: {str(e)}")
            
    def toggle_dialog(self) -> None:
        """Toggle the dialog open/closed state."""
        st.session_state.is_dialog_open = not st.session_state.is_dialog_open
        if st.session_state.is_dialog_open:
            self.show_dialog()