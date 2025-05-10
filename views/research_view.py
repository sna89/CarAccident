import streamlit as st
from components.qa.dialog import Dialog

def show_research_view():
    st.markdown('<h1 style="text-align: center; color: #2E86C1;">AI Road Expert Chat</h1>', unsafe_allow_html=True)

    research_question = st.text_area(
            "Enter your research question:",
            placeholder="e.g., What are the most common causes of accidents in urban areas?",
            height=100
        )
        
    if research_question:
        with st.chat_message("user"):
            Dialog.show_message(research_question)
        
        with st.chat_message("assistant"):
            response = Dialog.rag.chat(research_question)
            Dialog.show_message(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Clear Research"):
        st.session_state.messages = []

    if st.button("← Back to Main Menu", key="back_to_main"):
        st.session_state.current_view = "main"
        st.rerun()

if __name__ == "__main__":
    show_research_view()
