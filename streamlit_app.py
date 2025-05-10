import streamlit as st
from utils.session.session_handler import initialize_session
from views.main_screen import show_main_screen_view
from views.map_view import show_map_view
from views.chat_view import show_chat_view
from views.research_view import show_research_view 

class CarAccidentApp:
    def __init__(self):
        st.set_page_config(
            layout="wide",
            page_title="Car Accident Analysis",
            page_icon="🚗"
        )

        if "initialized" not in st.session_state:
            initialize_session()
            st.session_state.current_view = "main"  # Initialize the current view
            st.session_state.messages = []  # Initialize chat messages
            st.session_state.research_messages = []  # Initialize research messages
                    

    def run(self):
        if st.session_state.current_view == "main":
            show_main_screen_view()
        elif st.session_state.current_view == "map":
            show_map_view()
        elif st.session_state.current_view == "chat":
            show_chat_view()
        elif st.session_state.current_view == "research":
            show_research_view()


if __name__ == "__main__":
    app = CarAccidentApp()
    app.run()
