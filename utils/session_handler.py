import streamlit as st
from config import INITIAL_DF
from utils.sql_llm_agent import SqlLLMAgent


def initialize_session() -> None:
    """Initialize Streamlit session state variables."""
    st.session_state.latest_feature = None
    st.session_state.df = INITIAL_DF
    st.session_state.initialized = True
    st.session_state.new_entry = False
    st.session_state.clear_data = False

    st.session_state.sql_llm_agent = SqlLLMAgent()

    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_dataframe():
    """Clear session data and reset the application state."""
    st.session_state.leafmap_draw = {}
    st.session_state.all_drawings = []
    st.session_state.clear_data = True
    st.session_state.df = INITIAL_DF
    st.rerun()
