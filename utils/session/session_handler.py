import streamlit as st
from config import INITIAL_DF, LLM_MODEL
from utils.sql.sql_llm_agent import SqlLLMAgent
from utils.llm.model import get_llm_model
from utils.sql.sql_db import SqlDb

def initialize_session() -> None:
    """Initialize Streamlit session state variables."""
    st.session_state.latest_feature = None
    st.session_state.df = INITIAL_DF
    st.session_state.initialized = True
    st.session_state.new_entry = False
    st.session_state.clear_data = False
    st.session_state.current_view = "main"
    
    # Initialize database connection
    st.session_state.db = SqlDb()
    
    # Initialize agents and models
    st.session_state.sql_llm_agent = SqlLLMAgent()
    st.session_state.llm_model = get_llm_model(provider="openai", model_name=LLM_MODEL)

    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_dataframe():
    """Clear session data and reset the application state."""
    # Clear all map-related states
    st.session_state.leafmap_draw = {}
    st.session_state.all_drawings = []
    st.session_state.latest_feature = None
    st.session_state.clear_data = True
    
    # Reset dataframe
    st.session_state.df = INITIAL_DF
    
    st.rerun()
