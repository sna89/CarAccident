import streamlit as st
from components.map_component import render_map
from components.data_component import update_dataframe, show_dataframe, filter_dataframe, render_dataframe
from components.analysis_component import analyze_dataframe
from utils.session_handler import clear_dataframe

def show_map_view():
    st.markdown('<h1 style="text-align: center; color: #2E86C1;">Interactive Map Analysis</h1>', unsafe_allow_html=True)
        
    if st.button("← Back to Main Menu", key="back_to_main"):
        st.session_state.current_view = "main"
        st.rerun()

    # Define layout columns
    map_col, dataframe_col = st.columns([2.5, 2.5], gap="medium")

    with map_col:
        draw_data = render_map()

    draw_data = update_dataframe(draw_data)

    with dataframe_col:
        render_dataframe(draw_data)
        filtered_locations, filter_level, filtered_df = filter_dataframe()
        show_dataframe(filtered_df)
        if st.button("🔄 Clear Data", key="clear", help="Reset all selections"):
            clear_dataframe()

    cause_col, outcome_col = st.columns([1, 1])
    with cause_col:
        analyze_dataframe(filtered_locations,
                        filter_level,
                        filtered_df,
                        "cause")
    with outcome_col:
        analyze_dataframe(filtered_locations,
                        filter_level,
                        filtered_df,
                        "outcome")

if __name__ == "__main__":
    show_map_view()