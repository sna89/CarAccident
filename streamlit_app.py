import streamlit as st
from components.analysis_component import analyze_dataframe
from components.data_component import update_dataframe, show_dataframe, filter_dataframe, render_dataframe
from components.map_component import render_map
from utils.session_handler import initialize_session, clear_dataframe


class CarAccidentApp:
    def __init__(self):
        st.set_page_config(layout="wide")  # Ensure full-width layout

        # Initialize session state if not already done
        if "initialized" not in st.session_state:
            initialize_session()

    def run(self):
        # Define layout columns
        map_col, dataframe_col = st.columns([2.5, 2.5], gap="medium")

        with map_col:
            draw_data = render_map()

        draw_data = update_dataframe(draw_data)

        with dataframe_col:
            render_dataframe(draw_data)
            filter_level, filtered_df = filter_dataframe()
            show_dataframe(filtered_df)
            if st.button("🔄 Clear Data", key="clear", help="Reset all selections"):
                clear_dataframe()

            analyze_dataframe(filter_level, filtered_df)


if __name__ == "__main__":
    app = CarAccidentApp()
    app.run()
