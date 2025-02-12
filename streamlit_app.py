import streamlit as st
from components.analysis_component import analyze_dataframe
from components.data_component import update_dataframe, show_dataframe, filter_dataframe, render_dataframe
from components.dialog import Dialog
from components.map_component import render_map
from utils.session_handler import initialize_session, clear_dataframe


class CarAccidentApp:
    def __init__(self):
        st.set_page_config(layout="wide")  # Ensure full-width layout

        if "initialized" not in st.session_state:
            initialize_session()

    @staticmethod
    def show_dialog_side():
        if st.sidebar.button("Chat with AI Road Expert"):
            Dialog.show_dialog()

    @staticmethod
    def show_about_side():
        st.sidebar.header("About")
        st.sidebar.info(
            """
            This application is an educational project developed solely for learning purposes.
            It utilizes accident data from LAMAS covering the years 2003 to 2023.
            """
        )

    def run(self):
        self.show_dialog_side()
        st.sidebar.divider()
        self.show_about_side()

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
    app = CarAccidentApp()
    app.run()
