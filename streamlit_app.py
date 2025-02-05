import streamlit as st
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import pandas as pd
from geocoding import GeoHelper
from constants import INITIAL_DF, DF_COLUMNS, INFERENCE_COLUMNS
from sql_db import LangChainSQL


def initialize_session() -> None:
    """
    Initialize Streamlit session state variables.
    """
    st.session_state.latest_feature = None
    st.session_state.df = INITIAL_DF
    st.session_state.initialized = True
    st.session_state.new_entry = False
    st.session_state.clear_data = False


def update_session_with_new_drawing(draw_data: dict) -> dict:
    """
    Update session state based on new drawing data.
    """
    if not st.session_state.clear_data:
        last_drawing = draw_data.get("last_active_drawing")
        if st.session_state.latest_feature != last_drawing:
            st.session_state.latest_feature = last_drawing
            st.session_state.new_entry = True
        else:
            st.session_state.new_entry = False
    else:
        st.session_state.clear_data = False

    return draw_data


def render_map() -> dict:
    """
    Render the interactive map and return its drawing data.
    """
    st.title("Interactive Map")
    m = leafmap.Map(center=[32.783087, 34.965285], zoom=10, google_map="SATELLITE")
    m.add_layer_control()
    map_data = st_folium(m, width=700, height=500, key="leafmap_draw")
    return map_data


def render_dataframe(draw_data: dict) -> dict:
    """
    Render the address details table and update session data if a new drawing is added.
    """

    def fill_key_in_dict(data: dict, keys: list) -> dict:
        """
        Ensure all required keys exist in the dictionary.
        """
        for col in keys:
            data.setdefault(col, None)
        return data

    def update_session_df(address_data: dict) -> None:
        """
        Update the session DataFrame with new address data.
        """
        if address_data:
            address_data = fill_key_in_dict(address_data, DF_COLUMNS)
            latest_df = pd.DataFrame(
                {str.upper(key): [value] for key, value in address_data.items() if key in DF_COLUMNS})
            st.session_state.df = pd.concat([st.session_state.df, latest_df], ignore_index=True)
            st.session_state.df = st.session_state.df.drop_duplicates()

    st.title("🏠 Address Details")

    # If there is a drawing or if data exists in the session DataFrame, proceed
    if st.session_state.latest_feature is not None or not st.session_state.df.equals(INITIAL_DF):
        if st.session_state.new_entry:
            # Extract coordinates from the latest feature
            try:
                lon, lat = st.session_state.latest_feature["geometry"]["coordinates"]
                geocode_result = GeoHelper.reverse_geocode(lat, lon)
                address_data = geocode_result.get("address", {})
            except Exception as e:
                st.error(f"Reverse geocoding failed: {e}")
                address_data = {}
            update_session_df(address_data)

    else:
        st.info("Draw a point on the map to display.")

    return draw_data


def filter_dataframe():
    def filter_df_by_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """
        Filter the DataFrame based on the selected granularity level.
        """
        granularity_location_options = df[granularity].unique().tolist()
        selected_location = st.selectbox(f"Select {granularity} from chosen points", granularity_location_options)
        filtered_df = df[df[granularity] == selected_location]
        return filtered_df

    st.subheader("Select Urban Area To Analyze For Accidents")
    filter_level = st.selectbox("Filter Level",
                                [None, "CITY", "TOWN", "CITY_DISTRICT", "SUBURB", "ROAD"])
    filtered_df = pd.DataFrame()

    if filter_level:
        filtered_df = filter_df_by_granularity(st.session_state.df, filter_level)
        st.dataframe(filtered_df, width=800, height=200)
    else:
        st.dataframe(st.session_state.df, width=800, height=200)

    return filter_level, filtered_df


def analyze_dataframe(filter_level, filtered_df):
    @st.cache_data()
    def call_llm_query(filter_level, filtered_df):
        sql_llm = LangChainSQL()
        if filter_level == "ROAD":
            filter_dict = filtered_df.to_dict()
        else:
            filter_dict = filtered_df[[filter_level]].to_dict()

        output = sql_llm.query_llm("Please find all accidents in the provided location: {}."
                                   "Please provide the top 5 reasons for accidents in the area, "
                                   "regarding factors which contribute to the possibility of an "
                                   "accident. Dont use columns from {} for this explanation."
                                   "Try to aggregate informative columns to get insights regarding the data"
                                   "In addition, please provide data regarding the number of accidents for each "
                                   "relevant insight you mention. "
                                   "Consider missing value while executing the sql query and retrieve all relevant "
                                   "records, but When summarizing the accident cause, please only mention informative "
                                   "cases, meaning, dont provide summary regarding missing or unknown data. "
                                   "After summarizing the accidents causes, please write a short paragraph which "
                                   "explains the main reasons for accidents in this area. From these reasons, "
                                   "suggest steps and actions that can help reduce the number of car accidents in the area."
                                   "If there are no accidents in this area, please state that your database does not "
                                   "include data regarding accidents in this area".format(
            filter_dict, INFERENCE_COLUMNS))
        return output

    if st.button("Analyze"):
        if not filter_level:
            st.error("Please choose a filter level.")
        else:
            if filtered_df.empty:
                st.error("Please choose point on the map and filter level to proceed")
            else:
                output = call_llm_query(filter_level, filtered_df)
                with st.expander("Show Analysis"):
                    st.write(output.content)


def clear_dataframe():
    st.session_state.leafmap_draw = {}
    st.session_state.all_drawings = []
    st.session_state.clear_data = True
    st.session_state.df = INITIAL_DF
    st.rerun()


if __name__ == "__main__":
    st.set_page_config(layout="wide")  # Ensure full-width layout

    # Initialize session state if not already done
    if "initialized" not in st.session_state:
        initialize_session()

    # Define layout columns
    map_col, dataframe_col = st.columns([2.5, 2.5], gap="medium")

    with map_col:
        draw_data = render_map()

    draw_data = update_session_with_new_drawing(draw_data)

    with dataframe_col:
        render_dataframe(draw_data)
        filter_level, filtered_df = filter_dataframe()

        if st.button("Clear Data"):
            clear_dataframe()

        analyze_dataframe(filter_level, filtered_df)
