import streamlit as st
import pandas as pd
from constants import DF_COLUMNS, INITIAL_DF
from utils.geocoding import GeoHelper


def filter_dataframe():
    """Filter accident data based on user selection."""
    st.subheader("Select an urban area for accident analysis.")
    filter_level = st.selectbox("Filter Level", [None, "CITY", "TOWN", "CITY_DISTRICT", "SUBURB", "ROAD"])
    filtered_df = pd.DataFrame()
    filtered_locations = []

    if filter_level:
        filtered_locations = st.multiselect(
            f"Select {filter_level}",
            list(st.session_state.df[filter_level].unique())
        )
        filtered_df = st.session_state.df[st.session_state.df[filter_level].isin(filtered_locations)]

    return filtered_locations, filter_level, filtered_df


def show_dataframe(filtered_df):
    if not filtered_df.empty:
        st.dataframe(filtered_df, width=800, height=200)
    else:
        if not st.session_state.df.empty:
            st.dataframe(st.session_state.df, width=800, height=200)
        else:
            st.info("📌 Select a location to see accident details.")


def update_dataframe(draw_data):
    """Update session state with new drawing data."""

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

    st.markdown("### 🏡 Address Details")

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

    return draw_data