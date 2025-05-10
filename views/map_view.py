import streamlit as st
from typing import Tuple, Dict, Any
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
from components.map.data_component import (
    handle_drawing_state,
    show_dataframe,
    filter_dataframe,
    render_dataframe
)
from components.map.analysis_component import analyze_dataframe
from utils.session.session_handler import clear_dataframe

def render_header() -> None:
    """Render the page header."""
    st.markdown(
        '<h1 style="text-align: center; color: #2E86C1;">Interactive Map Analysis</h1>',
        unsafe_allow_html=True
    )
    
    if st.button("← Back to Main Menu", key="back_to_main"):
        st.session_state.current_view = "main"
        clear_dataframe()
        st.rerun()


def render_map_section() -> Dict[str, Any]:
    """Render the map section and return the draw data."""
    try:
        st.markdown("## 📍 Interactive Map")

        with st.expander("🛠️ How to Analyze Accident Data"):
            st.markdown("""
            1. **📍 Select a Location**: Click on the map to drop a point and fetch address details.
            2. **🎛 Choose Filter Level**: Use the dropdown menu to filter by road, suburb, town, city, or district.
            3. **📊 Analyze Data**: Click the **Analyze** button to process and display accident data.
            4. **🗑 Clear Data**: Use the **Clear Data** button to reset your selection.
            """)

        m = leafmap.Map(center=[32.783087, 34.965285], zoom=10, use_container_width=True)
        m.add_layer_control()
        return st_folium(m, width=700, height=500, key="leafmap_draw")
    except Exception as e:
        st.error(f"Error rendering map: {str(e)}")
        return {}

def render_dataframe_section(draw_data: Dict[str, Any]) -> Tuple[list, str, Any]:
    """Render the dataframe section and return filtered data."""
    try:
        render_dataframe(draw_data)
        filtered_locations, filter_level, filtered_df = filter_dataframe()
        show_dataframe(filtered_df)
                   
        return filtered_locations, filter_level, filtered_df
    except Exception as e:
        st.error(f"Error processing dataframe: {str(e)}")
        return [], "", None

def render_analysis_sections(filtered_data: Tuple[list, str, Any]) -> None:
    """Render analysis sections based on filtered data."""
    filtered_locations, filter_level, filtered_df = filtered_data
    
    if filtered_df is None:
        return
        
    st.markdown("### Analysis")
    
    # Define analysis types and their display names
    analysis_types = {
        "general": "General Analysis",
        "cause": "Cause Analysis",
        "outcome": "Outcome Analysis"
    }
    
    # Render each analysis type in its own container
    for analysis_type, _ in analysis_types.items():
        with st.container():
            analyze_dataframe(filtered_locations, filter_level, filtered_df, analysis_type)

def show_map_view() -> None:
    """Main function to show the map view."""
    try:
        # Render header
        render_header()
        
        # Define layout columns
        map_col, dataframe_col = st.columns([2.5, 2.5], gap="medium")
        
        # Render map section
        with map_col:
            draw_data = render_map_section()
            
        # Update and process data
        draw_data = handle_drawing_state(draw_data)
        
        # Render dataframe section
        with dataframe_col:
            filtered_data = render_dataframe_section(draw_data)
            
        # Render analysis sections
        render_analysis_sections(filtered_data)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    show_map_view()