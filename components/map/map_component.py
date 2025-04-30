import streamlit as st
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium


def render_map():
    """Render the interactive map and return drawing data."""
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
