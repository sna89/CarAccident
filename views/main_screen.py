import streamlit as st

def show_main_screen_view():
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5em;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 1em;
        }
        .welcome-text {
            font-size: 1.2em;
            color: #34495E;
            text-align: center;
            margin-bottom: 2em;
        }
        .option-card {
            background-color: white;
            padding: 2em;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .option-card:hover {
            transform: translateY(-5px);
        }
        .option-title {
            font-size: 1.5em;
            color: #2C3E50;
            margin-bottom: 0.5em;
        }
        .option-description {
            color: #7F8C8D;
            margin-bottom: 1em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Car Accident Analysis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Welcome to our comprehensive car accident analysis platform. Choose an option below to begin your analysis.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="option-title">📍 Interactive Map</h2>', unsafe_allow_html=True)
        st.markdown('<p class="option-description">Explore accident data through our interactive map interface. Select locations and analyze patterns.</p>', unsafe_allow_html=True)
        if st.button("Open Map", key="map_button"):
            st.session_state.current_view = "map"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="option-title">💬 AI Road Expert</h2>', unsafe_allow_html=True)
        st.markdown('<p class="option-description">Chat with our AI expert to get instant insights about road safety and accident patterns.</p>', unsafe_allow_html=True)
        if st.button("Start Chat", key="chat_button"):
            st.session_state.current_view = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="option-title">🔍 Deep Research</h2>', unsafe_allow_html=True)
        st.markdown('<p class="option-description">Conduct in-depth research and analysis of accident data with advanced AI capabilities.</p>', unsafe_allow_html=True)
        if st.button("Start Research", key="research_button"):
            st.session_state.current_view = "research"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True) 


if __name__ == "__main__":
    show_main_screen_view()
