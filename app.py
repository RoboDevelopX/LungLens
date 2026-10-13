"""LungLens: See what the AI sees. Entry point of the Streamlit app.

Run with:  streamlit run app.py

This file runs first on EVERY page view: it sets the page layout and the shared
look, then shows the navigation menu and runs the page the user picked
(the page files live in pages/).
"""

import streamlit as st

from src import ui

ui.setup_page()

navigation = st.navigation(
    [
        st.Page("pages/screen.py", title="Screen an X-ray", icon=":material/radiology:", default=True),
        st.Page("pages/how_it_works.py", title="How it works", icon=":material/school:"),
        st.Page("pages/model_performance.py", title="Model performance", icon=":material/monitoring:"),
        st.Page("pages/sanity_check.py", title="Sanity check", icon=":material/fact_check:"),
        st.Page("pages/limitations_and_ethics.py", title="Limitations & ethics", icon=":material/balance:"),
    ]
)
with st.sidebar:
    st.caption("LungLens is an educational screening aid, not a diagnostic device.")
navigation.run()
