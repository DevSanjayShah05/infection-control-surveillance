"""
Infection Control Command Center -- multi-page entry point.
Real data: CMS Provider Data Catalog (HAI measures + Hospital General Info).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import streamlit as st
import theme

st.set_page_config(page_title="Infection Control Command Center", page_icon="🦠", layout="wide")
st.markdown(theme.inject_css(), unsafe_allow_html=True)

from pages_src import executive_summary, facility_benchmarking, statistical_testing, predictive_model, cusum_demo

pages = [
    st.Page(executive_summary.render, title="Executive Summary", icon="📋", default=True, url_path="executive-summary"),
    st.Page(facility_benchmarking.render, title="Facility Benchmarking", icon="🗺️", url_path="facility-benchmarking"),
    st.Page(statistical_testing.render, title="Statistical Testing", icon="📐", url_path="statistical-testing"),
    st.Page(predictive_model.render, title="Predictive Model", icon="🤖", url_path="predictive-model"),
    st.Page(cusum_demo.render, title="Control Chart Demo (Synthetic)", icon="🟠", url_path="control-chart-demo"),
]

nav = st.navigation(pages)
nav.run()
