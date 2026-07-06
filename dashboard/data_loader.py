import duckdb
import joblib
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@st.cache_data
def load_enriched():
    con = duckdb.connect(str(DATA_DIR / "hai.duckdb"), read_only=True)
    df = con.execute("SELECT * FROM hai_facility_measures_enriched").df()
    control_df = con.execute("SELECT * FROM hai_control_limits").df()
    con.close()
    return df, control_df


@st.cache_resource
def load_model_results():
    return joblib.load(DATA_DIR / "predictive_model_results.joblib")
