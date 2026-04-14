from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Kirsh Digital Twin", layout="wide")

# ---------------------------
# CONFIG (RELATIVE PATHS)
# ---------------------------
DATA_DIR = Path("data/Tel_Aviv/digital_twin")

@st.cache_data
def load_data():
    field_master = pd.read_parquet(DATA_DIR / "field_master.parquet")
    field_state = pd.read_parquet(DATA_DIR / "field_state_biweekly.parquet")
    field_alerts = pd.read_parquet(DATA_DIR / "field_alerts.parquet")
    field_changes = pd.read_parquet(DATA_DIR / "field_changes_latest.parquet")
    field_map = pd.read_parquet(DATA_DIR / "field_master_map.parquet")

    return field_master, field_state, field_alerts, field_changes, field_map

field_master, field_state, field_alerts, field_changes, field_map = load_data()

# ---------------------------
# HEADER
# ---------------------------
st.title("Kirsh Digital Twin")

c1, c2, c3 = st.columns(3)
c1.metric("Fields", len(field_master))
c2.metric("Alerts", len(field_alerts))
c3.metric("Area (ha)", f"{field_master.get('area_ha_est', pd.Series()).sum():,.1f}")

# ---------------------------
# MAP
# ---------------------------
st.subheader("Field Map")

if {"lat", "lon"}.issubset(field_map.columns):

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=field_map,
        get_position="[lon, lat]",
        get_radius=50,
        get_fill_color=[200, 30, 0, 160],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=field_map["lat"].mean(),
        longitude=field_map["lon"].mean(),
        zoom=9,
    )

    st.pydeck_chart(pdk.Deck(
        initial_view_state=view_state,
        layers=[layer],
    ))

# ---------------------------
# HEALTH
# ---------------------------
if "health_label" in field_master.columns:
    st.subheader("Health Distribution")
    st.plotly_chart(
        px.histogram(field_master, x="health_label"),
        use_container_width=True
    )

# ---------------------------
# ALERTS
# ---------------------------
st.subheader("Alerts")
st.dataframe(field_alerts.head(100))

# ---------------------------
# CHANGE DETECTION
# ---------------------------
if "health_change" in field_changes.columns:
    st.subheader("Recent Health Changes")
    st.plotly_chart(
        px.bar(field_changes.head(50), x="health_change"),
        use_container_width=True
    )
