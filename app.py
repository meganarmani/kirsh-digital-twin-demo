from __future__ import annotations

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Kirsh Digital Twin", layout="wide")

ROOT_DIR = Path(__file__).parent / "data"

# Colors
CROP_COLOR_MAP = {
    "wheat": [214, 174, 96, 180],
    "potato": [140, 98, 57, 180],
    "carrot": [230, 120, 40, 180],
    "unknown": [140, 140, 140, 150],
    "other": [200, 200, 200, 120],
}

HEALTH_COLOR_MAP = {
    "strong": [46, 125, 50, 180],
    "moderate": [251, 192, 45, 180],
    "weak": [244, 124, 32, 180],
    "critical": [198, 40, 40, 180],
    "unknown": [140, 140, 140, 150],
}

ALERT_COLOR_MAP = {
    "normal": [76, 175, 80, 180],
    "watch": [255, 193, 7, 180],
    "moderate": [255, 112, 67, 180],
    "severe": [211, 47, 47, 180],
    "unknown": [140, 140, 140, 150],
}

# ============================================================
# HELPERS
# ============================================================

def get_region_dirs(root_dir: Path):
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def get_field_id_col(df):
    for c in ["field_id", "plot_id", "id", "fid", "OBJECTID"]:
        if c in df.columns:
            return c
    raise KeyError("No field id column found")


def to_geojson_with_color(gdf, value_col, color_map):
    gdf = gdf.copy()
    gdf["color"] = gdf[value_col].map(color_map).apply(
        lambda x: x if isinstance(x, list) else [140, 140, 140, 120]
    )
    return json.loads(gdf.to_json())


@st.cache_data
def load_data(region_dir):
    twin_dir = region_dir / "digital_twin"

    field_master = pd.read_parquet(twin_dir / "field_master.parquet")
    field_state = pd.read_parquet(twin_dir / "field_state_biweekly.parquet")
    field_alerts = pd.read_parquet(twin_dir / "field_alerts.parquet")
    field_changes = pd.read_parquet(twin_dir / "field_changes_latest.parquet")
    fields_gdf = gpd.read_file(twin_dir / "field_master.gpkg")

    fields_gdf = fields_gdf.to_crs("EPSG:4326")

    return field_master, field_state, field_alerts, field_changes, fields_gdf


# ============================================================
# LOAD
# ============================================================

regions = get_region_dirs(ROOT_DIR)
region_name = st.sidebar.selectbox("Region", [r.name for r in regions])
region_dir = ROOT_DIR / region_name

field_master, field_state, field_alerts, field_changes, fields_gdf = load_data(region_dir)

field_id_col = get_field_id_col(field_master)

# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("## Filters")

crop_filter = st.sidebar.selectbox(
    "Crop",
    ["All"] + sorted(field_master["field_crop_final"].dropna().unique().tolist())
)

health_filter = st.sidebar.selectbox(
    "Health",
    ["All"] + sorted(field_master.get("health_label", pd.Series()).dropna().unique().tolist())
)

# Apply filters
df = field_master.copy()

if crop_filter != "All":
    df = df[df["field_crop_final"] == crop_filter]

if health_filter != "All" and "health_label" in df.columns:
    df = df[df["health_label"] == health_filter]

ids = set(df[field_id_col].astype(str))
fields_f = fields_gdf[fields_gdf[field_id_col].astype(str).isin(ids)]

# ============================================================
# HEADER
# ============================================================

st.title(" Kirsh Digital Twin")

c1, c2, c3 = st.columns(3)

c1.metric("Fields", len(df))
c2.metric("Alerts", df["alert_level"].isin(["watch", "moderate", "severe"]).sum() if "alert_level" in df.columns else "NA")
c3.metric("Weak Fields", df["health_label"].isin(["weak", "critical"]).sum() if "health_label" in df.columns else "NA")

# ============================================================
# TABS
# ============================================================

tabs = st.tabs(["Map", "Field Analytics", "Alerts", "QA"])

# ============================================================
# MAP
# ============================================================

with tabs[0]:

    layer_type = st.radio("Layer", ["Crop", "Health", "Alert"], horizontal=True)

    if layer_type == "Crop":
        geo = to_geojson_with_color(fields_f, "field_crop_final", CROP_COLOR_MAP)
    elif layer_type == "Health":
        geo = to_geojson_with_color(fields_f, "health_label", HEALTH_COLOR_MAP)
    else:
        geo = to_geojson_with_color(fields_f, "alert_level", ALERT_COLOR_MAP)

    centroid = fields_f.geometry.centroid
    view = pdk.ViewState(
        latitude=float(centroid.y.mean()),
        longitude=float(centroid.x.mean()),
        zoom=9
    )

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geo,
        get_fill_color="properties.color",
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

# ============================================================
# FIELD ANALYTICS
# ============================================================

with tabs[1]:

    field_ids = df[field_id_col].astype(str).tolist()
    selected = st.selectbox("Select Field", field_ids)

    state_f = field_state[field_state[field_id_col].astype(str) == selected]

    if not state_f.empty:
        metrics = ["ndvi_mean", "ndre_mean", "health_score"]
        metrics = [m for m in metrics if m in state_f.columns]

        plot_df = state_f.melt(
            id_vars=["bin_start"],
            value_vars=metrics
        )

        fig = px.line(plot_df, x="bin_start", y="value", color="variable")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ALERTS
# ============================================================

with tabs[2]:
    st.dataframe(field_alerts.head(100))

# ============================================================
# QA
# ============================================================

with tabs[3]:

    weak = df[
        (df.get("majority_ratio_known", 1) < 0.6) |
        (df.get("field_crop_prob_score", 1) < 0.6)
    ]

    st.dataframe(weak.head(50))
