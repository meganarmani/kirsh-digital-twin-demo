from pathlib import Path
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Kirsh Digital Twin", layout="wide")

DATA_DIR = Path("data/Tel_Aviv/digital_twin")


@st.cache_data
def load_data():
    field_master = pd.read_parquet(DATA_DIR / "field_master.parquet")
    field_state = pd.read_parquet(DATA_DIR / "field_state_biweekly.parquet")
    field_alerts = pd.read_parquet(DATA_DIR / "field_alerts.parquet")
    field_changes = pd.read_parquet(DATA_DIR / "field_changes_latest.parquet")
    field_map = pd.read_parquet(DATA_DIR / "field_master_map.parquet")
    return field_master, field_state, field_alerts, field_changes, field_map


def get_field_id_col(df):
    for c in ["field_id", "plot_id", "id", "fid", "OBJECTID"]:
        if c in df.columns:
            return c
    raise KeyError("No field id column found")


field_master, field_state, field_alerts, field_changes, field_map = load_data()
field_id_col = get_field_id_col(field_master)

# force numeric map coords
if "lon" in field_map.columns:
    field_map["lon"] = pd.to_numeric(field_map["lon"], errors="coerce")
if "lat" in field_map.columns:
    field_map["lat"] = pd.to_numeric(field_map["lat"], errors="coerce")

# drop bad rows
if {"lon", "lat"}.issubset(field_map.columns):
    field_map = field_map.dropna(subset=["lon", "lat"]).copy()

st.title("Kirsh Digital Twin")

c1, c2, c3 = st.columns(3)
c1.metric("Fields", len(field_master))
c2.metric("Alerts", len(field_alerts))
c3.metric("Area (ha)", f"{field_master.get('area_ha_est', pd.Series(dtype=float)).fillna(0).sum():,.1f}")

tabs = st.tabs(["Overview", "Map", "Field Analytics", "Alerts", "Change Detection", "Debug"])

with tabs[0]:
    a, b = st.columns(2)
    
    if "field_crop_final" in field_master.columns:
        crop_counts = (
            field_master["field_crop_final"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("crop")
            .reset_index(name="n_fields")
        )
        a.plotly_chart(
            px.bar(crop_counts, x="crop", y="n_fields", title="Fields by Crop"),
            use_container_width=True,
        )
    
    if "health_label" in field_master.columns:
        health_counts = (
            field_master["health_label"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("health")
            .reset_index(name="n_fields")
        )
        b.plotly_chart(
            px.bar(health_counts, x="health", y="n_fields", title="Fields by Health"),
            use_container_width=True,
        )

with tabs[1]:
    st.subheader("Field Map")
    
    if field_map.empty:
        st.error("field_master_map.parquet loaded, but no rows remain after lon/lat cleanup.")
    elif not {"lat", "lon"}.issubset(field_map.columns):
        st.error("field_master_map.parquet is missing lon/lat columns.")
        st.write("Columns found:", list(field_map.columns))
    else:
        st.write(f"Map rows: {len(field_map):,}")
        st.write("Latitude range:", float(field_map["lat"].min()), "to", float(field_map["lat"].max()))
        st.write("Longitude range:", float(field_map["lon"].min()), "to", float(field_map["lon"].max()))
    
        # brighter visible points
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=field_map,
            get_position='[lon, lat]',
            get_radius=120,
            get_fill_color=[0, 200, 255, 180],
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
            pickable=True,
            stroked=True,
            opacity=0.9,
        )
    
        view_state = pdk.ViewState(
            latitude=float(field_map["lat"].mean()),
            longitude=float(field_map["lon"].mean()),
            zoom=8,
            pitch=0,
        )
    
        tooltip = {
            "html": f"""
            <b>Field ID:</b> {{{field_id_col}}}<br/>
            <b>Crop:</b> {{field_crop_final}}<br/>
            <b>Health:</b> {{health_label}}<br/>
            <b>Alert:</b> {{alert_level}}<br/>
            <b>Area (ha):</b> {{area_ha_est}}
            """
        }
    
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip,
            )
        )
    
        st.markdown("### Sample mapped rows")
        show_cols = [c for c in [field_id_col, "lon", "lat", "field_crop_final", "health_label", "alert_level"] if c in field_map.columns]
        st.dataframe(field_map[show_cols].head(20), use_container_width=True)

with tabs[2]:
    field_ids = field_master[field_id_col].astype(str).sort_values().tolist()
    selected = st.selectbox("Select Field", field_ids)
    
    state_f = field_state[field_state[field_id_col].astype(str) == selected].copy()
    
    metrics = [c for c in ["ndvi_mean", "ndre_mean", "evi_mean", "lai_mean", "health_score", "anomaly_score"] if c in state_f.columns]
    if metrics and not state_f.empty:
        plot_df = state_f.melt(
            id_vars=["bin_start"],
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
        fig = px.line(plot_df, x="bin_start", y="value", color="metric", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(state_f.head(50), use_container_width=True)

with tabs[3]:
    st.dataframe(field_alerts.head(100), use_container_width=True)

with tabs[4]:
    st.dataframe(field_changes.head(100), use_container_width=True)

with tabs[5]:
    st.markdown("### Debug")
    st.write("field_master shape:", field_master.shape)
    st.write("field_state shape:", field_state.shape)
    st.write("field_alerts shape:", field_alerts.shape)
    st.write("field_changes shape:", field_changes.shape)
    st.write("field_map shape:", field_map.shape)
    st.write("field_map columns:", list(field_map.columns))
