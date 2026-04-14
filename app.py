from __future__ import annotations

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Kirsh Digital Twin", layout="wide")

ROOT_DIR = Path(__file__).parent / "data"

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

def get_region_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def get_field_id_col(df: pd.DataFrame) -> str:
    for c in ["field_id", "plot_id", "id", "fid", "OBJECTID"]:
        if c in df.columns:
            return c
    raise KeyError("No field id column found")


def to_geojson_with_color(gdf: gpd.GeoDataFrame, value_col: str, color_map: dict) -> dict:
    gdf = gdf.copy()

    if value_col not in gdf.columns:
        gdf[value_col] = "unknown"

    gdf["color"] = gdf[value_col].map(color_map).apply(
        lambda x: x if isinstance(x, list) else [140, 140, 140, 120]
    )
    return json.loads(gdf.to_json())


@st.cache_data
def load_data(region_dir: Path):
    twin_dir = region_dir / "digital_twin"

    required_files = [
        twin_dir / "field_master.parquet",
        twin_dir / "field_state_biweekly.parquet",
        twin_dir / "field_alerts.parquet",
        twin_dir / "field_changes_latest.parquet",
        twin_dir / "field_master.gpkg",
    ]

    missing = [str(fp) for fp in required_files if not fp.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required digital twin files:\n" + "\n".join(missing)
        )

    field_master = pd.read_parquet(twin_dir / "field_master.parquet")
    field_state = pd.read_parquet(twin_dir / "field_state_biweekly.parquet")
    field_alerts = pd.read_parquet(twin_dir / "field_alerts.parquet")
    field_changes = pd.read_parquet(twin_dir / "field_changes_latest.parquet")
    fields_gdf = gpd.read_file(twin_dir / "field_master.gpkg")

    if fields_gdf.crs is None:
        fields_gdf = fields_gdf.set_crs("EPSG:4326")
    else:
        fields_gdf = fields_gdf.to_crs("EPSG:4326")

    # Parse dates safely if present
    for df in [field_master, field_state, field_alerts, field_changes]:
        for c in ["bin_start", "latest_bin_start", "first_active_date", "peak_date", "last_active_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    return field_master, field_state, field_alerts, field_changes, fields_gdf


def safe_metric(df: pd.DataFrame, col: str, condition_values: list[str]) -> int | str:
    if col not in df.columns:
        return "NA"
    return int(df[col].isin(condition_values).sum())


# ============================================================
# LOAD
# ============================================================

st.sidebar.markdown("## Region")

regions = get_region_dirs(ROOT_DIR)
if not regions:
    st.error(
        f"No region folders found under {ROOT_DIR}. "
        "Make sure your repo contains data/<RegionName>/digital_twin/..."
    )
    st.stop()

region_name = st.sidebar.selectbox("Region", [r.name for r in regions], index=0)
region_dir = ROOT_DIR / region_name

try:
    field_master, field_state, field_alerts, field_changes, fields_gdf = load_data(region_dir)
except Exception as e:
    st.error(f"Failed to load region data: {e}")
    st.stop()

try:
    field_id_col = get_field_id_col(field_master)
except Exception as e:
    st.error(f"Failed to detect field id column: {e}")
    st.stop()

# Normalize ids as strings
field_master[field_id_col] = field_master[field_id_col].astype(str)
field_state[field_id_col] = field_state[field_id_col].astype(str)
field_alerts[field_id_col] = field_alerts[field_id_col].astype(str)
field_changes[field_id_col] = field_changes[field_id_col].astype(str)
fields_gdf[field_id_col] = fields_gdf[field_id_col].astype(str)

# Bring non-geometry fields into map layer if missing
missing_cols = [c for c in field_master.columns if c not in fields_gdf.columns]
if missing_cols:
    fields_gdf = fields_gdf.merge(
        field_master[[field_id_col] + missing_cols],
        on=field_id_col,
        how="left",
)


# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("## Filters")

crop_values = []
if "field_crop_final" in field_master.columns:
    crop_values = sorted(field_master["field_crop_final"].dropna().astype(str).unique().tolist())
crop_filter = st.sidebar.selectbox("Crop", ["All"] + crop_values, index=0)

health_values = []
if "health_label" in field_master.columns:
    health_values = sorted(field_master["health_label"].dropna().astype(str).unique().tolist())
health_filter = st.sidebar.selectbox("Health", ["All"] + health_values, index=0)

# Apply filters
df = field_master.copy()

if crop_filter != "All" and "field_crop_final" in df.columns:
    df = df[df["field_crop_final"].astype(str) == crop_filter].copy()

if health_filter != "All" and "health_label" in df.columns:
    df = df[df["health_label"].astype(str) == health_filter].copy()

ids = set(df[field_id_col].astype(str))
fields_f = fields_gdf[fields_gdf[field_id_col].astype(str).isin(ids)].copy()


# ============================================================
# HEADER
# ============================================================

st.title("Kirsh Digital Twin")
st.caption(f"Region: {region_name}")

c1, c2, c3 = st.columns(3)

c1.metric("Fields", len(df))
c2.metric("Alerts", safe_metric(df, "alert_level", ["watch", "moderate", "severe"]))
c3.metric("Weak Fields", safe_metric(df, "health_label", ["weak", "critical"]))


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(["Map", "Field Analytics", "Alerts", "QA"])


# ============================================================
# MAP
# ============================================================

with tabs[0]:
    st.subheader("Field Map")

    if fields_f.empty:
        st.info("No fields match the current filters.")
    else:
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
            zoom=9,
            pitch=0,
        )

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geo,
            get_fill_color="properties.color",
            get_line_color=[40, 40, 40, 120],
            line_width_min_pixels=1,
            pickable=True,
            stroked=True,
            filled=True,
        )

        tooltip = {
            "html": f"""
            <b>Field ID:</b> {{{field_id_col}}}<br/>
            <b>Crop:</b> {{field_crop_final}}<br/>
            <b>Health:</b> {{health_label}}<br/>
            <b>Alert:</b> {{alert_level}}<br/>
            <b>Score:</b> {{field_crop_prob_score}}<br/>
            <b>Area (ha):</b> {{area_ha_est}}
            """
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip=tooltip,
            )
        )


# ============================================================
# FIELD ANALYTICS
# ============================================================

with tabs[1]:
    st.subheader("Field Analytics")

    field_ids = df[field_id_col].astype(str).sort_values().tolist()
    if not field_ids:
        st.info("No fields available for the current filters.")
    else:
        selected = st.selectbox("Select Field", field_ids, index=0)

        state_f = field_state[field_state[field_id_col].astype(str) == selected].copy()

        if state_f.empty:
            st.info("No time-series records found for the selected field.")
        else:
            metrics = ["ndvi_mean", "ndre_mean", "health_score"]
            metrics = [m for m in metrics if m in state_f.columns]

            if not metrics:
                st.info("No analytics metrics found for the selected field.")
            else:
                plot_df = state_f.melt(
                    id_vars=["bin_start"],
                    value_vars=metrics,
                    var_name="variable",
                    value_name="value",
                )

                fig = px.line(
                    plot_df,
                    x="bin_start",
                    y="value",
                    color="variable",
                    markers=True,
                    title=f"Field {selected} Time Series",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(state_f.sort_values("bin_start", ascending=False).head(20), use_container_width=True)


# ============================================================
# ALERTS
# ============================================================

with tabs[2]:
    st.subheader("Alerts")

    if field_alerts.empty:
        st.info("No alert rows found.")
    else:
        st.dataframe(field_alerts.head(100), use_container_width=True)


# ============================================================
# QA
# ============================================================

with tabs[3]:
    st.subheader("QA")

    weak = df.copy()

    maj = weak["majority_ratio_known"] if "majority_ratio_known" in weak.columns else pd.Series(1.0, index=weak.index)
    score = weak["field_crop_prob_score"] if "field_crop_prob_score" in weak.columns else pd.Series(1.0, index=weak.index)

    weak = weak[
        (maj.fillna(1.0) < 0.6) |
        (score.fillna(1.0) < 0.6)
    ].copy()

    if weak.empty:
        st.success("No weak / low-confidence fields under the current filters.")
    else:
        st.dataframe(weak.head(50), use_container_width=True)
