from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Kirsh Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data/Tel_Aviv/digital_twin")
FIELD_ID = "id"


# ============================================================
# HELPERS
# ============================================================

def safe_values(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def safe_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def choose_metric_cols(df: pd.DataFrame) -> list[str]:
    preferred = [
        "ndvi_mean",
        "ndre_mean",
        "evi_mean",
        "lai_mean",
        "confidence_mean",
        "margin_mean",
    ]
    return [c for c in preferred if c in df.columns]


def crop_color(crop: str) -> list[int]:
    crop = str(crop).lower()
    if crop == "carrot":
        return [255, 140, 0, 190]   # orange
    if crop == "potato":
        return [139, 69, 19, 190]   # brown
    if crop == "wheat":
        return [34, 139, 34, 190]   # green
    return [160, 160, 160, 160]     # gray


def health_color(label: str) -> list[int]:
    label = str(label).lower()
    if label == "strong":
        return [46, 125, 50, 190]
    if label == "moderate":
        return [251, 192, 45, 190]
    if label == "weak":
        return [244, 124, 32, 190]
    if label == "critical":
        return [198, 40, 40, 190]
    return [160, 160, 160, 160]


def alert_color(label: str) -> list[int]:
    label = str(label).lower()
    if label == "normal":
        return [76, 175, 80, 190]
    if label == "watch":
        return [255, 193, 7, 190]
    if label == "moderate":
        return [255, 112, 67, 190]
    if label == "severe":
        return [211, 47, 47, 190]
    return [160, 160, 160, 160]


def add_color_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    
    if mode == "Crop":
        out["color"] = out["field_crop_final"].apply(crop_color)
    elif mode == "Health":
        if "health_label" in out.columns:
            out["color"] = out["health_label"].apply(health_color)
        else:
            out["color"] = [[160, 160, 160, 160]] * len(out)
    else:
        if "alert_level" in out.columns:
            out["color"] = out["alert_level"].apply(alert_color)
        else:
            out["color"] = [[160, 160, 160, 160]] * len(out)
    
    out[["r", "g", "b", "a"]] = pd.DataFrame(out["color"].tolist(), index=out.index)
    return out


@st.cache_data
def load_all():
    field_master = pd.read_parquet(DATA_DIR / "field_master.parquet")
    field_map = pd.read_parquet(DATA_DIR / "field_master_map.parquet")
    field_state = pd.read_parquet(DATA_DIR / "field_state_biweekly.parquet")
    field_alerts = pd.read_parquet(DATA_DIR / "field_alerts.parquet")
    field_changes = pd.read_parquet(DATA_DIR / "field_changes_latest.parquet")
    
    for df in [field_master, field_map, field_state, field_alerts, field_changes]:
        for c in [
            "bin_start",
            "latest_bin_start",
            "previous_bin_start",
            "first_active_date",
            "peak_date",
            "last_active_date",
        ]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    
    return field_master, field_map, field_state, field_alerts, field_changes


# ============================================================
# LOAD
# ============================================================

field_master, field_map, field_state, field_alerts, field_changes = load_all()

for df in [field_master, field_map, field_state, field_alerts, field_changes]:
    if FIELD_ID in df.columns:
        df[FIELD_ID] = df[FIELD_ID].astype(str)

# Keep only fields that exist in field_master
valid_ids = set(field_master[FIELD_ID].astype(str))
field_map = field_map[field_map[FIELD_ID].astype(str).isin(valid_ids)].copy()

# numeric coords
field_map["lon"] = pd.to_numeric(field_map["lon"], errors="coerce")
field_map["lat"] = pd.to_numeric(field_map["lat"], errors="coerce")
field_map = field_map.dropna(subset=["lon", "lat"]).copy()

# merge field summary into map if needed
missing_master_cols = [c for c in field_master.columns if c not in field_map.columns]
if missing_master_cols:
    field_map = field_map.merge(
        field_master[[FIELD_ID] + missing_master_cols],
        on=FIELD_ID,
        how="left",
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Kirsh Digital Twin")
st.sidebar.caption("Field intelligence workspace")

crop_filter = st.sidebar.selectbox(
    "Crop",
    ["All"] + safe_values(field_master, "field_crop_final"),
    index=0,
)

health_filter = st.sidebar.selectbox(
    "Health",
    ["All"] + safe_values(field_master, "health_label") if "health_label" in field_master.columns else ["All"],
    index=0,
)

alert_filter = st.sidebar.selectbox(
    "Alert",
    ["All"] + safe_values(field_master, "alert_level") if "alert_level" in field_master.columns else ["All"],
    index=0,
)

season_filter = st.sidebar.selectbox(
    "Season",
    ["All"] + safe_values(field_state, "season_year_inferred") if "season_year_inferred" in field_state.columns else ["All"],
    index=0,
)

stage_filter = st.sidebar.selectbox(
    "Stage",
    ["All"] + safe_values(field_state, "stage_name") if "stage_name" in field_state.columns else ["All"],
    index=0,
)

top_n = st.sidebar.slider("Top ranked fields", 10, 100, 25)

st.sidebar.markdown(
    """
### Crop Legend
- 🟧 Carrot 
- 🟫 Potato 
- 🟩 Wheat 
- ⬜ Unknown
"""
)

# Apply field-level filters
master_f = field_master.copy()

if crop_filter != "All":
    master_f = master_f[master_f["field_crop_final"].astype(str) == crop_filter].copy()

if health_filter != "All" and "health_label" in master_f.columns:
    master_f = master_f[master_f["health_label"].astype(str) == health_filter].copy()

if alert_filter != "All" and "alert_level" in master_f.columns:
    master_f = master_f[master_f["alert_level"].astype(str) == alert_filter].copy()

keep_ids = set(master_f[FIELD_ID].astype(str))

map_f = field_map[field_map[FIELD_ID].astype(str).isin(keep_ids)].copy()
state_f = field_state[field_state[FIELD_ID].astype(str).isin(keep_ids)].copy()
alerts_f = field_alerts[field_alerts[FIELD_ID].astype(str).isin(keep_ids)].copy()
changes_f = field_changes[field_changes[FIELD_ID].astype(str).isin(keep_ids)].copy()

if season_filter != "All" and "season_year_inferred" in state_f.columns:
    state_f = state_f[state_f["season_year_inferred"].astype(str) == season_filter].copy()

if stage_filter != "All" and "stage_name" in state_f.columns:
    state_f = state_f[state_f["stage_name"].astype(str) == stage_filter].copy()


# ============================================================
# HEADER
# ============================================================

st.title("Kirsh Digital Twin")
st.caption("Crop classification, field health, alerts, and seasonal monitoring")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fields", f"{len(master_f):,}")
k2.metric("Alerts", f"{len(alerts_f):,}")
k3.metric("Area (ha)", f"{safe_sum(master_f, 'area_ha_est'):,.1f}")
k4.metric(
    "Weak / Review",
    f"{int(((master_f['majority_ratio_known'].fillna(1) < 0.60) | (master_f['field_crop_prob_score'].fillna(1) < 0.60)).sum()):,}"
    if {"majority_ratio_known", "field_crop_prob_score"}.issubset(master_f.columns) else "NA"
)

tabs = st.tabs([
    "Overview",
    "Map",
    "Field Twin",
    "Alerts",
    "Change Detection",
    "Inspection Queue",
    "QA",
    "Debug",
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:
    a, b = st.columns(2)
    
    with a:
        crop_counts = (
            master_f["field_crop_final"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("crop")
            .reset_index(name="n_fields")
        )
        fig = px.bar(
            crop_counts,
            x="crop",
            y="n_fields",
            title="Fields by Crop",
            text_auto=True,
            color="crop",
            color_discrete_map={
                "carrot": "#ff8c00",
                "potato": "#8b4513",
                "wheat": "#228b22",
                "unknown": "#a0a0a0",
                "other": "#c8c8c8",
            },
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with b:
        if "health_label" in master_f.columns:
            health_counts = (
                master_f["health_label"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .rename_axis("health")
                .reset_index(name="n_fields")
            )
            fig = px.bar(
                health_counts,
                x="health",
                y="n_fields",
                title="Fields by Health",
                text_auto=True,
                color="health",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    c, d = st.columns(2)
    
    with c:
        area_by_crop = (
            master_f.groupby("field_crop_final", as_index=False)["area_ha_est"]
            .sum()
            .sort_values("area_ha_est", ascending=False)
        )
        fig = px.bar(
            area_by_crop,
            x="field_crop_final",
            y="area_ha_est",
            title="Area by Crop",
            text_auto=".1f",
            color="field_crop_final",
            color_discrete_map={
                "carrot": "#ff8c00",
                "potato": "#8b4513",
                "wheat": "#228b22",
                "unknown": "#a0a0a0",
                "other": "#c8c8c8",
            },
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with d:
        if "alert_level" in master_f.columns:
            alert_counts = (
                master_f["alert_level"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .rename_axis("alert")
                .reset_index(name="n_fields")
            )
            fig = px.bar(
                alert_counts,
                x="alert",
                y="n_fields",
                title="Alert Levels",
                text_auto=True,
                color="alert",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Inspection Queue")
    queue = master_f.sort_values("inspection_rank") if "inspection_rank" in master_f.columns else master_f.copy()
    queue_cols = [
        c for c in [
            "id",
            "field_crop_final",
            "health_label",
            "alert_level",
            "area_ha_est",
            "field_crop_prob_score",
            "majority_ratio_known",
            "inspection_priority_score",
            "inspection_rank",
        ] if c in queue.columns
    ]
    st.dataframe(queue[queue_cols].head(top_n), use_container_width=True)


# ============================================================
# MAP
# ============================================================

with tabs[1]:
    st.subheader("Field Map")
    
    if map_f.empty:
        st.warning("No map rows after filtering.")
    else:
        map_mode = st.radio("Map layer", ["Crop", "Health", "Alert"], horizontal=True)
        map_plot = add_color_columns(map_f, map_mode)
    
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_plot,
            get_position='[lon, lat]',
            get_radius=120,
            get_fill_color='[r, g, b, a]',
            get_line_color='[255, 255, 255, 220]',
            line_width_min_pixels=1,
            pickable=True,
            stroked=True,
            opacity=0.9,
        )
    
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=float(map_plot["lat"].mean()),
                longitude=float(map_plot["lon"].mean()),
                zoom=8,
                pitch=0,
            ),
            layers=[layer],
            tooltip={
                "html": """
                <b>Field ID:</b> {id}<br/>
                <b>Crop:</b> {field_crop_final}<br/>
                <b>Health:</b> {health_label}<br/>
                <b>Alert:</b> {alert_level}<br/>
                <b>Area (ha):</b> {area_ha_est}<br/>
                <b>Score:</b> {field_crop_prob_score}
                """
            },
        )
        st.pydeck_chart(deck)
    
        st.caption(f"Mapped fields: {len(map_plot):,}")


# ============================================================
# FIELD TWIN
# ============================================================

with tabs[2]:
    st.subheader("Field Twin")
    
    field_ids = master_f[FIELD_ID].astype(str).sort_values().tolist()
    selected_field = st.selectbox("Select field", field_ids, index=0)
    
    one_master = master_f[master_f[FIELD_ID].astype(str) == selected_field].copy()
    one_state = state_f[state_f[FIELD_ID].astype(str) == selected_field].copy()
    
    if not one_master.empty:
        row = one_master.iloc[0]
        c1, c2 = st.columns(2)
    
        with c1:
            st.markdown("### Field Summary")
            for c in ["id", "field_crop_final", "area_ha_est", "majority_ratio_known", "field_crop_prob_score"]:
                if c in row.index:
                    st.write(f"**{c}**: {row.get(c)}")
    
        with c2:
            st.markdown("### Twin Status")
            for c in ["health_label", "alert_level", "inspection_priority_score", "inspection_rank"]:
                if c in row.index:
                    st.write(f"**{c}**: {row.get(c)}")
    
    metric_cols = choose_metric_cols(one_state)
    if metric_cols and not one_state.empty:
        plot_df = one_state.melt(
            id_vars=[c for c in ["bin_start", "stage_name"] if c in one_state.columns],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        fig = px.line(
            plot_df,
            x="bin_start",
            y="value",
            facet_col="metric",
            facet_col_wrap=2,
            markers=True,
            title=f"Field {selected_field} Seasonal Signals",
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(height=720)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Recent Field Records")
    show_state_cols = [
        c for c in [
            "bin_start",
            "season_year_inferred",
            "season_progress",
            "stage_name",
            "ndvi_mean",
            "ndre_mean",
            "evi_mean",
            "lai_mean",
            "confidence_mean",
            "margin_mean",
            "maturity_flag",
            "harvest_flag",
        ] if c in one_state.columns
    ]
    st.dataframe(one_state[show_state_cols].head(50), use_container_width=True)


# ============================================================
# ALERTS
# ============================================================

with tabs[3]:
    st.subheader("Alerts")
    
    if not alerts_f.empty and "alert_level" in alerts_f.columns:
        alert_counts = (
            alerts_f["alert_level"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("alert")
            .reset_index(name="n_rows")
        )
        fig = px.bar(
            alert_counts,
            x="alert",
            y="n_rows",
            title="Alert Records by Level",
            text_auto=True,
            color="alert",
        )
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(alerts_f.head(200), use_container_width=True)


# ============================================================
# CHANGE DETECTION
# ============================================================

with tabs[4]:
    st.subheader("Change Detection")
    
    if not changes_f.empty and "ndvi_change" in changes_f.columns:
        fig = px.bar(
            changes_f.sort_values("ndvi_change").head(30),
            x="id",
            y="ndvi_change",
            color="field_crop_final" if "field_crop_final" in changes_f.columns else None,
            title="Largest Recent NDVI Declines",
            color_discrete_map={
                "carrot": "#ff8c00",
                "potato": "#8b4513",
                "wheat": "#228b22",
                "unknown": "#a0a0a0",
                "other": "#c8c8c8",
            },
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(changes_f.head(100), use_container_width=True)


# ============================================================
# INSPECTION QUEUE
# ============================================================

with tabs[5]:
    st.subheader("Inspection Queue")
    queue = master_f.sort_values("inspection_rank") if "inspection_rank" in master_f.columns else master_f.copy()
    queue_cols = [
        c for c in [
            "id",
            "field_crop_final",
            "health_label",
            "alert_level",
            "area_ha_est",
            "field_crop_prob_score",
            "majority_ratio_known",
            "inspection_priority_score",
            "inspection_rank",
        ] if c in queue.columns
    ]
    st.dataframe(queue[queue_cols].head(100), use_container_width=True)


# ============================================================
# QA
# ============================================================

with tabs[6]:
    st.subheader("QA")
    
    if {"majority_ratio_known", "field_crop_prob_score"}.issubset(master_f.columns):
        weak = master_f[
            (pd.to_numeric(master_f["majority_ratio_known"], errors="coerce").fillna(1) < 0.60) |
            (pd.to_numeric(master_f["field_crop_prob_score"], errors="coerce").fillna(1) < 0.60)
        ].copy()
    
        qa_cols = [
            c for c in [
                "id",
                "field_crop_final",
                "majority_ratio_known",
                "field_crop_prob_score",
                "unknown_share",
                "area_ha_est",
                "health_label",
                "alert_level",
            ] if c in weak.columns
        ]
        st.dataframe(weak[qa_cols].head(100), use_container_width=True)


# ============================================================
# DEBUG
# ============================================================

with tabs[7]:
    st.subheader("Debug")
    st.write("field_master shape:", field_master.shape)
    st.write("field_map shape:", field_map.shape)
    st.write("field_state shape:", field_state.shape)
    st.write("field_alerts shape:", field_alerts.shape)
    st.write("field_changes shape:", field_changes.shape)
    
    st.write("field_master columns:", list(field_master.columns))
    st.write("field_map columns:", list(field_map.columns))
    st.write("field_state columns:", list(field_state.columns))
