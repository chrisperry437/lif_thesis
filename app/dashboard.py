from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh


PREDICTIONS_PATH = Path("./results/realtime/predictions.csv")

st.set_page_config(
    page_title="Rapid-E Bioaerosol Dashboard",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 Rapid-E Bioaerosol Dashboard")

refresh_seconds = st.sidebar.slider(
    "Refresh every",
    min_value=1,
    max_value=30,
    value=5,
    key="home_refresh_seconds",
)

st_autorefresh(interval=refresh_seconds * 1000, key="home_refresh")


@st.cache_data(ttl=2)
def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "processed_at" in df.columns:
        df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")

    return df


df = load_predictions(PREDICTIONS_PATH)

if df.empty:
    st.warning(f"No predictions found at: {PREDICTIONS_PATH}")
    st.stop()


time_col = "timestamp" if "timestamp" in df.columns else "processed_at"

df = df.dropna(subset=[time_col]).copy()

if df.empty:
    st.warning("Predictions were found, but no valid timestamps are available.")
    st.stop()


latest_time = df[time_col].max()

last_minute = df[df[time_col] >= latest_time - pd.Timedelta(minutes=1)]
last_hour = df[df[time_col] >= latest_time - pd.Timedelta(hours=1)]

particles_per_minute = len(last_minute)
particles_per_hour = len(last_hour)

current_throughput = (
    particles_per_minute / 60
    if particles_per_minute > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total particles", f"{len(df):,}")
col2.metric("Particles / minute", f"{particles_per_minute:,}")
col3.metric("Particles / hour", f"{particles_per_hour:,}")
col4.metric("Current throughput", f"{current_throughput:.1f} particles/sec")


st.divider()

st.subheader("Particle Count Over Time")

rate_bin = st.selectbox(
    "Time bin",
    ["10s", "30s", "1min", "5min"],
    index=2,
    key="home_particle_rate_bin",
)

rate_df = df.copy()
rate_df["time_bin"] = rate_df[time_col].dt.floor(rate_bin)

particle_counts = (
    rate_df
    .groupby("time_bin")
    .size()
    .reset_index(name="particle_count")
)

fig = px.line(
    particle_counts,
    x="time_bin",
    y="particle_count",
    markers=True,
    title=f"Particle count per {rate_bin}",
)

fig.update_xaxes(title="Time")
fig.update_yaxes(title="Particle count")

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Live Sample Composition Trend")

composition_bin = st.selectbox(
    "Composition time bin",
    ["10s", "30s", "1min", "5min"],
    index=2,
    key="home_composition_bin",
)

composition_df = df.dropna(subset=[time_col]).copy()
composition_df["time_bin"] = composition_df[time_col].dt.floor(composition_bin)

composition_counts = (
    composition_df
    .groupby(["time_bin", "predicted_label"])
    .size()
    .reset_index(name="count")
)

composition_counts["total_in_bin"] = (
    composition_counts
    .groupby("time_bin")["count"]
    .transform("sum")
)

composition_counts["percentage"] = (
    composition_counts["count"]
    / composition_counts["total_in_bin"]
    * 100
)

fig = px.line(
    composition_counts,
    x="time_bin",
    y="percentage",
    color="predicted_label",
    markers=True,
    title=f"Live sample composition per {composition_bin}",
)

fig.update_yaxes(title="Percentage", range=[0, 100])
fig.update_xaxes(title="Time")

st.plotly_chart(fig, use_container_width=True)

st.caption("Use the sidebar to open Live Predictions, Model Selection, or Particle Profiles.")