from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PREDICTIONS_PATH = Path("./results/realtime/predictions.csv")

st.set_page_config(
    page_title="Raw File Explorer",
    page_icon="📁",
    layout="wide",
)

st.title("📁 Raw File Explorer")


@st.cache_data(ttl=2)
def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    for col in ["processed_at", "event_time", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


df = load_predictions(PREDICTIONS_PATH)

if df.empty:
    st.warning(f"No predictions found at: {PREDICTIONS_PATH}")
    st.stop()


required_cols = [
    "raw_file",
    "predicted_label",
    "prediction_confidence",
    "size",
    "time_asymmetry",
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()


st.sidebar.header("Raw File Filters")

raw_summary = (
    df.groupby("raw_file")
    .agg(
        particle_count=("raw_file", "size"),
        mean_confidence=("prediction_confidence", "mean"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
    .reset_index()
    .sort_values("last_timestamp", ascending=False)
)

raw_file_options = raw_summary["raw_file"].tolist()

selected_raw_file = st.sidebar.selectbox(
    "Select raw file",
    raw_file_options,
    key="raw_file_explorer_selected_file",
)

selected_df = df[df["raw_file"] == selected_raw_file].copy()

if selected_df.empty:
    st.warning("No particles found for selected raw file.")
    st.stop()


st.subheader("Raw File Overview")
st.code(selected_raw_file)

particle_count = len(selected_df)
mean_confidence = selected_df["prediction_confidence"].mean()
median_confidence = selected_df["prediction_confidence"].median()
mean_size = selected_df["size"].mean()

time_col = "timestamp" if "timestamp" in selected_df.columns else None

if time_col:
    valid_time = selected_df.dropna(subset=[time_col])
    start_time = valid_time[time_col].min() if not valid_time.empty else None
    end_time = valid_time[time_col].max() if not valid_time.empty else None
else:
    start_time = None
    end_time = None

col1, col2, col3, col4 = st.columns(4)

col1.metric("Particle count", f"{particle_count:,}")
col2.metric("Mean confidence", f"{mean_confidence:.1%}")
col3.metric("Median confidence", f"{median_confidence:.1%}")
col4.metric("Mean size", f"{mean_size:.2f} µm")

if start_time is not None and end_time is not None:
    col1, col2 = st.columns(2)
    col1.info(f"Start time: {start_time}")
    col2.info(f"End time: {end_time}")


st.divider()

st.subheader("Raw File Composition")

composition = (
    selected_df["predicted_label"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)

composition.columns = ["predicted_label", "percentage"]

composition_counts = (
    selected_df["predicted_label"]
    .value_counts()
    .reset_index()
)

composition_counts.columns = ["predicted_label", "count"]

composition = composition.merge(composition_counts, on="predicted_label")

col1, col2 = st.columns([1, 2])

with col1:
    dominant = composition.iloc[0]

    st.metric(
        "Dominant species",
        dominant["predicted_label"],
        f"{dominant['percentage']:.1f}%",
    )

    st.dataframe(
        composition.assign(percentage=composition["percentage"].round(2)),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    fig = px.bar(
        composition,
        x="predicted_label",
        y="percentage",
        text=composition["percentage"].round(1),
        title="Predicted composition for selected raw file",
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_yaxes(title="Percentage", range=[0, 100])
    fig.update_xaxes(title="Predicted species")

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Confidence Profile")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        selected_df,
        x="prediction_confidence",
        nbins=30,
        color="predicted_label",
        title="Confidence distribution by species",
    )

    fig.update_xaxes(title="Prediction confidence")
    fig.update_yaxes(title="Particle count")

    st.plotly_chart(fig, use_container_width=True)

with col2:
    species_conf = (
        selected_df
        .groupby("predicted_label")
        .agg(
            mean_confidence=("prediction_confidence", "mean"),
            median_confidence=("prediction_confidence", "median"),
            particle_count=("predicted_label", "size"),
        )
        .reset_index()
        .sort_values("mean_confidence", ascending=False)
    )

    fig = px.bar(
        species_conf,
        x="predicted_label",
        y="mean_confidence",
        text=species_conf["mean_confidence"].round(3),
        title="Mean confidence by predicted species",
    )

    fig.update_yaxes(title="Mean confidence", range=[0, 1])
    fig.update_xaxes(title="Predicted species")

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Raw File Timeline")

if "timestamp" not in selected_df.columns or selected_df["timestamp"].isna().all():
    st.info("No valid timestamp column available for this raw file.")
else:
    time_bin = st.selectbox(
        "Timeline bin",
        ["1s", "5s", "10s", "30s", "1min"],
        index=2,
        key="raw_file_timeline_bin",
    )

    timeline = selected_df.dropna(subset=["timestamp"]).copy()
    timeline["time_bin"] = timeline["timestamp"].dt.floor(time_bin)

    timeline_counts = (
        timeline
        .groupby(["time_bin", "predicted_label"])
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        timeline_counts,
        x="time_bin",
        y="count",
        color="predicted_label",
        markers=True,
        title="Particle predictions over time within raw file",
    )

    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Particle count")

    st.plotly_chart(fig, use_container_width=True)

    timeline_counts["total_in_bin"] = (
        timeline_counts
        .groupby("time_bin")["count"]
        .transform("sum")
    )

    timeline_counts["percentage"] = (
        timeline_counts["count"]
        / timeline_counts["total_in_bin"]
        * 100
    )

    fig = px.line(
        timeline_counts,
        x="time_bin",
        y="percentage",
        color="predicted_label",
        markers=True,
        title="Composition percentage over time within raw file",
    )

    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Percentage", range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Particle Profile Summary")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        selected_df,
        x="size",
        color="predicted_label",
        nbins=40,
        title="Particle size distribution",
    )

    fig.update_xaxes(title="Size (µm)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        selected_df,
        x="size",
        y="time_asymmetry",
        color="predicted_label",
        hover_data=[
            "particle_index",
            "prediction_confidence",
        ],
        title="Size vs time asymmetry",
    )

    fig.update_xaxes(title="Size (µm)")
    fig.update_yaxes(title="Time asymmetry")

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Raw File Table")

display_cols = [
    "timestamp",
    "raw_file",
    "particle_index",
    "size",
    "time_asymmetry",
    "predicted_label",
    "prediction_confidence",
]

available_display_cols = [col for col in display_cols if col in selected_df.columns]

st.dataframe(
    selected_df[available_display_cols].sort_values(
        "particle_index",
        ascending=True,
    ),
    use_container_width=True,
    hide_index=True,
)