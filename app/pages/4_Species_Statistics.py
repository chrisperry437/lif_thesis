from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PREDICTIONS_PATH = Path("./results/realtime/predictions.csv")

st.set_page_config(
    page_title="Species Statistics",
    page_icon="🧫",
    layout="wide",
)

st.title("🧫 Species-Specific Statistics")


SPECIES_INFO = {
    "B. cereus": {
        "full_name": "Bacillus cereus",
        "genus": "Bacillus",
        "family": "Bacillaceae",
        "phylum": "Firmicutes",
        "gram_status": "Gram-positive",
        "shape": "Rod-shaped",
        "notes": (
            "Common environmental bacterium found in soil, dust, and air. "
            "Some strains are associated with foodborne illness."
        ),
    },
    "B. endophyticus": {
        "full_name": "Bacillus endophyticus",
        "genus": "Bacillus",
        "family": "Bacillaceae",
        "phylum": "Firmicutes",
        "gram_status": "Gram-positive",
        "shape": "Rod-shaped",
        "notes": (
            "Environmental Bacillus species originally associated with plant tissues. "
            "Likely relevant to outdoor and plant-associated bioaerosols."
        ),
    },
    "K. salsicia": {
        "full_name": "Kocuria salsicia",
        "genus": "Kocuria",
        "family": "Micrococcaceae",
        "phylum": "Actinobacteria",
        "gram_status": "Gram-positive",
        "shape": "Coccus-shaped",
        "notes": (
            "Environmental and skin-associated bacterium. Members of Kocuria are often "
            "found in air, soil, water, and human-associated environments."
        ),
    },
    "M. luteus": {
        "full_name": "Micrococcus luteus",
        "genus": "Micrococcus",
        "family": "Micrococcaceae",
        "phylum": "Actinobacteria",
        "gram_status": "Gram-positive",
        "shape": "Coccus-shaped",
        "notes": (
            "Common airborne and skin-associated bacterium. Frequently used as a "
            "representative environmental micrococcus."
        ),
    },
    "S. huminis": {
        "full_name": "Staphylococcus hominis",
        "genus": "Staphylococcus",
        "family": "Staphylococcaceae",
        "phylum": "Firmicutes",
        "gram_status": "Gram-positive",
        "shape": "Coccus-shaped",
        "notes": (
            "Human skin-associated Staphylococcus species. Its presence in aerosols may "
            "reflect human-associated indoor or urban bioaerosol sources."
        ),
    },
}


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
    "predicted_label",
    "prediction_confidence",
    "size",
    "time_asymmetry",
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()


st.sidebar.header("Filters")

selected_species = st.sidebar.selectbox(
    "Species",
    ["All"] + sorted(df["predicted_label"].dropna().unique().tolist()),
    key="species_stats_species_filter",
)

min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    key="species_stats_min_confidence",
)

filtered = df[df["prediction_confidence"] >= min_confidence].copy()

if selected_species != "All":
    filtered = filtered[filtered["predicted_label"] == selected_species].copy()

if filtered.empty:
    st.warning("No particles match the selected filters.")
    st.stop()


st.subheader("Species Summary Table")

species_stats = (
    filtered
    .groupby("predicted_label")
    .agg(
        particle_count=("predicted_label", "size"),
        mean_confidence=("prediction_confidence", "mean"),
        median_confidence=("prediction_confidence", "median"),
        mean_size_um=("size", "mean"),
        median_size_um=("size", "median"),
        mean_time_asymmetry=("time_asymmetry", "mean"),
        median_time_asymmetry=("time_asymmetry", "median"),
    )
    .reset_index()
)

total_particles = species_stats["particle_count"].sum()
species_stats["percentage_of_filtered_sample"] = (
    species_stats["particle_count"] / total_particles * 100
)

st.dataframe(
    species_stats.sort_values("particle_count", ascending=False).style.format(
        {
            "mean_confidence": "{:.2%}",
            "median_confidence": "{:.2%}",
            "mean_size_um": "{:.2f}",
            "median_size_um": "{:.2f}",
            "mean_time_asymmetry": "{:.3f}",
            "median_time_asymmetry": "{:.3f}",
            "percentage_of_filtered_sample": "{:.2f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


st.divider()

st.subheader("Species Composition")

fig = px.bar(
    species_stats.sort_values("particle_count", ascending=False),
    x="predicted_label",
    y="percentage_of_filtered_sample",
    text=species_stats["percentage_of_filtered_sample"].round(1),
    title="Predicted species composition",
)

fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_yaxes(title="Percentage", range=[0, 100])
fig.update_xaxes(title="Predicted species")

st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Biological Information")

species_to_show = (
    sorted(filtered["predicted_label"].dropna().unique().tolist())
    if selected_species == "All"
    else [selected_species]
)

for species in species_to_show:
    info = SPECIES_INFO.get(species)

    if info is None:
        continue

    with st.expander(f"{species} — {info['full_name']}", expanded=(selected_species != "All")):
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Phylum", info["phylum"])
        col2.metric("Family", info["family"])
        col3.metric("Gram status", info["gram_status"])
        col4.metric("Shape", info["shape"])

        st.write(info["notes"])


st.divider()

st.subheader("Species Profile Distributions")

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        filtered,
        x="predicted_label",
        y="size",
        color="predicted_label",
        title="Particle size by predicted species",
    )

    fig.update_xaxes(title="Predicted species")
    fig.update_yaxes(title="Size (µm)")

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        filtered,
        x="predicted_label",
        y="prediction_confidence",
        color="predicted_label",
        title="Prediction confidence by predicted species",
    )

    fig.update_xaxes(title="Predicted species")
    fig.update_yaxes(title="Confidence", range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        filtered,
        x="size",
        color="predicted_label",
        nbins=40,
        title="Size distribution by species",
    )

    fig.update_xaxes(title="Size (µm)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        filtered,
        x="time_asymmetry",
        color="predicted_label",
        nbins=40,
        title="Time asymmetry distribution by species",
    )

    fig.update_xaxes(title="Time asymmetry")
    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("Species Probability Columns")

prob_cols = [
    "prob_B_cereus",
    "prob_B_endophyticus",
    "prob_K_salsicia",
    "prob_M_luteus",
    "prob_S_huminis",
]

available_prob_cols = [col for col in prob_cols if col in filtered.columns]

if available_prob_cols:
    prob_summary = (
        filtered
        .groupby("predicted_label")[available_prob_cols]
        .mean()
        .reset_index()
    )

    prob_long = prob_summary.melt(
        id_vars="predicted_label",
        value_vars=available_prob_cols,
        var_name="probability_column",
        value_name="mean_probability",
    )

    prob_long["probability_column"] = (
        prob_long["probability_column"]
        .str.replace("prob_", "", regex=False)
        .str.replace("_", " ")
    )

    fig = px.bar(
        prob_long,
        x="predicted_label",
        y="mean_probability",
        color="probability_column",
        barmode="group",
        title="Mean class probabilities by predicted species",
    )

    fig.update_yaxes(title="Mean probability", range=[0, 1])
    fig.update_xaxes(title="Predicted species")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No probability columns found.")


st.divider()

st.subheader("Recent Particles for Selected View")

display_cols = [
    "timestamp",
    "raw_file",
    "particle_index",
    "size",
    "time_asymmetry",
    "predicted_label",
    "prediction_confidence",
]

available_display_cols = [col for col in display_cols if col in filtered.columns]

st.dataframe(
    filtered[available_display_cols]
    .tail(200)
    .sort_index(ascending=False),
    use_container_width=True,
    hide_index=True,
)