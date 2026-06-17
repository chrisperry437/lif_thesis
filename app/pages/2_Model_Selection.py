import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


MODELS_DIR = Path("models/trained")
CONFIG_PATH = Path("configs/active_model.json")

st.set_page_config(
    page_title="Model Selection",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Model Selection")


def load_active_model() -> str | None:
    if not CONFIG_PATH.exists():
        return None

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("active_model")


def save_active_model(model_name: str) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump({"active_model": model_name}, f, indent=2)


model_files = sorted(
    [
        p for p in MODELS_DIR.iterdir()
        if p.is_file() and p.suffix in [".pt", ".joblib", ".pkl"]
    ]
)

model_dirs = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()])
available_models = model_dirs + model_files

if not available_models:
    st.warning(f"No models found in {MODELS_DIR}")
    st.stop()


model_names = [p.name for p in available_models]
active_model = load_active_model()

default_index = model_names.index(active_model) if active_model in model_names else 0

selected_model = st.selectbox(
    "Select active model",
    model_names,
    index=default_index,
    key="active_model_selector",
)

if st.button("Set active model", key="set_active_model_button"):
    save_active_model(selected_model)
    st.success(f"Active model updated to: {selected_model}")

st.subheader("Current Active Model")
st.code(load_active_model() or "No active model selected")


st.divider()

st.subheader("Model Comparison")

comparison_df = pd.DataFrame(
    [
        {
            "Model": "Paper RF",
            "Experiment": "exp00",
            "Accuracy": 0.807,
            "Balanced Accuracy": 0.654,
            "Macro F1": 0.641,
        },
        {
            "Model": "Tuned RF",
            "Experiment": "exp03",
            "Accuracy": 0.853,
            "Balanced Accuracy": 0.716,
            "Macro F1": 0.721,
        },
        {
            "Model": "Baseline CNN",
            "Experiment": "exp04",
            "Accuracy": 0.552,
            "Balanced Accuracy": 0.385,
            "Macro F1": 0.363,
        },
        {
            "Model": "Multimodal Deep Learning",
            "Experiment": "exp05",
            "Accuracy": 0.876,
            "Balanced Accuracy": 0.769,
            "Macro F1": 0.769,
        },
    ]
)

st.dataframe(
    comparison_df.style.format(
        {
            "Accuracy": "{:.3f}",
            "Balanced Accuracy": "{:.3f}",
            "Macro F1": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

metrics_long = comparison_df.melt(
    id_vars=["Model", "Experiment"],
    value_vars=["Accuracy", "Balanced Accuracy", "Macro F1"],
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    metrics_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Model performance comparison",
)

fig.update_yaxes(title="Score", range=[0, 1])
fig.update_xaxes(title="Model")

st.plotly_chart(fig, use_container_width=True)


best_row = comparison_df.sort_values(
    "Balanced Accuracy",
    ascending=False,
).iloc[0]

st.success(
    f"Best overall model by balanced accuracy: "
    f"{best_row['Model']} ({best_row['Balanced Accuracy']:.3f})"
)


st.divider()

st.subheader("Available Model Artifacts")

for path in available_models:
    st.write(f"**{path.name}**")
    st.caption(str(path))

    metadata_path = path / "metadata.json" if path.is_dir() else None

    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        with st.expander(f"Metadata: {path.name}"):
            st.json(metadata)