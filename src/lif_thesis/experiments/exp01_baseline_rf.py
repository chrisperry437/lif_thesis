##Reproduction of originla Random forest Baseline

from pathlib import Path

import pandas as pd

from lif_thesis.models.rf import run_baseline_rf_experiment


def main():
    data_path = Path("data/processed/bacterial_samples.parquet")

    df = pd.read_parquet(data_path)

    run_baseline_rf_experiment(
        df=df,
        label_col="label",
        group_col="raw_file",
        spectra_col="spectrometer",
        output_dir="results/baseline_rf",
        random_state=42,
    )


if __name__ == "__main__":
    main()