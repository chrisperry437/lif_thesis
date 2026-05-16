##Data preprocessing pipeline
from lif_thesis.data.loaders import process_raw_dataset


if __name__ == "__main__":
    process_raw_dataset(
        keep_thresholds=False,
        extra_params=True,
        overwrite=True,
    )