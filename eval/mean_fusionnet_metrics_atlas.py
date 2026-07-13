from pathlib import Path

import pandas as pd

FILE_PATH = Path("fusionnet_metrics_atlas.csv")
START_COLUMN_INDEX = 4
END_COLUMN_INDEX = 14


def main() -> None:
    df = pd.read_csv(FILE_PATH)

    # Excel columns E to O correspond to zero-based indices 4 through 14.
    metric_columns = df.columns[START_COLUMN_INDEX : END_COLUMN_INDEX + 1].tolist()
    mean_values = df[metric_columns].apply(pd.to_numeric, errors="coerce").mean()

    print(f"File: {FILE_PATH}")
    print("Mean of columns E to O:")
    print(mean_values.to_string())


if __name__ == "__main__":
    main()
