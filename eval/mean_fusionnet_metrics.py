import pandas as pd

FILE_PATH = "fusionnet_metrics_em5_allmods.csv"


def compute_means(frame: pd.DataFrame, label: str, metric_columns: list[str]) -> None:
    column_means = frame[metric_columns].mean(numeric_only=True)
    print(f"\nMean columns E to O for {label}:")
    print(column_means)


df = pd.read_csv(FILE_PATH)

# Columns E to O correspond to indices 4 through 14.
metric_columns = df.columns[4:15].tolist()

for k_value in [2, 3, 4]:
    compute_means(df[df["K"] == k_value], f"K = {k_value}", metric_columns)

compute_means(df, "all rows", metric_columns)
