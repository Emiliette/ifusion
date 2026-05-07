import pandas as pd

file_path = "fusionnet_metrics_atlas.csv"
df = pd.read_csv(file_path)

# Choose columns from E to O
cols_E_to_O = df.iloc[:, 4:15]  

# Calculate mean of each column
column_means = cols_E_to_O.mean()

print("Mean from columns E to O:")
print(column_means)