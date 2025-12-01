import pandas as pd
import glob

# Step 1: Read multiple CSV files
# Assuming all five CSVs are stored in a folder called 'data' and each has a column 'Date' and 'Return'
file_paths = glob.glob("*.csv")  # adjust path if needed

# Step 2: Load them into a dictionary of DataFrames
dfs = {}
for i, file in enumerate(file_paths, start=1):
    df = pd.read_csv(file, index_col=0)
    if file != "SOL-AUD.csv":
        df = df[101:]
    dfs[f"{file}"] = df

# Step 3: Concatenate returns into one DataFrame
returns = pd.concat([df['daily_return'] for df in dfs.values()], axis=1)
returns.columns = list(dfs.keys())

# Step 4: Descriptive statistics
desc_stats = returns.describe()  # mean, std, min, max, quartiles
print("Basic Descriptive Statistics:\n", desc_stats)

# Step 5: Additional metrics (skewness, kurtosis)
extra_stats = pd.DataFrame({
    'Skewness': returns.skew(),
    'Kurtosis': returns.kurt()
})
print("\nSkewness & Kurtosis:\n", extra_stats)

# Step 6: Correlation matrix
correlation = returns.dropna().corr()
print("\nCorrelation Matrix:\n", correlation)

desc_stats.to_csv("descriptive_stats.csv")
extra_stats.to_csv("extra_stats.csv")
correlation.to_csv("correlation_matrix.csv")
