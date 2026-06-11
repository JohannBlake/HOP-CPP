import pandas as pd
import numpy as np

# Define the file path
csv_file_path = 'ablation_studies_evaluation_dataset copy.csv'

# Read the CSV file
try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Filter by ablation_study_name
target_ablation = "With HAM, OPTV, visit frequency (Radiation)"
df_filtered = df[df['ablation_study_name'] == target_ablation]

# Filter by epoch
target_epoch = "epoch-100"
df_filtered = df_filtered[df_filtered['epoch'] == target_epoch]

if df_filtered.empty:
    print(f"No data found for ablation study '{target_ablation}' and epoch '{target_epoch}'.")
    exit(0)

# Calculate ratio c = "episode steps" / "path length"
# steps_end / path_length_end
df_filtered['c_ratio'] = df_filtered['steps_end'] / df_filtered['path_length_end']

# Statistics of c
stats = df_filtered['c_ratio'].describe()
mean_c = df_filtered['c_ratio'].mean()
median_c = df_filtered['c_ratio'].median()
std_c = df_filtered['c_ratio'].std()

print(f"Statistics for ratio c = steps_end / path_length_end:")
print(stats)
print(f"\nMean c: {mean_c}")
print(f"Median c: {median_c}")

# Calculate percentage within 0.1% of mean_c
tolerance_pct = 0.1 # 0.1%
tolerance = tolerance_pct / 100.0

def check_percentage(center_value):
    lower_bound = center_value * (1 - tolerance)
    upper_bound = center_value * (1 + tolerance)
    within = df_filtered[(df_filtered['c_ratio'] >= lower_bound) & (df_filtered['c_ratio'] <= upper_bound)]
    return (len(within) / len(df_filtered)) * 100

perc_mean = check_percentage(mean_c)
print(f"\nUsing Mean c = {mean_c:.6f}:")
print(f"Percentage of runs where steps = c * path_length +- 0.1%: {perc_mean:.2f}%")

perc_median = check_percentage(median_c)
print(f"\nUsing Median c = {median_c:.6f}:")
print(f"Percentage of runs where steps = c * path_length +- 0.1%: {perc_median:.2f}%")

# Check if there is ANY c that gives a high percentage?
# We can sweep c from min to max
c_values = np.linspace(df_filtered['c_ratio'].min(), df_filtered['c_ratio'].max(), 1000)
best_c = 0
best_perc = 0
for c in c_values:
    p = check_percentage(c)
    if p > best_perc:
        best_perc = p
        best_c = c

print(f"\nBest found c (maximizing percentage): {best_c:.6f}")
print(f"Percentage: {best_perc:.2f}%")

print("\nTop 5 rows:")
print(df_filtered[['map', 'steps_end', 'path_length_end', 'c_ratio']].head())

# Also check for "path_length = c * episode steps" (inverse) just in case
df_filtered['inv_c_ratio'] = df_filtered['path_length_end'] / df_filtered['steps_end']
mean_inv = df_filtered['inv_c_ratio'].mean()
median_inv = df_filtered['inv_c_ratio'].median()
print(f"\nInverse Ratio (Path / Steps): Mean={mean_inv:.6f}, Median={median_inv:.6f}")
