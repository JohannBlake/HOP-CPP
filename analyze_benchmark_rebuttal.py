import pandas as pd
import os

# Define file paths
files = [
    r"c:\Users\johan\programming\HOP-CPP (State Of Sweep 59etj5gd)\Simulation\misc\logs\benchmark_results\rebuttal_loo_eval.csv",
    r"c:\Users\johan\programming\HOP-CPP (State Of Sweep 59etj5gd)\Simulation\misc\logs\benchmark_results\rebuttal_noise_study.csv"
]

data_frames = []

for file_path in files:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            data_frames.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

if not data_frames:
    print("No data loaded.")
    exit()

# Concatenate dataframes
full_df = pd.concat(data_frames, ignore_index=True)

# Process 'epoch' column to extract integer
# format is expected to be 'epoch-X'
try:
    full_df['epoch_num'] = full_df['epoch'].astype(str).str.replace('epoch-', '').astype(int)
except ValueError as e:
    print(f"Error parsing epoch column: {e}")
    # Display some problematic values
    print("Problematic epoch values:")
    print(full_df[~full_df['epoch'].astype(str).str.startswith('epoch-')]['epoch'].unique())
    exit()

# 1. Calc avg metrics grouped by ablation_study_name, epoch (avg over maps)
# We need to make sure we are averaging only numeric columns of interest.
numeric_cols = ['coverage_end', 'path_length_end']
# Ensure columns exist
missing_cols = [col for col in numeric_cols if col not in full_df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    exit()

# Group by ablation_study_name and epoch_num, then calculate mean
grouped_df = full_df.groupby(['ablation_study_name', 'epoch_num'])[numeric_cols].mean().reset_index()

# 2. Filter for epochs 91 to 100
filtered_df = grouped_df[(grouped_df['epoch_num'] >= 10) & (grouped_df['epoch_num'] <= 12)]

if filtered_df.empty:
    print("No data found for epochs 91-100.")
    # Print max epoch to help debug
    print(f"Max epoch found: {grouped_df['epoch_num'].max()}")
    exit()

# 3. Calc avg of the avg metrics with std over the filtered epochs
final_stats = filtered_df.groupby('ablation_study_name')[numeric_cols].agg(['mean', 'std'])

# Report results
print("\nResults (Avg +/- Std over epochs 91-100):")
print("-" * 60)

for study_name in final_stats.index:
    print(f"Ablation Study: {study_name}")
    for metric in numeric_cols:
        mean_val = final_stats.loc[study_name, (metric, 'mean')]
        std_val = final_stats.loc[study_name, (metric, 'std')]
        print(f"  {metric}: {mean_val:.4f} +/- {std_val:.4f}")
    print("-" * 60)
