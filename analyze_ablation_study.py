import pandas as pd
import os

# Define the file path
csv_file_path = 'ablation_studies_evaluation_dataset copy.csv'

# Check if file exists
if not os.path.exists(csv_file_path):
    print(f"Error: File '{csv_file_path}' not found.")
    exit(1)

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

# Group by 'map' and calculate mean and std for 'path_length_end' and 'steps_end'
# Using 'steps_end' as 'episode steps'
result = df_filtered.groupby('map')[['path_length_end', 'steps_end']].agg(['mean', 'std'])

print(f"Analysis for '{target_ablation}' at '{target_epoch}':")
print("-" * 30)
print(result)
print("-" * 30)

# Calculate overall average across all maps (if needed, or just display the per-map stats as requested)
print("\nOverall Average across filtered data:")
print(df_filtered[['path_length_end', 'steps_end']].agg(['mean', 'std']))
