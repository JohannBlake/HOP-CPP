import pandas as pd
import numpy as np
from scipy import stats

# Define the file path
csv_file_path = 'ablation_studies_evaluation_dataset copy.csv'

# Read the CSV file
try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Filter by ablation_study_name and epoch
target_ablation = "With HAM, OPTV, visit frequency (Radiation)"
target_epoch = "epoch-100"
df_filtered = df[(df['ablation_study_name'] == target_ablation) & (df['epoch'] == target_epoch)]

if df_filtered.empty:
    print("No data found.")
    exit(0)

x = df_filtered['path_length_end']
y = df_filtered['steps_end']

# 1. Pearson Correlation
pearson_corr, _ = stats.pearsonr(x, y)

# 2. Linear Regression (y = mx + b)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value**2

print("-" * 30)
print(f"Correlation Analysis for '{target_ablation}' @ {target_epoch}")
print("-" * 30)
print(f"Pearson Correlation Coefficient: {pearson_corr:.6f}")
print(f"  (1.0 = perfect positive linear correlation, 0.0 = no linear correlation)")
print("-" * 30)
print(f"Linear Regression Fit (Steps = m * PathLength + b):")
print(f"  Slope (m): {slope:.6f}")
print(f"  Intercept (b): {intercept:.6f}")
print(f"  R-squared: {r_squared:.6f}")
print(f"  (R^2 represents the proportion of variance in 'steps' explained by 'path_length'.)")
print("-" * 30)

# 3. Check assumption of y = c * x (Intercept = 0)
# We can force intercept to 0 and calculate R^2 for that specific model
# Model: y_pred = c * x
# optimal c for minimizing MSE is sum(x*y) / sum(x*x)
c_optimal = np.sum(x * y) / np.sum(x * x)
y_pred_origin = c_optimal * x
ss_res_origin = np.sum((y - y_pred_origin)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared_origin = 1 - (ss_res_origin / ss_tot)

print(f"Strict Proportionality Fit (Steps = c * PathLength, intercept=0):")
print(f"  Optimal c: {c_optimal:.6f}")
print(f"  R-squared (forced origin): {r_squared_origin:.6f}")
