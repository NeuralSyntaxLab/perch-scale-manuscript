import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from pipeline import read_timeseries

base_path = Path(__file__).parent
print(base_path)

control_bpath = base_path / "control_exp" 
# current_fname = '40'
res = {}
control_fnames = ['5', '15', '17', '26', '30', '40']
for fname in control_fnames:
    # Read to csv
    current_fname = fname
    print(f"Processing file: {current_fname}")
    current_path = control_bpath / f"control_{current_fname}_weight_report.csv"
    # Read to csv file
    df = pd.read_csv(current_path, parse_dates=['Time'])
    if 'weight' not in df.columns:
        # assign column #1 to 'weight' if it exists
        if len(df.columns) > 1:
            df.rename(columns={df.columns[1]: 'weight'}, inplace=True)
        else:
            raise ValueError("DataFrame does not contain 'weight' column and has no other columns to rename.")
    # Filter out rows with 'weight' less than 2 or greater than 50
    # max_threshold = (int(current_fname) + 10) if current_fname.isdigit() else 25
    df = df[(df['weight'] >= 1) & (df['weight'] <= 50)]

    print(df.head())    
    print(len(df))
    # Calculate mode value of 'weight' column
    mode_weight = df['weight'].mode().iloc[0] if not df['weight'].empty else None
    print(f"Mode weight: {mode_weight}")

    # calculate median value of 'weight' column
    median_weight = df['weight'].median() if not df['weight'].empty else None
    print(f"Median weight: {median_weight}")
    # caculate standard deviation of 'weight' column
    std_weight = df['weight'].std() if not df['weight'].empty else None
    print(f"Standard deviation of weight: {std_weight}")

    # Store results in dictionary
    res[current_fname] = {
        'mode_weight': mode_weight,
        'median_weight': median_weight,
        'std_weight': std_weight,
        'num_records': len(df)
    }

    # generate histogram of 'weight' column
    plt.figure(figsize=(10, 6))
    plt.hist(df['weight'].dropna(), bins=40, edgecolor='black', alpha=0.7)
    plt.title('Weight Distribution')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    if mode_weight is not None:
        plt.axvline(mode_weight, color='red', linestyle='dashed', linewidth=1.5, label=f'Mode: {mode_weight}')
    if median_weight is not None:
        plt.axvline(median_weight, color='blue', linestyle='dashed', linewidth=1.5, label=f'Median: {median_weight}')
    plt.legend()
    plt.show()

actual_weight_mapping = {}
# Define actual weights for each control file
actual_weights = [5, 15.75, 17.3, 26.8, 30, 40.6]
# Map control_fnames to actual weights
for fname, weight in zip(control_fnames, actual_weights):
    actual_weight_mapping[fname] = weight

# Ensure all control_fnames are in the actual_weight_mapping
for fname in control_fnames:
    if fname not in actual_weight_mapping:
        print(f"Warning: {fname} not found in actual weight mapping. Defaulting to None.")
        actual_weight_mapping[fname] = None

# Add actual weight to results
for fname, actual_weight in actual_weight_mapping.items():
    if fname in res:
        res[fname]['actual_weight'] = actual_weight
    else:
        print(f"Warning: {fname} not found in results.")

# Print results
for fname, metrics in res.items():
    print(f"File: {fname}")
    print(f"  Mode Weight: {metrics['mode_weight']}")
    print(f"  Median Weight: {metrics['median_weight']}")
    print(f"  Standard Deviation: {metrics['std_weight']}")
    print(f"  Number of Records: {metrics['num_records']}")
    print(f"  Actual Weight: {metrics.get('actual_weight', 'N/A')}")
    print()  # New line for better readability

# # save results to JSON file
# output_path = control_bpath / "control_weight_analysis_results.json"
# with open(output_path, 'w') as f:
#     json.dump(res, f, indent=4)
# print(f"Results saved to {output_path}")