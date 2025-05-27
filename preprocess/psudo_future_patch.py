import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import re
import os
import pickle

# parameters
patch_size = 30
window_size = 5
short_quatile_start = 0.9   #short quantile start
short_patches = 2           # only take the first 2 patches
middle_quantile = 0.95     # middle quantile
middle_patches = 2        # middle take 3 patches 
long_patches = 2          # long take the last 3 patches
csv_path = "/Users/echohe/Desktop/Research/spark/data/GEM1h.csv"
output_dir = "data/pseudo_future_patches"
os.makedirs(output_dir, exist_ok=True)

# group columns by prefix
def group_columns_by_prefix(csv_path):
    df = pd.read_csv(csv_path)
    if 'Time' in df.columns:
        df = df.drop(columns=["Time"])
    prefix_dict = {}
    for col in df.columns:
        m = re.match(r"u(\d)\d+", col)
        if m:
            prefix = f"u{m.group(1)}"
            if prefix not in prefix_dict:
                prefix_dict[prefix] = []
            prefix_dict[prefix].append(col)
    for prefix in prefix_dict:
        prefix_dict[prefix] = df[prefix_dict[prefix]]
    return prefix_dict

# extract pseudo-future patches
def extract_pseudo_future_patches(group_df):
    scaler = StandardScaler()
    patches = {
        "short": {"patches": [], "sensors": [], "indices": [], "count": 0},
        "middle": {"patches": [], "sensors": [], "indices": [], "count": 0},
        "long": {"patches": [], "sensors": [], "indices": [], "count": 0}
    }
    
    seq_length = len(group_df.iloc[:, 0])
    
    # calculate short patch start
    short_start = int(seq_length * short_quatile_start)
    # only take the first 2 patches
    short_indices = [short_start + i * patch_size for i in range(short_patches)]

    # calculate middle and long start
    middle_center = int(seq_length * middle_quantile)
    middle_start = max(0, middle_center - patch_size)
    middle_end = min(seq_length - (patch_size + window_size), middle_center + patch_size)
    long_start = seq_length - (long_patches * patch_size + window_size)

    print(f"Short patch indices: {short_indices}")
    print(f"Middle patch indices: {middle_start} to {middle_end}")
    print(f"Long patch indices: {long_start} to {seq_length}")

    print(f"\nDebug info for {group_df.columns[0]}")
    print(f"Sequence length: {seq_length}")
    print(f"Middle center (95%): {middle_center}")
    print(f"Middle range: {middle_start} to {middle_end}")

    for sensor_col in group_df.columns:
        signal = group_df[sensor_col].astype(str).str.replace(" µA", "").astype(float).values
        signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        
        # short patches (only take the first 2 patches)
        for start_idx in short_indices:
            end_idx = start_idx + patch_size + window_size
            if end_idx <= len(signal_scaled):
                full_patch = signal_scaled[start_idx:end_idx]
                patch_segment = full_patch[window_size:]
                x = torch.tensor(patch_segment, dtype=torch.float32)
                patches["short"]["patches"].append(x)
                patches["short"]["sensors"].append(sensor_col)
                patches["short"]["indices"].append(start_idx)
                patches["short"]["count"] += 1
        
        # middle patches (95% quantile)
        middle_indices = np.linspace(middle_start, middle_end, middle_patches, dtype=int)
        for start_idx in middle_indices:
            end_idx = start_idx + patch_size + window_size
            if end_idx <= len(signal_scaled):
                full_patch = signal_scaled[start_idx:end_idx]
                patch_segment = full_patch[window_size:]
                x = torch.tensor(patch_segment, dtype=torch.float32)
                patches["middle"]["patches"].append(x)
                patches["middle"]["sensors"].append(sensor_col)
                patches["middle"]["indices"].append(start_idx)
                patches["middle"]["count"] += 1
        
        # long patches (take the last 2 patches)
        for i in range(long_patches):
            start_idx = long_start + i * patch_size
            end_idx = start_idx + patch_size + window_size
            if end_idx <= len(signal_scaled):
                full_patch = signal_scaled[start_idx:end_idx]
                patch_segment = full_patch[window_size:]
                x = torch.tensor(patch_segment, dtype=torch.float32)
                patches["long"]["patches"].append(x)
                patches["long"]["sensors"].append(sensor_col)
                patches["long"]["indices"].append(start_idx)
                patches["long"]["count"] += 1

    # convert to tensor
    for window_type in patches:
        if patches[window_type]["patches"]:
            patches[window_type]["patches"] = torch.stack(patches[window_type]["patches"])
        else:
            patches[window_type]["patches"] = torch.empty((0, patch_size))

    return patches

# main logic
if __name__ == "__main__":
    grouped = group_columns_by_prefix(csv_path)
    for prefix, group_df in grouped.items():
        result = extract_pseudo_future_patches(group_df)
        with open(os.path.join(output_dir, f"{prefix}_pseudo_future.pkl"), 'wb') as f:
            pickle.dump(result, f)
        print(f"\nGroup {prefix}:")
        for window_type in ["short", "middle", "long"]:
            print(f"  {window_type.capitalize()}: {result[window_type]['count']} patches")
