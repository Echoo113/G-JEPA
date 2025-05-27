import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import re
import os
import pickle

# Global parameters for patch extraction
patch_size = 30
window_size = 5
stride = 5
threshold_ratio = 0.015  # for old label (if used later)
suppression_gap = 5
end_ratio = 1  

# Global parameters for window generation
short_range = (0.75, 1)
middle_step = 0.09
long_step = 0.16
holdout_ratio = 0.05

def generate_quantile_windows(length):
    """
    Generate patch window start indices for three types: short-term, middle-term, and long-term,
    avoiding the last holdout_ratio portion.
    Uses global variables as parameters.
    
    Args:
        length: sequence length

    Returns:
        dict containing "short", "middle", "long" window start indices
    """
    usable_len = int(length * (1 - holdout_ratio)) - (patch_size + window_size)
    short_start = int(length * short_range[0])
    short_end = int(length * short_range[1]) - (patch_size + window_size)
    
    # Short-term windows: dense extraction to capture burst changes
    short_indices = list(range(short_start, short_end, patch_size))

    # Middle-term windows: sample at regular intervals
    middle_indices = []
    step_middle = int(length * middle_step)
    for start in range(0, usable_len, step_middle):
        if start + patch_size + window_size < length * (1 - holdout_ratio):
            middle_indices.append(start)

    # Long-term windows: sparse global trends
    long_indices = []
    step_long = int(length * long_step)
    for start in range(0, usable_len, step_long):
        if start + patch_size + window_size < length * (1 - holdout_ratio):
            long_indices.append(start)

    return {
        "short": short_indices,
        "middle": middle_indices,
        "long": long_indices
    }

def group_columns_by_prefix(csv_path):
    """
    Read CSV file and group columns by u1, u2, etc. prefixes.
    Returns a dictionary where key is prefix and value is the group DataFrame.
    Example: u101, u102, u103 will be grouped under u1
    """
    df = pd.read_csv(csv_path)

    if 'Time' in df.columns:
        df = df.drop(columns=["Time"])
    
    prefix_dict = {}
    for col in df.columns:
        # Modified regex to match only the first digit after 'u'
        m = re.match(r"u(\d)\d+", col)
        if m:
            prefix = f"u{m.group(1)}"  # take only the first digit
            if prefix not in prefix_dict:
                prefix_dict[prefix] = []
            prefix_dict[prefix].append(col)
    
    print("\nFound prefixes:", sorted(list(prefix_dict.keys())))
    
    # Convert to DataFrame
    for prefix in prefix_dict:
        prefix_dict[prefix] = df[prefix_dict[prefix]]
        print(f"\nColumns in {prefix}:", prefix_dict[prefix].columns.tolist())
    
    return prefix_dict

def extract_patches_from_df(df):
    """
    Extract patches from a DataFrame (multiple columns) and return a list of patch tensors.
    """
    scaler = StandardScaler()
    X_all = []
    for sensor_col in df.columns:
        signal = df[sensor_col].astype(str).str.replace(" µA", "").astype(float).values
        signal_reshaped = signal.reshape(-1, 1)
        signal_scaled = scaler.fit_transform(signal_reshaped).flatten()
        num_patches = (len(signal_scaled) - patch_size - window_size) // patch_size
        for i in range(num_patches):
            start = i * patch_size
            end = start + patch_size + window_size
            full_patch = signal_scaled[start:end]
            patch_segment = full_patch[window_size:]  # skip context part
            x = torch.tensor(patch_segment, dtype=torch.float32)
            X_all.append(x)
    if X_all:
        X_all_tensor = torch.stack(X_all)
    else:
        X_all_tensor = torch.empty((0, patch_size))
    return X_all_tensor

def store_patches_by_group(group_df, windows, group_name):
    """
    Store patches for a group, categorized by short-term, middle-term, and long-term.
    
    Args:
        group_df: grouped DataFrame
        windows: window indices dictionary containing "short", "middle", "long" windows
        group_name: group name (e.g., "u1", "u2")
    
    Returns:
        dict containing patches and metadata for each time scale
    """
    scaler = StandardScaler()
    patches = {
        "short": {
            "patches": [],
            "sensors": [],
            "indices": [],
            "count": 0
        },
        "middle": {
            "patches": [],
            "sensors": [],
            "indices": [],
            "count": 0
        },
        "long": {
            "patches": [],
            "sensors": [],
            "indices": [],
            "count": 0
        }
    }
    
    # For each sensor column
    for sensor_col in group_df.columns:
        # Standardize data
        signal = group_df[sensor_col].astype(str).str.replace(" µA", "").astype(float).values
        signal_reshaped = signal.reshape(-1, 1)
        signal_scaled = scaler.fit_transform(signal_reshaped).flatten()
        
        # For each time scale
        for window_type in ["short", "middle", "long"]:
            for start_idx in windows[window_type]:
                end_idx = start_idx + patch_size + window_size
                if end_idx <= len(signal_scaled):
                    full_patch = signal_scaled[start_idx:end_idx]
                    patch_segment = full_patch[window_size:]  # skip context part
                    x = torch.tensor(patch_segment, dtype=torch.float32)
                    
                    # Store in corresponding time scale list
                    patches[window_type]["patches"].append(x)
                    patches[window_type]["sensors"].append(sensor_col)
                    patches[window_type]["indices"].append(start_idx)
                    patches[window_type]["count"] += 1
    
    # Convert to tensor
    for window_type in patches:
        if patches[window_type]["patches"]:
            patches[window_type]["patches"] = torch.stack(patches[window_type]["patches"])
        else:
            patches[window_type]["patches"] = torch.empty((0, patch_size))
    
    return patches

if __name__ == "__main__":
    print("Starting patch extraction by group...")
    groups = group_columns_by_prefix("/Users/echohe/Desktop/Research/spark/data/GEM1h.csv")
    print("\nAnalyzing windows for each group...")
    
    # Create storage directory
    os.makedirs("data/downstream_patches", exist_ok=True)
    
    for prefix, group_df in groups.items():
        # Get sequence length
        seq_length = len(group_df.iloc[:, 0])
        
        # Generate windows
        windows = generate_quantile_windows(length=seq_length)
        
        # Store patches
        patches = store_patches_by_group(group_df, windows, prefix)
        
        # Print statistics
        print(f"\nGroup {prefix}:")
        print(f"  Sequence length: {seq_length}")
        for window_type in ["short", "middle", "long"]:
            print(f"  {window_type.capitalize()}: {patches[window_type]['count']} patches")
        
        # Save patches
        save_path = f"data/downstream_patches/{prefix}_downstream_patches.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(patches, f)
        print(f"  Saved to: {save_path}")
