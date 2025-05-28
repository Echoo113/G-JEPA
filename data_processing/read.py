import numpy as np
import pandas as pd
from collections import Counter

# read the npy file
data = np.load('/Users/echohe/Desktop/Research/spark/data/SWAT_train.npy')

# Basic information
print("=== Basic Information ===")
print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Number of features: {data.shape[1] if len(data.shape) > 1 else 1}")

# Check for missing values
print("\n=== Missing Values Analysis ===")
missing_values = np.isnan(data)
total_missing = np.sum(missing_values)
print(f"Total missing values: {total_missing}")
print(f"Percentage of missing values: {(total_missing / data.size) * 100:.2f}%")

# Basic statistics
print("\n=== Basic Statistics ===")
print("Mean:", np.mean(data, axis=0))
print("Standard deviation:", np.std(data, axis=0))
print("Minimum:", np.min(data, axis=0))
print("Maximum:", np.max(data, axis=0))

# Check for data balance (if the data has labels)
if len(data.shape) > 1 and data.shape[1] > 1:
    # Assuming the last column is the label
    labels = data[:, -1]
    label_counts = Counter(labels)
    print("\n=== Data Balance Analysis ===")
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples ({count/len(labels)*100:.2f}%)")

# Check for outliers using IQR method
print("\n=== Outlier Analysis ===")
Q1 = np.percentile(data, 25, axis=0)
Q3 = np.percentile(data, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = np.sum((data < lower_bound) | (data > upper_bound), axis=0)
print(f"Number of outliers per feature: {outliers}")

# Display first few samples
print("\n=== First 5 samples ===")
print(data[:5])