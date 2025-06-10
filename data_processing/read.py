import numpy as np

# 文件路径（请根据你实际路径修改）
train_path = 'data/MSL/MSL_train.npy'
test_path = 'data/MSL/MSL_test.npy'
label_path = 'data/MSL/MSL_test_label.npy'

# 加载数据
train_data = np.load(train_path)
test_data = np.load(test_path)
test_labels = np.load(label_path)

# 打印信息
print("=== MSL_train.npy ===")
print(f"Shape: {train_data.shape}")
print(f"Dtype: {train_data.dtype}")
print(f"Min: {np.min(train_data):.4f}, Max: {np.max(train_data):.4f}")
print(f"Example rows:\n{train_data[:3]}")

print("\n=== MSL_test.npy ===")
print(f"Shape: {test_data.shape}")
print(f"Dtype: {test_data.dtype}")
print(f"Min: {np.min(test_data):.4f}, Max: {np.max(test_data):.4f}")
print(f"Example rows:\n{test_data[:3]}")

print("\n=== MSL_test_label.npy ===")
#print count of 0 and 1 in test_labels
print(f"Count of 0 in test_labels: {np.sum(test_labels==0)}")
print(f"Count of 1 in test_labels: {np.sum(test_labels==1)}")
#print 5 rows of test_labels

print(f"Shape: {test_labels.shape}")
print(f"Dtype: {test_labels.dtype}")
print(f"Unique values: {np.unique(test_labels)}")
print(f"Example: {test_labels[:20]}")
#print the anomaly ratio in test_labels
print(f"Anomaly ratio in test_labels: {np.sum(test_labels)/len(test_labels)}")

#print the true label ratio in test_labels
print(f"True label ratio in test_labels: {np.sum(test_labels)/len(test_labels)}")
#print first true label index in test_labels
print(f"First true label index in test_labels: {np.where(test_labels == 1)[0][0]}")