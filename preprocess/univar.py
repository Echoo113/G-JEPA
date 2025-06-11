import numpy as np
import os

# ==== 全局参数 ====
X_LEN = 30             # 输入 Patch 长度
Y_LEN = 10             # 输出 Patch 长度
STRIPE = 5             # 滑动步长
DATA_PATH = "data/MSL/MSL_test.npy"
LABEL_PATH = "data/MSL/MSL_test_label.npy"
VAR_INDICES = [0]      # 选用的变量维度列表
SAVE_DIR = "data/MSL/patches"
# ==== 全局变量（Patch数据） ====
X_patch = None         # shape: (N, T, F)
Y_patch = None         # shape: (N, T, F)
Y_label = None         # shape: (N,)

X_train = None
Y_train = None
Y_train_label = None

X_val = None
Y_val = None
Y_val_label = None

X_test = None
Y_test = None
Y_test_label = None


class AnomalyPatchExtractor:
    def __init__(self):
        print(f"[INIT] AnomalyPatchExtractor initialized.")
        print(f"  ➤ X_LEN={X_LEN}, Y_LEN={Y_LEN}, STRIPE={STRIPE}")
        print(f"  ➤ Data path: {DATA_PATH}, Label path: {LABEL_PATH}")
        print(f"  ➤ Variable indices: {VAR_INDICES}")

    def load_data(self):
        global data, label
        print("[LOAD] Loading data...")
        full_data = np.load(DATA_PATH)
        data = full_data[:, VAR_INDICES]  # Select multiple variables
        label = np.load(LABEL_PATH).astype(int)
        print(f"  ✔ Data shape: {data.shape}")
        print(f"  ✔ Label shape: {label.shape}")
        print(f"  ✔ True label ratio in test_labels: {np.mean(label):.4f}")
        print(f"  ✔ First true label index: {np.argmax(label)}")

    def extract_patches(self):
        global X_patch, Y_patch, Y_label
        print("[PATCH] Extracting sliding window patches...")
        total_len = len(data)

        Xs, Ys, Ls = [], [], []
        for i in range(0, total_len - X_LEN - Y_LEN + 1, STRIPE):
            x = data[i:i+X_LEN]
            y = data[i+X_LEN:i+X_LEN+Y_LEN]
            y_tag = label[i+X_LEN:i+X_LEN+Y_LEN]

            Xs.append(x)
            Ys.append(y)
            Ls.append(int(y_tag.any()))  # 只要 y_patch 中有任何异常点，则整个 patch 视为异常

        # 转换为 (N, T, F) 格式，确保只有一个特征维度
        X_patch = np.array(Xs).reshape(-1, X_LEN, 1)  # shape: (N, T, F)
        Y_patch = np.array(Ys).reshape(-1, Y_LEN, 1)  # shape: (N, T, F)
        Y_label = np.array(Ls)                        # shape: (N,)

        print(f"  ✔ Total patches extracted: {len(X_patch)}")
        print(f"  ✔ Positive (anomalous) patches: {np.sum(Y_label)}")
        print(f"  ✔ Patch shape: X: {X_patch.shape}, Y: {Y_patch.shape}, Label: {Y_label.shape}")

    def split_data(self, train_ratio=0.7, val_ratio=0.1):
        global X_train, Y_train, Y_train_label
        global X_val, Y_val, Y_val_label
        global X_test, Y_test, Y_test_label

        print("[SPLIT] Splitting dataset...")
        N = len(X_patch)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)

        X_train = X_patch[:n_train]
        Y_train = Y_patch[:n_train]
        Y_train_label = Y_label[:n_train]
        #print first few labels of Y_train_label
        print(f"  ✔ First few labels of Y_train_label: {Y_train_label[:5]}")

        X_val = X_patch[n_train:n_train+n_val]
        Y_val = Y_patch[n_train:n_train+n_val]
        Y_val_label = Y_label[n_train:n_train+n_val]

        X_test = X_patch[n_train+n_val:]
        Y_test = Y_patch[n_train+n_val:]
        Y_test_label = Y_label[n_train+n_val:]
       
        print(f"  ✔ X_train: {X_train.shape}, Y_train: {Y_train.shape}, Labels: {np.mean(Y_train_label):.4f}")
        print(f"  ✔ X_val:   {X_val.shape}, Y_val:   {Y_val.shape}, Labels: {np.mean(Y_val_label):.4f}")
        print(f"  ✔ X_test:  {X_test.shape}, Y_test:  {Y_test.shape}, Labels: {np.mean(Y_test_label):.4f}")

    def save_data(self):
        print("[SAVE] Saving data to NPZ files...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Save train data
        np.savez(
            os.path.join(SAVE_DIR, "train.npz"),
            x_patches=X_train,
            y_patches=Y_train,
            label=Y_train_label
        )
        
        # Save validation data
        np.savez(
            os.path.join(SAVE_DIR, "val.npz"),
            x_patches=X_val,
            y_patches=Y_val,
            label=Y_val_label
        )
        
        # Save test data
        np.savez(
            os.path.join(SAVE_DIR, "test.npz"),
            x_patches=X_test,
            y_patches=Y_test,
            label=Y_test_label
        )
        print(f"  ✔ Data saved to {SAVE_DIR}")


# ✅ 使用方式（建议放在主脚本中）：
if __name__ == "__main__":
    extractor = AnomalyPatchExtractor()
    extractor.load_data()
    extractor.extract_patches()
    extractor.split_data()
    extractor.save_data()
