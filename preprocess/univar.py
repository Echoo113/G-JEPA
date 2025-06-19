import numpy as np
import os

# ==== 推荐参数设置 ====
HISTORY_WINDOW_LEN = 80    # 历史窗口的总长度 (L)
FUTURE_WINDOW_LEN = 16     # 预测窗口的总长度 (T)
PATCH_LEN = 16             # 每个Patch的长度

# 检查参数是否可以整除
assert HISTORY_WINDOW_LEN % PATCH_LEN == 0, "历史窗口长度必须能被Patch长度整除"
assert FUTURE_WINDOW_LEN % PATCH_LEN == 0, "预测窗口长度必须能被Patch长度整除"

# 根据参数自动计算Patch数量
NUM_HISTORY_PATCHES = HISTORY_WINDOW_LEN // PATCH_LEN
NUM_FUTURE_PATCHES = FUTURE_WINDOW_LEN // PATCH_LEN

DATA_PATH = "data/MSL/MSL_test.npy"
LABEL_PATH = "data/MSL/MSL_test_label.npy"
VAR_INDICES = [0]
SAVE_DIR = f"data/MSL/patches"
class AnomalyWindowPatcher:
    def __init__(self, history_len, future_len, patch_len, var_indices):
        self.history_len = history_len
        self.future_len = future_len
        self.patch_len = patch_len
        self.var_indices = var_indices

        if self.history_len % self.patch_len != 0 or self.future_len % self.patch_len != 0:
            raise ValueError("窗口长度必须能被Patch长度整除！")
            
        self.num_history_patches = self.history_len // self.patch_len
        self.num_future_patches = self.future_len // self.patch_len

        print(f"[INIT] AnomalyWindowPatcher initialized.")
        print(f"  ➤ History Window (X): {self.history_len} steps -> {self.num_history_patches} patches of length {self.patch_len}")
        print(f"  ➤ Future Window  (Y): {self.future_len} steps -> {self.num_future_patches} patches of length {self.patch_len}")

    def load_data(self, data_path, label_path):
        print("[LOAD] Loading data...")
        full_data = np.load(data_path)
        self.data = full_data[:, self.var_indices]
        self.label = np.load(label_path).astype(int)
        print(f"  ✔ Raw Data shape: {self.data.shape}")

    def _extract_and_patch_windows(self, data, label):
        """
        内部辅助函数：提取非重叠窗口，切分Patch，并为X和Y分别生成窗口级标签。
        """
        total_len = len(data)
        stride = self.history_len
        num_features = data.shape[1]

        # <--- 变化点: 初始化四个列表，分别存储X, X的标签, Y, Y的标签
        Xs, X_labels, Ys, Y_labels = [], [], [], []

        for i in range(0, total_len - self.history_len - self.future_len + 1, stride):
            # 1. 提取数据窗口
            history_window = data[i : i + self.history_len]
            future_window = data[i + self.history_len : i + self.history_len + self.future_len]
            
            # <--- 变化点: 同时提取X和Y对应的标签段
            history_labels_window = label[i : i + self.history_len]
            future_labels_window = label[i + self.history_len : i + self.history_len + self.future_len]

            # 2. 将数据窗口 reshape 成 patches
            history_patches = history_window.reshape(self.num_history_patches, self.patch_len, num_features)
            future_patches = future_window.reshape(self.num_future_patches, self.patch_len, num_features)

            # 3. <--- 变化点: 分别计算X和Y的窗口级标签
            x_label = int(history_labels_window.any())
            y_label = int(future_labels_window.any())

            Xs.append(history_patches)
            X_labels.append(x_label)
            Ys.append(future_patches)
            Y_labels.append(y_label)

        if not Xs:
            return None, None, None, None

        # <--- 变化点: 返回四个数组
        return np.array(Xs), np.array(X_labels), np.array(Ys), np.array(Y_labels)

    def process_and_split_data(self, train_ratio=0.7, val_ratio=0.1):
        """
        核心方法：先分割原始时间序列，再在每个部分中提取并切分窗口
        """
        print("\n[PROCESS & SPLIT] Splitting raw data first, then extracting and patching windows...")
        
        n_total = len(self.data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_data, train_label = self.data[:n_train], self.label[:n_train]
        val_data, val_label = self.data[n_train:n_train+n_val], self.label[n_train:n_train+n_val]
        test_data, test_label = self.data[n_train+n_val:], self.label[n_train+n_val:]

        print(f"  Split info: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # <--- 变化点: 解包四个返回值，并使用更清晰的变量名
        self.X_train, self.X_train_label, self.Y_train, self.Y_train_label = self._extract_and_patch_windows(train_data, train_label)
        self.X_val, self.X_val_label, self.Y_val, self.Y_val_label = self._extract_and_patch_windows(val_data, val_label)
        self.X_test, self.X_test_label, self.Y_test, self.Y_test_label = self._extract_and_patch_windows(test_data, test_label)

        print("\n--- Extracted Data Statistics ---")
        # <--- 变化点: 更新打印信息，分别显示X和Y的标签比例
        if self.X_train is not None:
            print(f"  ✔ Train: X shape={self.X_train.shape}, Y shape={self.Y_train.shape}")
            print(f"           X Label ratio: {np.mean(self.X_train_label):.4f}, Y Label ratio: {np.mean(self.Y_train_label):.4f}")
        if self.X_val is not None:
            print(f"  ✔ Val:   X shape={self.X_val.shape}, Y shape={self.Y_val.shape}")
            print(f"           X Label ratio: {np.mean(self.X_val_label):.4f}, Y Label ratio: {np.mean(self.Y_val_label):.4f}")
        if self.X_test is not None:
            print(f"  ✔ Test:  X shape={self.X_test.shape}, Y shape={self.Y_test.shape}")
            print(f"           X Label ratio: {np.mean(self.X_test_label):.4f}, Y Label ratio: {np.mean(self.Y_test_label):.4f}")

    def save_data(self, save_dir):
        print(f"\n[SAVE] Saving data to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        
        # <--- 变化点: 保存所有四个关键数组，并使用明确的键名
        if self.X_train is not None:
            np.savez(os.path.join(save_dir, "train.npz"), 
                     x_patches=self.X_train, x_label=self.X_train_label, 
                     y_patches=self.Y_train, y_label=self.Y_train_label)
        if self.X_val is not None:
            np.savez(os.path.join(save_dir, "val.npz"), 
                     x_patches=self.X_val, x_label=self.X_val_label, 
                     y_patches=self.Y_val, y_label=self.Y_val_label)
        if self.X_test is not None:
            np.savez(os.path.join(save_dir, "test.npz"), 
                     x_patches=self.X_test, x_label=self.X_test_label, 
                     y_patches=self.Y_test, y_label=self.Y_test_label)

        print(f"  ✔ Data saved successfully.")

if __name__ == "__main__":
    patcher = AnomalyWindowPatcher(
        history_len=HISTORY_WINDOW_LEN, 
        future_len=FUTURE_WINDOW_LEN, 
        patch_len=PATCH_LEN, 
        var_indices=VAR_INDICES
    )
    patcher.load_data(data_path=DATA_PATH, label_path=LABEL_PATH)
    patcher.process_and_split_data(train_ratio=0.65, val_ratio=0.1)
    patcher.save_data(save_dir=SAVE_DIR)