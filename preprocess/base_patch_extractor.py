import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocess.datatool import DataTool

# ====== Global Constants ======
DEFAULT_FILENAME   = "data/SOLAR/solar_10_minutes_dataset.csv"
WINDOW_LEN         = 96    # 每个窗口的长度
WINDOW_STRIDE      = 96    # 窗口步长，设置为与窗口长度相同，确保不重叠
TRAIN_RATIO        = 0.8   # 训练集比例
VALID_RATIO        = 0.1   # 验证集比例

# Patch 设置
PATCH_LEN          = 16    # 每个 patch 的长度
PATCH_STRIDE       = 16    # patch 步长，设置为与 patch 长度相同，确保不重叠


class PatchExtractor:
    def __init__(
        self,
        filename: str             = DEFAULT_FILENAME,
        window_len: int           = WINDOW_LEN,
        window_stride: int        = WINDOW_STRIDE,
        train_ratio: float        = TRAIN_RATIO,
        valid_ratio: float        = VALID_RATIO,
        patch_len: int            = PATCH_LEN,
        patch_stride: int         = PATCH_STRIDE,
        debug: bool               = False
    ):
        self.data_tool    = DataTool(filename, debug=debug)
        self.window_len   = window_len
        self.window_stride= window_stride
        self.train_ratio  = train_ratio
        self.valid_ratio  = valid_ratio
        self.patch_len    = patch_len
        self.patch_stride = patch_stride
        self.debug        = debug

    def _extract_sliding_windows(self, data: np.ndarray) -> np.ndarray:
        """
        从标准化后的多变量时间序列 (T, C) 中提取不重叠的窗口。
        
        Returns:
            windows: np.ndarray, shape = (N, window_len, C)
        """
        T, C = data.shape
        max_start = T - self.window_len + 1
        windows = []

        for start in range(0, max_start, self.window_stride):
            window = data[start : start + self.window_len]  # (window_len, C)
            windows.append(window)

        return np.stack(windows, axis=0)  # (N, window_len, C)

    def _split_window_into_patches(self, window: np.ndarray, window_start_idx: int = 0) -> tuple:
        """
        将单个窗口分割成不重叠的 patches。
        
        Args:
            window: np.ndarray of shape (window_len, C)
            window_start_idx: 窗口在整个序列中的起始索引
            
        Returns:
            x_patches: np.ndarray of shape (num_patches-1, patch_len, C)
            y_patches: np.ndarray of shape (num_patches-1, patch_len, C)
        """
        window_len, C = window.shape
        num_patches = window_len // self.patch_len
        
        if self.debug:
            print(f"\n=== Window Patch Analysis ===")
            print(f"Window start index: {window_start_idx}")
            print(f"Window length: {window_len}")
            print(f"Number of patches per window: {num_patches}")
            print(f"Each patch length: {self.patch_len}")
        
        # 初始化 patches 数组
        patches = np.zeros((num_patches, self.patch_len, C), dtype=window.dtype)
        
        # 提取所有 patches
        for i in range(num_patches):
            start = i * self.patch_len
            end = start + self.patch_len
            patches[i] = window[start:end]
            if self.debug:
                print(f"Patch {i}: [{window_start_idx + start}:{window_start_idx + end}]")
        
        # 构造输入和输出
        x_patches = patches[:-1]  # 除了最后一个 patch
        y_patches = patches[1:]   # 除了第一个 patch
        
        if self.debug:
            print("\n=== Patch Pairs ===")
            for i in range(len(x_patches)):
                x_start = window_start_idx + i * self.patch_len
                x_end = x_start + self.patch_len
                y_start = window_start_idx + (i + 1) * self.patch_len
                y_end = y_start + self.patch_len
                print(f"Pair {i}:")
                print(f"  X: [{x_start}:{x_end}] → Y: [{y_start}:{y_end}]")
        
        return x_patches, y_patches

    def _split_indices(self, total_windows: int) -> tuple:
        """
        计算训练/验证/测试集的索引。
        """
        n_train = int(total_windows * self.train_ratio)
        n_valid = int(total_windows * self.valid_ratio)
        n_test  = total_windows - n_train - n_valid

        train_idx = (0, n_train)
        valid_idx = (n_train, n_train + n_valid)
        test_idx  = (n_train + n_valid, total_windows)
        return train_idx, valid_idx, test_idx

    def extract_and_store_all(self, save_dir: str = "data/SOLAR/patches"):
        """
        提取不重叠窗口的 patches，分割为训练/验证/测试集，并保存为 .npz 文件。
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) 加载并标准化整个数据集
        data = self.data_tool.get_data()  # shape = (T, C)
        
        if self.debug:
            print("\n=== Dataset Info ===")
            print(f"Total time steps: {data.shape[0]}")
            print(f"Number of features: {data.shape[1]}")

        # 2) 提取所有不重叠窗口
        windows = self._extract_sliding_windows(data)
        N = len(windows)
        
        if self.debug:
            print(f"\n=== Window Info ===")
            print(f"Total number of windows: {N}")
            print(f"Window length: {self.window_len}")
            print(f"Window stride: {self.window_stride}")

        # 3) 将每个窗口分割成 patches
        x_patches_list = []
        y_patches_list = []
        
        for i, window in enumerate(windows):
            window_start_idx = i * self.window_stride
            if self.debug and i < 2:  # 只打印前两个窗口的信息
                print(f"\nProcessing window {i} (start index: {window_start_idx}):")
            x_patches, y_patches = self._split_window_into_patches(window, window_start_idx)
            x_patches_list.append(x_patches)
            y_patches_list.append(y_patches)
        
        x_patches_all = np.stack(x_patches_list, axis=0)  # (N, num_patches-1, patch_len, C)
        y_patches_all = np.stack(y_patches_list, axis=0)  # (N, num_patches-1, patch_len, C)

        if self.debug:
            print("\n=== Final Patch Shapes ===")
            print(f"x_patches_all shape: {x_patches_all.shape}")
            print(f"y_patches_all shape: {y_patches_all.shape}")
            print(f"Number of patches per window: {x_patches_all.shape[1]}")
            print(f"Patch length: {x_patches_all.shape[2]}")
            print(f"Number of features: {x_patches_all.shape[3]}")

        # 4) 确定训练/验证/测试集的索引
        (t0, t1), (v0, v1), (s0, s1) = self._split_indices(N)

        # 5) 按索引分割 patches
        x_train_patches = x_patches_all[t0:t1]
        y_train_patches = y_patches_all[t0:t1]

        x_valid_patches = x_patches_all[v0:v1]
        y_valid_patches = y_patches_all[v0:v1]

        x_test_patches  = x_patches_all[s0:s1]
        y_test_patches  = y_patches_all[s0:s1]

        if self.debug:
            print("\n=== Split patch shapes ===")
            print(f"Train x patches: {x_train_patches.shape}, Train y patches: {y_train_patches.shape}")
            print(f"Valid x patches: {x_valid_patches.shape}, Valid y patches: {y_valid_patches.shape}")
            print(f"Test  x patches: {x_test_patches.shape},  Test y patches: {y_test_patches.shape}")

        # 6) 保存每个分割为 .npz 文件
        train_path = os.path.join(save_dir, "solar_train.npz")
        valid_path = os.path.join(save_dir, "solar_val.npz")
        test_path  = os.path.join(save_dir, "solar_test.npz")

        np.savez_compressed(train_path,
                            x_patches=x_train_patches,
                            y_patches=y_train_patches)
        np.savez_compressed(valid_path,
                            x_patches=x_valid_patches,
                            y_patches=y_valid_patches)
        np.savez_compressed(test_path,
                            x_patches=x_test_patches,
                            y_patches=y_test_patches)

        if self.debug:
            print(f"\nSaved train patches to: {train_path}")
            print(f"Saved valid patches to: {valid_path}")
            print(f"Saved test patches  to: {test_path}")

        return train_path, valid_path, test_path

    @staticmethod
    def load_patch_split(split_path: str) -> tuple:
        """
        Load pre-saved .npz split and return (x_patches, y_patches) as numpy arrays.
        Args:
            split_path: path to one of solar_train_patches.npz, etc.
        Returns:
            x_patches: np.ndarray, shape=(N_split, num_patches_in, patch_len, C)
            y_patches: np.ndarray, shape=(N_split, num_patches_out, patch_len, C)
        """
        data = np.load(split_path)
        return data["x_patches"], data["y_patches"]

    @staticmethod
    def verify_saved_patch_split(split_path: str):
        """
        Print shapes of x_patches and y_patches inside a saved .npz split.
        """
        data = np.load(split_path)
        print(f"*** Contents of {split_path} ***")
        for key in data:
            print(f"{key}: {data[key].shape}")


if __name__ == "__main__":
    # Create save directory path
    save_dir = "data/SOLAR/patches"
    
    # Create extractor and process data
    extractor = PatchExtractor(debug=True)
    train_path, val_path, test_path = extractor.extract_and_store_all(save_dir=save_dir)

    # Verify each patch split
    PatchExtractor.verify_saved_patch_split(train_path)
    PatchExtractor.verify_saved_patch_split(val_path)
    PatchExtractor.verify_saved_patch_split(test_path)
