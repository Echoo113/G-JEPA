import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocess.datatool import DataTool

# ====== Global Constants ======
DEFAULT_FILENAME   = "data/SOLAR/solar_10_minutes_dataset.csv"
INPUT_LEN          = 96    # Number of time steps in each input window
OUTPUT_LEN         = 96    # Number of time steps in each output window
WINDOW_STRIDE      = 48     # Sliding‐window stride for generating (input, output) pairs
TRAIN_RATIO        = 0.8   # Fraction of total windows used for training
VALID_RATIO        = 0.1   # Fraction of total windows used for validation (after training)

# Default patch settings within each 96‐step window
PATCH_LEN          = 16    # Length of each patch inside a 96‐step window
PATCH_STRIDE       = 8     # Stride for patch extraction within a 96‐step window


class PatchExtractor:
    def __init__(
        self,
        filename: str             = DEFAULT_FILENAME,
        input_len: int            = INPUT_LEN,
        output_len: int           = OUTPUT_LEN,
        window_stride: int        = WINDOW_STRIDE,
        train_ratio: float        = TRAIN_RATIO,
        valid_ratio: float        = VALID_RATIO,
        patch_len: int            = PATCH_LEN,
        patch_stride: int         = PATCH_STRIDE,
        debug: bool               = False
    ):
        self.data_tool    = DataTool(filename, debug=debug)
        self.input_len    = input_len
        self.output_len   = output_len
        self.window_stride= window_stride
        self.train_ratio  = train_ratio
        self.valid_ratio  = valid_ratio
        self.patch_len    = patch_len
        self.patch_stride = patch_stride
        self.debug        = debug

    def _extract_sliding_windows(self, data: np.ndarray) -> tuple:
        """
        From a standardized multivariate time series (T, C),
        extract all (input_window, output_window) pairs via sliding window.

        Returns:
            x_windows: np.ndarray, shape = (N, input_len, C)
            y_windows: np.ndarray, shape = (N, output_len, C)
        """
        T, C = data.shape
        window_size = self.input_len + self.output_len
        max_start = T - window_size + 1
        x_list, y_list = [], []

        for start in range(0, max_start, self.window_stride):
            x_win = data[start : start + self.input_len]                   # (input_len, C)
            y_win = data[start + self.input_len : start + window_size]     # (output_len, C)
            x_list.append(x_win)
            y_list.append(y_win)

        x_windows = np.stack(x_list, axis=0)  # (N, input_len, C)
        y_windows = np.stack(y_list, axis=0)  # (N, output_len, C)
        return x_windows, y_windows

    def _split_windows_into_patches(self, windows: np.ndarray) -> np.ndarray:
        """
        Given windows: np.ndarray of shape (N_windows, window_len, C),
        split each window into patches of length patch_len with stride patch_stride.

        Returns:
            patches: np.ndarray of shape (N_windows, num_patches, patch_len, C)
        """
        N_windows, window_len, C = windows.shape
        # Calculate number of patches per window
        num_patches = (window_len - self.patch_len) // self.patch_stride + 1

        # Initialize output array
        patches = np.zeros((N_windows, num_patches, self.patch_len, C), dtype=windows.dtype)

        for i in range(N_windows):
            for j in range(num_patches):
                start = j * self.patch_stride
                patches[i, j] = windows[i, start : start + self.patch_len]

        return patches  # shape: (N_windows, num_patches, patch_len, C)

    def _split_indices(self, total_windows: int) -> tuple:
        """
        Given total number of windows N, compute indices for train/valid/test splits.
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
        Extract sliding‐window patches (both input and output) from the entire dataset,
        split into train/valid/test, and save each split as .npz with patch arrays.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) Load & standardize whole dataset
        data = self.data_tool.get_data()  # shape = (T, C)

        # 2) Extract all sliding windows (x_windows, y_windows)
        x_all, y_all = self._extract_sliding_windows(data)
        N, _, C = x_all.shape

        # 3) Convert each window into patches
        #    x_all: (N, input_len, C) → x_patches_all: (N, num_patches_in, patch_len, C)
        #    y_all: (N, output_len, C) → y_patches_all: (N, num_patches_out, patch_len, C)
        x_patches_all = self._split_windows_into_patches(x_all)
        y_patches_all = self._split_windows_into_patches(y_all)

        if self.debug:
            print("=== After splitting into patches ===")
            print(f"x_patches_all shape: {x_patches_all.shape}")
            print(f"y_patches_all shape: {y_patches_all.shape}")

        # 4) Determine train/valid/test indices over windows
        (t0, t1), (v0, v1), (s0, s1) = self._split_indices(N)

        # 5) Slice patches by index
        x_train_patches = x_patches_all[t0:t1]  # (n_train, num_patches_in, patch_len, C)
        y_train_patches = y_patches_all[t0:t1]  # (n_train, num_patches_out, patch_len, C)

        x_valid_patches = x_patches_all[v0:v1]
        y_valid_patches = y_patches_all[v0:v1]

        x_test_patches  = x_patches_all[s0:s1]
        y_test_patches  = y_patches_all[s0:s1]

        if self.debug:
            print("\n=== Split patch shapes ===")
            print(f"Train x patches: {x_train_patches.shape}, Train y patches: {y_train_patches.shape}")
            print(f"Valid x patches: {x_valid_patches.shape}, Valid y patches: {y_valid_patches.shape}")
            print(f"Test  x patches: {x_test_patches.shape},  Test y patches: {y_test_patches.shape}")

        # 6) Save each split into .npz
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
