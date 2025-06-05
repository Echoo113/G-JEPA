import os
import numpy as np
import pandas as pd
from preprocess.datatool import DataTool

# ====== Global Constants ======
DEFAULT_FILENAME   = "data/MSL/MSL_train.npy"
DEFAULT_TEST_FILE  = "data/MSL/MSL_test.npy"  # 新增测试集文件路径
INPUT_LEN          = 100   # Number of time steps in each input window
OUTPUT_LEN         = 100   # Number of time steps in each output window
WINDOW_STRIDE      = 20    # Sliding‐window stride for generating (input, output) pairs
TRAIN_RATIO        = 0.9   # Fraction of total windows used for training
VALID_RATIO        = 0.1   # Fraction of total windows used for validation (after training)

# Default patch settings within each 100‐step window
PATCH_LEN          = 20    # Length of each patch inside a 100‐step window
PATCH_STRIDE       = 10    # Stride for patch extraction within a 100‐step window

# 测试集划分比例
FINAL_TEST_RATIO   = 2/10  # 最终测试集占比 (2/10)
TUNE_VAL_RATIO     = 0.1   # 微调验证集占比 (10%)


class AnomalyPatchExtractor:
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
        From a multivariate time series (T, C),
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
        Given total number of windows N, compute indices for train/valid splits.
        Note: For MSL, we don't need test split as it's provided separately.
        """
        n_train = int(total_windows * self.train_ratio)
        n_valid = total_windows - n_train

        train_idx = (0, n_train)
        valid_idx = (n_train, total_windows)
        return train_idx, valid_idx

    def extract_and_store_all(self, save_dir: str = "data/MSL/patches"):
        """
        Extract sliding‐window patches (both input and output) from the entire dataset,
        split into train/valid, and save each split as .npz with patch arrays.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) Load raw dataset
        data = self.data_tool.load()  # shape = (T, C)
        
        if self.debug:
            print("\n=== Raw Data Sample (First 5 timesteps, first 5 features) ===")
            print(data[:5, :5])
            print("\n=== Data Statistics ===")
            print(f"Mean: {np.mean(data):.4f}")
            print(f"Std: {np.std(data):.4f}")
            print(f"Min: {np.min(data):.4f}")
            print(f"Max: {np.max(data):.4f}")

        # 2) Extract all sliding windows (x_windows, y_windows)
        x_all, y_all = self._extract_sliding_windows(data)
        N, _, C = x_all.shape

        if self.debug:
            print("\n=== First Window Sample (First patch, first 5 timesteps, first 5 features) ===")
            print("Input window:")
            print(x_all[0, :5, :5])
            print("\nOutput window:")
            print(y_all[0, :5, :5])

        # 3) Convert each window into patches
        x_patches_all = self._split_windows_into_patches(x_all)
        y_patches_all = self._split_windows_into_patches(y_all)

        if self.debug:
            print("\n=== First Patch Sample (First 5 timesteps, first 5 features) ===")
            print("Input patch:")
            print(x_patches_all[0, 0, :5, :5])
            print("\nOutput patch:")
            print(y_patches_all[0, 0, :5, :5])

        if self.debug:
            print("\n=== After splitting into patches ===")
            print(f"x_patches_all shape: {x_patches_all.shape}")
            print(f"y_patches_all shape: {y_patches_all.shape}")

        # 4) Determine train/valid indices over windows
        (t0, t1), (v0, v1) = self._split_indices(N)

        # 5) Slice patches by index
        x_train_patches = x_patches_all[t0:t1]
        y_train_patches = y_patches_all[t0:t1]

        x_valid_patches = x_patches_all[v0:v1]
        y_valid_patches = y_patches_all[v0:v1]

        if self.debug:
            print("\n=== Split patch shapes ===")
            print(f"Train x patches: {x_train_patches.shape}, Train y patches: {y_train_patches.shape}")
            print(f"Valid x patches: {x_valid_patches.shape}, Valid y patches: {y_valid_patches.shape}")
            print("\n=== Train/Valid Data Statistics ===")
            print("Train data:")
            print(f"Mean: {np.mean(x_train_patches):.4f}")
            print(f"Std: {np.std(x_train_patches):.4f}")
            print(f"Min: {np.min(x_train_patches):.4f}")
            print(f"Max: {np.max(x_train_patches):.4f}")
            print("\nValid data:")
            print(f"Mean: {np.mean(x_valid_patches):.4f}")
            print(f"Std: {np.std(x_valid_patches):.4f}")
            print(f"Min: {np.min(x_valid_patches):.4f}")
            print(f"Max: {np.max(x_valid_patches):.4f}")

        # 6) Save each split into .npz
        train_path = os.path.join(save_dir, "msl_train.npz")
        valid_path = os.path.join(save_dir, "msl_val.npz")

        np.savez_compressed(train_path,
                            x_patches=x_train_patches,
                            y_patches=y_train_patches)
        np.savez_compressed(valid_path,
                            x_patches=x_valid_patches,
                            y_patches=y_valid_patches)

        if self.debug:
            print(f"\nSaved train patches to: {train_path}")
            print(f"Saved valid patches to: {valid_path}")

        return train_path, valid_path

    def extract_test_set(self, test_filename: str = DEFAULT_TEST_FILE, save_dir: str = "data/MSL/patches"):
        """
        Extract (input, output) patches from MSL test set
        
        Args:
            test_filename: Path to MSL test set file (default: data/MSL/MSL_test.npy)
            save_dir: Directory to save the test patches (default: data/MSL/patches)
            
        Returns:
            tuple: (x_test_patches, y_test_patches) - The extracted test patches
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) Load test dataset
        test_data_tool = DataTool(test_filename, debug=self.debug)
        test_data = test_data_tool.load()  # shape = (T, C)

        if self.debug:
            print("\n=== Test Data Sample (First 5 timesteps, first 5 features) ===")
            print(test_data[:5, :5])
            print("\n=== Test Data Statistics ===")
            print(f"Mean: {np.mean(test_data):.4f}")
            print(f"Std: {np.std(test_data):.4f}")
            print(f"Min: {np.min(test_data):.4f}")
            print(f"Max: {np.max(test_data):.4f}")

        # 2) Extract sliding windows
        x_test, y_test = self._extract_sliding_windows(test_data)

        if self.debug:
            print("\n=== First Test Window Sample (First patch, first 5 timesteps, first 5 features) ===")
            print("Input window:")
            print(x_test[0, :5, :5])
            print("\nOutput window:")
            print(y_test[0, :5, :5])

        # 3) Convert windows into patches
        x_test_patches = self._split_windows_into_patches(x_test)
        y_test_patches = self._split_windows_into_patches(y_test)

        if self.debug:
            print("\n=== First Test Patch Sample (First 5 timesteps, first 5 features) ===")
            print("Input patch:")
            print(x_test_patches[0, 0, :5, :5])
            print("\nOutput patch:")
            print(y_test_patches[0, 0, :5, :5])
            print("\n=== Test patch shapes ===")
            print(f"Test x patches: {x_test_patches.shape}, Test y patches: {y_test_patches.shape}")
            print("\n=== Test Patches Statistics ===")
            print(f"Mean: {np.mean(x_test_patches):.4f}")
            print(f"Std: {np.std(x_test_patches):.4f}")
            print(f"Min: {np.min(x_test_patches):.4f}")
            print(f"Max: {np.max(x_test_patches):.4f}")

        return x_test_patches, y_test_patches

    def split_test_into_tune_and_final(self, x_test_patches: np.ndarray, y_test_patches: np.ndarray, save_dir: str = "data/MSL/patches"):
        """
        将测试集划分为微调集和最终测试集
        
        Args:
            x_test_patches: Test input patches array
            y_test_patches: Test output patches array
            save_dir: Directory to save the split results
            
        Returns:
            tuple: (tune_train_path, tune_val_path, final_test_path)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1) Calculate split sizes
        final_test_size = int(len(x_test_patches) * FINAL_TEST_RATIO)  # 20% for final test
        tune_size = len(x_test_patches) - final_test_size
        
        # 2) Split into final test and tuning sets
        x_final_test = x_test_patches[-final_test_size:]
        y_final_test = y_test_patches[-final_test_size:]
        
        x_tune_all = x_test_patches[:-final_test_size]
        y_tune_all = y_test_patches[:-final_test_size]
        
        # 3) Further split tuning set into train and validation
        val_size = int(tune_size * TUNE_VAL_RATIO)
        train_size = tune_size - val_size
        
        x_tune_train = x_tune_all[:train_size]
        y_tune_train = y_tune_all[:train_size]
        
        x_tune_val = x_tune_all[train_size:]
        y_tune_val = y_tune_all[train_size:]
        
        # 4) Save all splits
        tune_train_path = os.path.join(save_dir, "msl_tune_train.npz")
        tune_val_path = os.path.join(save_dir, "msl_tune_val.npz")
        final_test_path = os.path.join(save_dir, "msl_final_test.npz")
        
        np.savez_compressed(tune_train_path,
                          x_patches=x_tune_train,
                          y_patches=y_tune_train)
        np.savez_compressed(tune_val_path,
                          x_patches=x_tune_val,
                          y_patches=y_tune_val)
        np.savez_compressed(final_test_path,
                          x_patches=x_final_test,
                          y_patches=y_final_test)
        
        if self.debug:
            print("\n=== Test Set Split Results ===")
            print(f"MSLTuneTrain: {x_tune_train.shape[0]} patches")
            print(f"MSLTuneValid: {x_tune_val.shape[0]} patches")
            print(f"MSLTuneTest : {x_final_test.shape[0]} patches")
        
        return tune_train_path, tune_val_path, final_test_path

    @staticmethod
    def load_patch_split(split_path: str) -> tuple:
        """
        Load pre-saved .npz split and return (x_patches, y_patches) as numpy arrays.
        Args:
            split_path: path to one of msl_train_patches.npz, etc.
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
    save_dir = "data/MSL/patches"
    
    # Create extractor and process data
    extractor = AnomalyPatchExtractor(debug=True)
    
    # 1) Process training and validation sets
    print("\n=== Processing Training and Validation Sets ===")
    train_path, val_path = extractor.extract_and_store_all(save_dir=save_dir)
    AnomalyPatchExtractor.verify_saved_patch_split(train_path)
    AnomalyPatchExtractor.verify_saved_patch_split(val_path)
    
    # 2) Process test set
    print("\n=== Processing Test Set ===")
    x_test_patches, y_test_patches = extractor.extract_test_set(save_dir=save_dir)
    if extractor.debug:
        print("\n=== Test patch shapes ===")
        print(f"Test x patches: {x_test_patches.shape}, Test y patches: {y_test_patches.shape}")
    
    # 3) Split test set into tuning and final test sets
    print("\n=== Splitting Test Set into Tuning and Final Test Sets ===")
    tune_train_path, tune_val_path, final_test_path = extractor.split_test_into_tune_and_final(
        x_test_patches, y_test_patches, save_dir=save_dir
    )
    AnomalyPatchExtractor.verify_saved_patch_split(tune_train_path)
    AnomalyPatchExtractor.verify_saved_patch_split(tune_val_path)
    AnomalyPatchExtractor.verify_saved_patch_split(final_test_path) 