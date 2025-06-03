import numpy as np
from preprocess.datatool import DataTool
import os

# ====== Global Constants ======
PATCH_SIZE        = 30     # Time steps per patch
STRIDE            = 10     # Sliding window stride

LONG_TERM_RATIO   = 0.4   # Take  10% of total patches
DEFAULT_FILENAME  = "data/SOLAR/solar_10_minutes_dataset.csv"
LONG_TERM_QUANTILE = 0.75   # 75% context, 25% future
TRAIN_RANGE       = 0.7    # Use first 80% of patches for training
VALIDATION_RANGE  = 0.1    # Use last 10% of patches for validation

class PatchExtractor:
    def __init__(
        self,
        filename: str = DEFAULT_FILENAME,
        patch_size: int = PATCH_SIZE,
        stride: int = STRIDE,
        long_term_ratio: float = LONG_TERM_RATIO,
        long_term_quantile: float = LONG_TERM_QUANTILE
    ):
        self.data_tool        = DataTool(filename, debug=False)
        self.patch_size       = patch_size
        self.stride           = stride

        self.long_term_ratio  = long_term_ratio


    def _split_context_future_disjoint(self, patches: np.ndarray) -> tuple:
        """
        Split patches into context and future using long-term quantile rule:
        - First X% → context
        - Last (1-X)% → future
        """
        N = patches.shape[0]
        split_idx = int(N * LONG_TERM_QUANTILE)
        context_part = patches[:split_idx]
        future_part  = patches[split_idx:]
        return context_part, future_part

    def _select_long_term(self, patches: np.ndarray) -> np.ndarray:
        """
        Select long-term patches by taking one patch every N% of total patches
        Args:
            patches: np.ndarray, shape=(M, patch_size, C)
        Returns:
            np.ndarray of selected patches
        """
        M = patches.shape[0]
        # Calculate step size based on the ratio
        # e.g., if ratio=0.01, we want to take 1% of the patches
        num_patches = max(1, int(M * self.long_term_ratio))
        step = max(1, M // num_patches)
        indices = list(range(0, M, step))
        return patches[indices]

    def _select_short_term(self, patches: np.ndarray, tail: bool) -> tuple:
        """
        Construct short-term patches:
        - For every group: append 3 context patches (as individual samples)
        - Then append 1 future patch
        Returns:
            short_term_context: (N_ctx, patch_size, C)
            short_term_future:  (N_future, patch_size, C)
        """
        M = patches.shape[0]
        group_size = 3
        context_patches = []
        future_patches = []

        i = 0
        group_idx = 0
        while i + group_size < M:
            for j in range(group_size):
                context_patches.append(patches[i + j])  # append individually

            future_patches.append(patches[i + group_size])  # one future patch

            i += group_size + 1
            group_idx += 1

        context_patches = np.stack(context_patches)  # shape: (3M, patch_size, C)
        future_patches  = np.stack(future_patches)   # shape: (M, patch_size, C)

        return context_patches, future_patches

    def store_patches(self, patches_dict: dict, save_path: str, compressed: bool = True):
        """
        Store all patches in a .npz file
        Args:
            patches_dict: Dictionary containing all patches
            save_path: Save path, e.g., 'data/patches/solar_patches.npz'
            compressed: Whether to use compression
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save patches
        if compressed:
            np.savez_compressed(save_path, **patches_dict)
        else:
            np.savez(save_path, **patches_dict)

    @staticmethod
    def load_patches(load_path: str) -> tuple:
        """
        Load all patches from .npz file
        Args:
            load_path: Load path, e.g., 'data/patches/solar_patches.npz'
        Returns:
            tuple: (long_term_context, long_term_future, short_term_context, short_term_future)
        """
        data = np.load(load_path)
        return (
            data['long_term_context'],
            data['long_term_future'],
            data['short_term_context'],
            data['short_term_future']
        )

    def debug_patches(self, patches, data_shape):
        """Print debug information about the extracted patches."""
        if isinstance(patches, dict):
            # If patches is a dictionary (stored patches)
            for key, value in patches.items():
                print(f"{key}: {value.shape}")
        else:
            # If patches is a numpy array (all_patches)
            print(f"Total patches: {len(patches)}")
            print(f"Patch shape: {patches[0].shape}")

    def verify_saved_patches(self, load_path: str):
        """
        Verify saved patches by printing their shapes
        Args:
            load_path: Load path, e.g., 'data/patches/solar_patches.npz'
        """
        # Load patches
        data = np.load(load_path)
        
        # Print shapes
        print("\nPatch shapes:")
        for key in data.keys():
            print(f"{key}: {data[key].shape}")

    def extract_patches(self, debug: bool = False) -> tuple:
        # 1) Load and normalize data
        data = self.data_tool.get_data()  # (T, C)
        T, C = data.shape

        # 2) Extract multivariate patches
        all_patches = []
        for start in range(0, T - self.patch_size + 1, self.stride):
            all_patches.append(data[start:start + self.patch_size])
        all_patches = np.stack(all_patches)  # (N, patch_size, C)

        # 2.5) Only use first 80% of patches for training
        train_size = int(len(all_patches) * TRAIN_RANGE)
        train_patches = all_patches[:train_size]

        # 3) Split into context & future using disjoint rule
        train_context_patches, train_future_patches = self._split_context_future_disjoint(train_patches)

        # 4) Long-term sampling: apply ratio to each part separately
        train_long_term_context = self._select_long_term(train_context_patches)  # 75% * ratio
        train_long_term_future  = self._select_long_term(train_future_patches)   # 25% * ratio

        # 5) Short-term selection: new logic
        train_short_term_context, train_short_term_future = self._select_short_term(train_patches, tail=False)

        # 6) Debug information
        if debug:
            train_dict = {
                'long_term_context': train_long_term_context,
                'long_term_future': train_long_term_future,
                'short_term_context': train_short_term_context,
                'short_term_future': train_short_term_future
            }
            self.debug_patches(train_dict, data.shape)
    
            print(f"\nTraining range info:")
            print(f"Total patches: {len(all_patches)}")
            print(f"Training patches: {train_size} ({TRAIN_RANGE*100}%)")

        return (
            train_long_term_context,
            train_long_term_future,
            train_short_term_context,
            train_short_term_future
        )

    def extract_validation_patches(self, all_patches: np.ndarray, debug: bool = False) -> tuple:
        """
        Extract validation patches from the validation portion of data
        Only extracts long-term context and future patches
        """
        # Get validation portion (10% after training)
        val_start = int(len(all_patches) * TRAIN_RANGE)
        val_end = int(len(all_patches) * (TRAIN_RANGE + VALIDATION_RANGE))
        val_patches = all_patches[val_start:val_end]

        # Split into context & future using disjoint rule
        val_context_patches, val_future_patches = self._split_context_future_disjoint(val_patches)

        # Long-term sampling
        val_long_term_context = self._select_long_term(val_context_patches)
        val_long_term_future = self._select_long_term(val_future_patches)

        if debug:
            val_dict = {
                'long_term_context': val_long_term_context,
                'long_term_future': val_long_term_future
            }
            print(f"\nValidation patch counts:")
            print(f"Long-term context patches: {len(val_long_term_context)} (from {len(val_context_patches)} total)")
            print(f"Long-term future patches: {len(val_long_term_future)} (from {len(val_future_patches)} total)")
            print(f"Context/Future ratio: {len(val_long_term_context)/len(val_long_term_future):.2f}")

        return val_long_term_context, val_long_term_future

    def extract_test_patches(self, all_patches: np.ndarray, debug: bool = False) -> tuple:
        """
        Extract test patches from the test portion of data
        Only extracts long-term context and future patches
        """
        # Get test portion (last 10%)
        test_start = int(len(all_patches) * (TRAIN_RANGE + VALIDATION_RANGE))
        test_patches = all_patches[test_start:]

        # Split into context & future using disjoint rule
        test_context_patches, test_future_patches = self._split_context_future_disjoint(test_patches)

        # Long-term sampling
        test_long_term_context = self._select_long_term(test_context_patches)
        test_long_term_future = self._select_long_term(test_future_patches)

        if debug:
            test_dict = {
                'long_term_context': test_long_term_context,
                'long_term_future': test_long_term_future
            }
            print(f"\nTest patch counts:")
            print(f"Long-term context patches: {len(test_long_term_context)} (from {len(test_context_patches)} total)")
            print(f"Long-term future patches: {len(test_long_term_future)} (from {len(test_future_patches)} total)")
            print(f"Context/Future ratio: {len(test_long_term_context)/len(test_long_term_future):.2f}")

        return test_long_term_context, test_long_term_future

if __name__ == "__main__":
    extractor = PatchExtractor()
    
    # 1) Extract all patches first
    data = extractor.data_tool.get_data()
    T, C = data.shape
    all_patches = []
    for start in range(0, T - PATCH_SIZE + 1, STRIDE):
        all_patches.append(data[start:start + PATCH_SIZE])
    all_patches = np.stack(all_patches)

    # 2) Extract training patches
    ltc, ltf, stc, stf = extractor.extract_patches(debug=True)
    train_dict = {
        'long_term_context': ltc,
        'long_term_future': ltf,
        'short_term_context': stc,
        'short_term_future': stf
    }
    train_path = 'data/SOLAR/patches/solar_train.npz'
    extractor.store_patches(train_dict, train_path, compressed=True)

    # 3) Extract validation patches
    val_context, val_future = extractor.extract_validation_patches(all_patches, debug=True)
    val_dict = {
        'long_term_context': val_context,
        'long_term_future': val_future
    }
    validation_path = 'data/SOLAR/patches/solar_val.npz'
    extractor.store_patches(val_dict, validation_path, compressed=True)

    # 4) Extract test patches
    test_context, test_future = extractor.extract_test_patches(all_patches, debug=True)
    test_dict = {
        'long_term_context': test_context,
        'long_term_future': test_future
    }
    test_path = 'data/SOLAR/patches/solar_test.npz'
    extractor.store_patches(test_dict, test_path, compressed=True)

    # 5) Verify saved data
    print("\nVerifying training data:")
    extractor.verify_saved_patches(train_path)
    print("\nVerifying validation data:")
    extractor.verify_saved_patches(validation_path)
    print("\nVerifying test data:")
    extractor.verify_saved_patches(test_path)
