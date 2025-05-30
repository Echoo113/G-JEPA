import numpy as np
from preprocess.datatool import DataTool
import os
import json

# ====== Global Parameters ======
PATCH_SIZE        = 30     # Time steps per patch
STRIDE            = 10     # Sliding window stride
CONTEXT_RATIO     = 0.8    # First 80% of patches as context
LONG_TERM_RATIO   = 0.05   # 1/ratio, similar to stride
DEFAULT_FILENAME  = "data/SOLAR/solar_10_minutes_dataset.csv"
SHORT_TERM_RATIO  = 0.3

class PatchExtractor:
    def __init__(
        self,
        filename: str = DEFAULT_FILENAME,
        patch_size: int = PATCH_SIZE,
        stride: int = STRIDE,
        context_ratio: float = CONTEXT_RATIO,
        long_term_ratio: float = LONG_TERM_RATIO,
        short_term_ratio: float = SHORT_TERM_RATIO
    ):
        self.data_tool        = DataTool(filename, debug=False)
        self.patch_size       = patch_size
        self.stride           = stride
        self.context_ratio    = context_ratio
        self.long_term_ratio  = long_term_ratio
        self.short_term_ratio = short_term_ratio

    def _split_context_future(self, patches: np.ndarray) -> tuple:
        N = patches.shape[0]
        split_idx = int(N * self.context_ratio)
        return patches[:split_idx], patches[split_idx:]

    def _select_long_term(self, patches: np.ndarray) -> np.ndarray:
        M = patches.shape[0]
        step = max(1, int(1 / self.long_term_ratio))
        indices = list(range(0, M, step))
        return patches[indices]

    def _select_short_term(self, patches: np.ndarray, tail: bool) -> np.ndarray:
        """
        Select short-term patches from the beginning or end based on ratio
        Args:
            patches: np.ndarray, shape=(M, patch_size, C)
            tail: True=select from end, False=select from beginning
        Returns:
            np.ndarray of short-term patches
        """
        M = patches.shape[0]
        k = max(1, int(M * self.short_term_ratio))
        return patches[-k:] if tail else patches[:k]

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
        
        # Save parameters to a separate json file
        params = {
            'patch_size': self.patch_size,
            'stride': self.stride,
            'context_ratio': self.context_ratio,
            'long_term_ratio': self.long_term_ratio,
            'short_term_ratio': self.short_term_ratio
        }
        params_path = save_path.replace('.npz', '_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        
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
        Verify saved patches and parameters
        Args:
            load_path: Load path, e.g., 'data/patches/solar_patches.npz'
        """
        # Load patches
        data = np.load(load_path)
        
        # Print shapes
        print("\nPatch shapes:")
        for key in ['long_term_context', 'long_term_future', 'short_term_context', 'short_term_future']:
            print(f"{key}: {data[key].shape}")
        
        # Load and print parameters
        params_path = load_path.replace('.npz', '_params.json')
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        print("\nParameters:")
        for key, value in params.items():
            print(f"{key}: {value}")

    def extract_patches(self, debug: bool = False) -> tuple:
        # 1) Load and normalize data
        data = self.data_tool.get_data()  # (T, C)
        T, C = data.shape

        # 2) Extract multivariate patches
        all_patches = []
        for start in range(0, T - self.patch_size + 1, self.stride):
            all_patches.append(data[start:start + self.patch_size])
        all_patches = np.stack(all_patches)  # (N, patch_size, C)

        # 3) Split into context & pseudo-future
        context_patches, future_patches = self._split_context_future(all_patches)

        # 4) Long-term sampling
        long_term_context = self._select_long_term(context_patches)
        long_term_future  = self._select_long_term(future_patches)

        # 5) Short-term selection: context from end, future from beginning
        short_term_context = self._select_short_term(context_patches, tail=True)
        short_term_future  = self._select_short_term(future_patches, tail=False)

        # 6) Debug information
        if debug:
            patches_dict = {
                'long_term_context': long_term_context,
                'long_term_future': long_term_future,
                'short_term_context': short_term_context,
                'short_term_future': short_term_future
            }
            self.debug_patches(patches_dict, data.shape)

        return (
            long_term_context,
            long_term_future,
            short_term_context,
            short_term_future
        )

if __name__ == "__main__":
    extractor = PatchExtractor()
    ltc, ltf, stc, stf = extractor.extract_patches(debug=True)
    
    # Store patches
    patches_dict = {
        'long_term_context': ltc,
        'long_term_future': ltf,
        'short_term_context': stc,
        'short_term_future': stf
    }
    save_path = 'data/SOLAR/patches/solar_patches.npz'
    extractor.store_patches(patches_dict, save_path, compressed=True)
    
    # Verify saved data
    extractor.verify_saved_patches(save_path)
