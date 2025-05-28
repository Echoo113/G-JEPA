import os
from preprocess.base_patch_extractor import BasePatchExtractor, PATCH_SIZE, WINDOW_SIZE

# Global parameters for window generation
SHORT_RANGE = (0.96, 0.98)  # 短期patches的范围 (start_ratio, end_ratio)
LONG_RANGE = (0.8, 1.0)   # 长期patches的范围 (start_ratio, end_ratio)
LONG_STEP = 0.02          # 长期patches的步长比例
STRIDE = 12

class PseudoFuturePatchExtractor(BasePatchExtractor):
    def __init__(self, csv_path, output_dir):
        super().__init__(csv_path, output_dir)
    
    def generate_short_term_windows(self, length):
        """
        Generate window indices for short-term patches (continuous extraction within a range)
        
        Args:
            length: length of the sequence
            
        Returns:
            list of window indices for short-term patches
        """
        windows = []
        start_idx = int(length * SHORT_RANGE[0])
        end_idx = int(length * SHORT_RANGE[1])
        
        # 在指定范围内连续取patches
        current_idx = start_idx
        while current_idx + PATCH_SIZE + WINDOW_SIZE <= end_idx:
            windows.append(current_idx)
            current_idx += PATCH_SIZE+STRIDE
        
        return windows
    
    def generate_long_term_windows(self, length):
        """
        Generate window indices for long-term patches (sparse extraction within a range)
        
        Args:
            length: length of the sequence
            
        Returns:
            list of window indices for long-term patches
        """
        windows = []
        start_idx = int(length * LONG_RANGE[0])
        end_idx = int(length * LONG_RANGE[1])
        step_size = int(length * LONG_STEP)
        
        # 在指定范围内跳跃取patches
        current_idx = start_idx
        while current_idx + PATCH_SIZE + WINDOW_SIZE <= end_idx:
            windows.append(current_idx)
            current_idx += step_size
        
        return windows
    
    def get_save_suffix(self):
        return "pseudo_future_patches"

if __name__ == "__main__":
    csv_path = "/Users/echohe/Desktop/Research/spark/data/SWAT/SWAT_train.npy"
    output_dir = "data/SWAT/pseudo_future_patches"
    
    extractor = PseudoFuturePatchExtractor(
        csv_path=csv_path,
        output_dir=output_dir
    )
    
    extractor.process_and_save()
