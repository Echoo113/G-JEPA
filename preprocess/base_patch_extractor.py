import torch
import os
from data_processing.dataset_processor import DatasetProcessor

# Global parameters for patch extraction
PATCH_SIZE = 30
WINDOW_SIZE = 5


class BasePatchExtractor:
    def __init__(self, csv_path, output_dir):
        """
        Initialize the base patch extractor
        
        Args:
            csv_path: path to the CSV file
            output_dir: directory to save extracted patches
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.processor = DatasetProcessor(csv_path)
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_short_term_windows(self, length):
        """
        Generate window indices for short-term patches (continuous extraction within a range).
        To be implemented by subclasses.
        
        Args:
            length: length of the sequence
            
        Returns:
            list of window indices for short-term patches
        """
        raise NotImplementedError
    
    def generate_long_term_windows(self, length):
        """
        Generate window indices for long-term patches (sparse extraction within a range).
        To be implemented by subclasses.
        
        Args:
            length: length of the sequence
            
        Returns:
            list of window indices for long-term patches
        """
        raise NotImplementedError
    
    def generate_windows(self, length):
        """
        Generate window indices for both short-term and long-term patches
        
        Args:
            length: length of the sequence
            
        Returns:
            dictionary containing window indices for short and long patches
        """
        return {
            "short": self.generate_short_term_windows(length),
            "long": self.generate_long_term_windows(length)
        }
    
    def extract_patches_from_df(self, df):
        """
        Extract patches from a DataFrame
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            list of patch tensors
        """
        X_all = []
        for sensor_col in df.columns:
            signal = self.processor.process_signal(df[sensor_col])
            num_patches = (len(signal) - PATCH_SIZE - WINDOW_SIZE) // PATCH_SIZE
            for i in range(num_patches):
                start = i * PATCH_SIZE
                end = start + PATCH_SIZE + WINDOW_SIZE
                full_patch = signal[start:end]
                patch_segment = full_patch[WINDOW_SIZE:]  # skip context part
                x = torch.tensor(patch_segment, dtype=torch.float32)
                X_all.append(x)
        if X_all:
            X_all_tensor = torch.stack(X_all)
        else:
            X_all_tensor = torch.empty((0, PATCH_SIZE))
        return X_all_tensor
    
    def store_patches_by_group(self, group_df, windows, group_name):
        """
        Store patches for a group
        
        Args:
            group_df: DataFrame containing the group data
            windows: dictionary of window indices
            group_name: name of the group
            
        Returns:
            dictionary containing patches and metadata
        """
        patches = {
            "short": {
                "patches": [],
                "sensors": [],
                "indices": [],
                "count": 0
            },
            "long": {
                "patches": [],
                "sensors": [],
                "indices": [],
                "count": 0
            }
        }
        
        for sensor_col in group_df.columns:
            signal = self.processor.process_signal(group_df[sensor_col])
            
            for window_type in ["short", "long"]:
                for start_idx in windows[window_type]:
                    end_idx = start_idx + PATCH_SIZE + WINDOW_SIZE
                    if end_idx <= len(signal):
                        full_patch = signal[start_idx:end_idx]
                        patch_segment = full_patch[WINDOW_SIZE:]
                        x = torch.tensor(patch_segment, dtype=torch.float32)
                        
                        patches[window_type]["patches"].append(x)
                        patches[window_type]["sensors"].append(sensor_col)
                        patches[window_type]["indices"].append(start_idx)
                        patches[window_type]["count"] += 1
        
        for window_type in patches:
            if patches[window_type]["patches"]:
                patches[window_type]["patches"] = torch.stack(patches[window_type]["patches"])
            else:
                patches[window_type]["patches"] = torch.empty((0, PATCH_SIZE))
        
        return patches
    
    def process_and_save(self):
        """
        Process the data and save the patches
        """
        print("Starting patch extraction by group...")
        groups = self.processor.group_columns_by_prefix()
        print("\nAnalyzing windows for each group...")
        
        for prefix, group_df in groups.items():
            seq_length = self.processor.get_sequence_length(prefix)
            windows = self.generate_windows(seq_length)
            patches = self.store_patches_by_group(group_df, windows, prefix)
            
            print(f"\nGroup {prefix}:")
            print(f"  Sequence length: {seq_length}")
            for window_type in ["short", "long"]:
                print(f"  {window_type.capitalize()}: {patches[window_type]['count']} patches")
            
            save_path = os.path.join(self.output_dir, f"{prefix}_{self.get_save_suffix()}.pkl")
            self.processor.save_processed_data(patches, save_path)
            print(f"  Saved to: {save_path}")
    
    def get_save_suffix(self):
        """
        Get the suffix for saved files.
        To be implemented by subclasses.
        """
        raise NotImplementedError 