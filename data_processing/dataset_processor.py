import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import re
import os
import pickle

class DatasetProcessor:
    def __init__(self, csv_path):
        """
        Initialize the dataset processor
        
        Args:
            csv_path: path to the CSV file containing the data
        """
        self.csv_path = csv_path
        self.df = None
        self.groups = None
        
    def load_and_clean_data(self):
        """
        Load the CSV file and perform initial cleaning
        - Remove Time column
        - Convert units (remove µA)
        - Convert to float
        """
        self.df = pd.read_csv(self.csv_path)
        if 'Time' in self.df.columns:
            self.df = self.df.drop(columns=["Time"])
        
        # Convert all columns to float, removing µA units
        for col in self.df.columns:
            if self.df[col].dtype == object and self.df[col].str.contains("µA").any():
                self.df[col] = self.df[col].str.replace(" µA", "").astype(float)
        
        return self.df
    
    def group_columns_by_prefix(self):
        """
        Group columns by their prefix (u1, u2, etc.)
        Returns a dictionary where key is prefix and value is the group DataFrame
        """
        if self.df is None:
            self.load_and_clean_data()
            
        prefix_dict = {}
        for col in self.df.columns:
            m = re.match(r"u(\d)\d+", col)
            if m:
                prefix = f"u{m.group(1)}"  # take only the first digit
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = []
                prefix_dict[prefix].append(col)
        
        # Convert to DataFrame
        for prefix in prefix_dict:
            prefix_dict[prefix] = self.df[prefix_dict[prefix]]
        
        self.groups = prefix_dict
        return prefix_dict
    
    def standardize_signal(self, signal):
        """
        Standardize a signal using StandardScaler
        
        Args:
            signal: numpy array of signal values
            
        Returns:
            standardized signal
        """
        scaler = StandardScaler()
        signal_reshaped = signal.reshape(-1, 1)
        return scaler.fit_transform(signal_reshaped).flatten()
    
    def process_signal(self, signal):
        """
        Process a signal: convert to float and standardize
        
        Args:
            signal: signal values (can be string with units or float)
            
        Returns:
            processed and standardized signal
        """
        if isinstance(signal, pd.Series):
            signal = signal.astype(str).str.replace(" µA", "").astype(float).values
        return self.standardize_signal(signal)
    
    def get_sequence_length(self, group_name):
        """
        Get the length of sequence for a specific group
        
        Args:
            group_name: name of the group (e.g., 'u1')
            
        Returns:
            length of the sequence
        """
        if self.groups is None:
            self.group_columns_by_prefix()
        return len(self.groups[group_name].iloc[:, 0])
    
    def save_processed_data(self, data, save_path):
        """
        Save processed data to a pickle file
        
        Args:
            data: data to save
            save_path: path to save the data
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_processed_data(self, load_path):
        """
        Load processed data from a pickle file
        
        Args:
            load_path: path to load the data from
            
        Returns:
            loaded data
        """
        with open(load_path, 'rb') as f:
            return pickle.load(f) 