import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataTool:
    def __init__(self, path, debug=False):
        self.path = path
        self.data = None
        self.scaled_data = None
        self.scaler = None
        self.debug = debug

    def load(self):
        """Load raw data without standardization"""
        if self.path.endswith(".npy"):
            self.data = np.load(self.path)
        elif self.path.endswith(".csv"):
            df = pd.read_csv(self.path, header=0)
            self.data = df.values
        else:
            raise ValueError("Unsupported file format. Use .csv or .npy")
        return self.data

    def standardize(self):
        """Standardize the data using StandardScaler"""
        if self.data is None:
            self.load()

        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        if self.debug:
            self.printDebug()
        
        return self.scaled_data

    def get_data(self):
        """Get standardized data"""
        if self.scaled_data is None:
            self.load()
            return self.standardize()
        return self.scaled_data
    
    def printDebug(self):
        print("Data shape:", self.data.shape)
        print("Scaled data shape:", self.scaled_data.shape)
        #print the first 5 rows of the data
        print("First 5 rows of the data:")
        print(self.data[:5, :5])
        print("First 5 rows of the scaled data:")
        print(self.scaled_data[:5, :5])
