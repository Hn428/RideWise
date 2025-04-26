import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self, filepath):
        """
        Load data from csv file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        # Load the data
        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df):
        """
        Clean the data by handling missing values and outliers
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Convert datetime string to datetime object
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Handle missing values in numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            # Replace missing values with median
            data[col] = data[col].fillna(data[col].median())
            
            # Handle outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers to the bounds
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        # Handle missing values in categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Replace missing values with mode
            data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def split_data(self, df, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Split data into train, validation and test sets
        """
        if train_ratio + valid_ratio + test_ratio != 1.0:
            raise ValueError("Train, validation and test ratios must sum to 1")
        
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Shuffle the data
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split the data
        train_size = int(len(data) * train_ratio)
        valid_size = int(len(data) * valid_ratio)
        
        train_data = data[:train_size]
        valid_data = data[train_size:train_size+valid_size]
        test_data = data[train_size+valid_size:]
        
        # Save the splits
        train_data.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(self.data_dir, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(self.data_dir, 'test.csv'), index=False)
        
        return train_data, valid_data, test_data
    
    def resample_data(self, df, target_col, strategy='upsample'):
        """
        Resample imbalanced data
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Get class distribution
        class_counts = data[target_col].value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        if strategy == 'upsample':
            # Upsample minority classes
            resampled_dfs = [data[data[target_col] == majority_class]]
            
            for cls, count in class_counts.items():
                if cls != majority_class:
                    class_df = data[data[target_col] == cls]
                    resampled_class = class_df.sample(majority_count, replace=True, random_state=42)
                    resampled_dfs.append(resampled_class)
            
            return pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
        elif strategy == 'downsample':
            # Downsample majority class
            minority_class = class_counts.idxmin()
            minority_count = class_counts.min()
            
            resampled_dfs = []
            for cls, count in class_counts.items():
                class_df = data[data[target_col] == cls]
                if cls == majority_class:
                    resampled_class = class_df.sample(minority_count, random_state=42)
                else:
                    resampled_class = class_df
                resampled_dfs.append(resampled_class)
            
            return pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
        else:
            raise ValueError("Strategy must be 'upsample' or 'downsample'")
    
    def preprocess_for_training(self, filepath, target_col=None, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        """
        Full preprocessing pipeline for training data
        """
        # Load the data
        df = self.load_data(filepath)
        
        # Clean the data
        clean_df = self.clean_data(df)
        
        # Split the data
        train_data, valid_data, test_data = self.split_data(
            clean_df, 
            train_ratio=train_ratio, 
            valid_ratio=valid_ratio, 
            test_ratio=test_ratio
        )
        
        # Resample training data if target column is provided
        if target_col is not None:
            train_data = self.resample_data(train_data, target_col)
            train_data.to_csv(os.path.join(self.data_dir, 'train_resampled.csv'), index=False)
        
        return train_data, valid_data, test_data 