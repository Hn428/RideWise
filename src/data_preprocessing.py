import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Tuple, Optional


class DataPreprocessor:
    """
    Class for loading, cleaning, and preprocessing ride and weather datasets.
    """
    
    def __init__(self, rides_path: str, weather_path: str):
        """
        Initialize the data preprocessor.
        
        Args:
            rides_path: Path to the ride data CSV file
            weather_path: Path to the weather data CSV file
        """
        self.rides_path = rides_path
        self.weather_path = weather_path
        self.rides_df = None
        self.weather_df = None
        self.merged_df = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ride and weather data from CSV files.
        
        Returns:
            Tuple containing ride dataframe and weather dataframe
        """
        print(f"Loading rides data from {self.rides_path}")
        self.rides_df = pd.read_csv(self.rides_path)
        print(f"Loaded {len(self.rides_df)} ride entries")
        
        print(f"Loading weather data from {self.weather_path}")
        self.weather_df = pd.read_csv(self.weather_path)
        print(f"Loaded {len(self.weather_df)} weather entries")
        
        return self.rides_df, self.weather_df
    
    def clean_rides_data(self) -> pd.DataFrame:
        """
        Clean the ride data, handling missing values and converting timestamps.
        
        Returns:
            Cleaned ride dataframe
        """
        if self.rides_df is None:
            raise ValueError("Ride data not loaded. Call load_data() first.")
        
        # Create a copy to avoid modifying the original
        df = self.rides_df.copy()
        
        # Check for 'time_stamp' column and rename to 'timestamp' if needed
        if 'time_stamp' in df.columns and 'timestamp' not in df.columns:
            df.rename(columns={'time_stamp': 'timestamp'}, inplace=True)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            # Check if timestamp is in Unix milliseconds format (large integers)
            if df['timestamp'].dtype != 'datetime64[ns]':
                try:
                    if df['timestamp'].astype(str).str.len().max() > 10:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    print(f"Warning: Error converting timestamps: {e}")
        
        # Handle missing values
        if 'surge_multiplier' in df.columns:
            # Fill missing surge multipliers with 1.0 (no surge)
            df['surge_multiplier'] = df['surge_multiplier'].fillna(1.0)
            
            # Remove extreme outliers in surge_multiplier (e.g., values > 5 might be errors)
            q1 = df['surge_multiplier'].quantile(0.25)
            q3 = df['surge_multiplier'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            # Filter out extreme values, but keep track of how many we remove
            outliers = df[df['surge_multiplier'] > upper_bound]
            if len(outliers) > 0:
                print(f"Removing {len(outliers)} outliers with surge > {upper_bound:.2f}")
                df = df[df['surge_multiplier'] <= upper_bound]
        
        # Handle distance values
        if 'distance' in df.columns:
            # Remove negative or extremely large distances
            df = df[df['distance'] >= 0]
            df = df[df['distance'] < df['distance'].quantile(0.99)] # Remove top 1% as outliers
        
        # Remove duplicate entries
        duplicates = df.duplicated()
        if duplicates.sum() > 0:
            print(f"Removing {duplicates.sum()} duplicate entries")
            df = df.drop_duplicates()
        
        self.rides_df = df
        return df
    
    def clean_weather_data(self) -> pd.DataFrame:
        """
        Clean the weather data, handling missing values and converting timestamps.
        
        Returns:
            Cleaned weather dataframe
        """
        if self.weather_df is None:
            raise ValueError("Weather data not loaded. Call load_data() first.")
        
        # Create a copy to avoid modifying the original
        df = self.weather_df.copy()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                print(f"Warning: Error converting timestamps: {e}")
        
        # Fill missing values in weather data
        if 'precipitation_type' in df.columns:
            # Fill missing precipitation type with 'clear'
            df['precipitation_type'] = df['precipitation_type'].fillna('clear')
        
        # Handle numeric columns
        numeric_cols = ['temperature', 'humidity', 'wind_speed']
        for col in numeric_cols:
            if col in df.columns:
                # Fill missing values with the median
                if df[col].isna().sum() > 0:
                    median_value = df[col].median()
                    print(f"Filling {df[col].isna().sum()} missing values in {col} with median ({median_value:.2f})")
                    df[col] = df[col].fillna(median_value)
        
        self.weather_df = df
        return df
    
    def merge_datasets(self, time_window: str = '1H') -> pd.DataFrame:
        """
        Merge the ride and weather datasets based on timestamp proximity.
        
        Args:
            time_window: Time window for matching entries (default: '1H' = 1 hour)
            
        Returns:
            Merged dataframe
        """
        if self.rides_df is None or self.weather_df is None:
            raise ValueError("Data not loaded or cleaned. Call load_data() and clean_*_data() first.")
        
        # Ensure both dataframes have their timestamps sorted
        rides_sorted = self.rides_df.sort_values('timestamp')
        weather_sorted = self.weather_df.sort_values('timestamp')
        
        # Merge using asof join - this matches each ride to the closest weather reading
        print(f"Merging datasets with time window: {time_window}")
        merged_df = pd.merge_asof(
            rides_sorted,
            weather_sorted,
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(time_window)
        )
        
        # Check how many rides didn't get matched with weather data
        unmatched = merged_df['temperature'].isna().sum() if 'temperature' in merged_df.columns else 0
        if unmatched > 0:
            print(f"Warning: {unmatched} rides ({unmatched/len(merged_df)*100:.2f}%) couldn't be matched with weather data")
            
            # For unmatched entries, fill with most common weather values
            for col in weather_sorted.columns:
                if col != 'timestamp' and col in merged_df.columns:
                    if merged_df[col].dtype == 'object':
                        # For categorical columns, use most frequent value
                        most_common = weather_sorted[col].mode()[0]
                        merged_df[col] = merged_df[col].fillna(most_common)
                    else:
                        # For numerical columns, use median
                        median_val = weather_sorted[col].median()
                        merged_df[col] = merged_df[col].fillna(median_val)
        
        self.merged_df = merged_df
        print(f"Final merged dataset contains {len(merged_df)} entries")
        
        return merged_df


if __name__ == "__main__":
    # Example usage
    rides_path = os.path.join("data", "cab_rides.csv")
    weather_path = os.path.join("data", "weather.csv")
    
    preprocessor = DataPreprocessor(rides_path, weather_path)
    preprocessor.load_data()
    preprocessor.clean_rides_data()
    preprocessor.clean_weather_data()
    merged_data = preprocessor.merge_datasets()
    
    print("\nMerged Data Sample:")
    print(merged_data.head())
    
    print("\nMerged Data Info:")
    print(merged_data.info()) 