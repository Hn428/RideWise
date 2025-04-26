import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List, Union
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from geopy.distance import geodesic

class DataProcessor:
    def __init__(self, rides_path: str, weather_path: str = None):
        self.rides_path = rides_path
        self.weather_path = weather_path
        self.rides_df = None
        self.weather_df = None
        self.merged_df = None
        self.encoders = {}
        self.location_clusters = None
        self.location_mapping = None

    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load and perform initial cleaning of the datasets."""
        # Load rides data
        self.rides_df = pd.read_csv(self.rides_path)
        
        # Rename 'time_stamp' to 'timestamp' if it exists
        if 'time_stamp' in self.rides_df.columns:
            self.rides_df.rename(columns={'time_stamp': 'timestamp'}, inplace=True)
        
        # Convert timestamp from unix milliseconds to datetime if needed
        if 'timestamp' in self.rides_df.columns and self.rides_df['timestamp'].dtype != 'datetime64[ns]':
            try:
                # Check if the timestamp is in unix milliseconds format (large integers)
                if self.rides_df['timestamp'].astype(str).str.len().max() > 10:
                    self.rides_df['timestamp'] = pd.to_datetime(self.rides_df['timestamp'], unit='ms')
                else:
                    self.rides_df['timestamp'] = pd.to_datetime(self.rides_df['timestamp'])
            except:
                pass
        
        # Clean ride data
        self._clean_ride_data()
        
        # Load weather data if file exists
        if self.weather_path and os.path.exists(self.weather_path):
            self.weather_df = pd.read_csv(self.weather_path)
            if 'timestamp' in self.weather_df.columns:
                self.weather_df['timestamp'] = pd.to_datetime(self.weather_df['timestamp'])
            self._clean_weather_data()
            return self.rides_df, self.weather_df
        else:
            # Create a synthetic weather dataframe for demonstration purposes
            self.create_synthetic_weather_data()
            return self.rides_df, self.weather_df

    def _clean_ride_data(self):
        """Clean and preprocess the ride data."""
        if self.rides_df is None:
            return
        
        # Remove duplicate entries
        self.rides_df = self.rides_df.drop_duplicates()
        
        # Handle missing values
        if 'surge_multiplier' in self.rides_df.columns:
            # Fill missing surge multipliers with 1.0 (no surge)
            self.rides_df['surge_multiplier'] = self.rides_df['surge_multiplier'].fillna(1.0)
        
        # Remove outliers in surge multiplier (e.g., surge > 5.0 might be errors)
        if 'surge_multiplier' in self.rides_df.columns:
            surge_q1 = self.rides_df['surge_multiplier'].quantile(0.25)
            surge_q3 = self.rides_df['surge_multiplier'].quantile(0.75)
            surge_iqr = surge_q3 - surge_q1
            surge_upper = surge_q3 + (1.5 * surge_iqr)
            self.rides_df = self.rides_df[self.rides_df['surge_multiplier'] <= surge_upper]
        
        # Create location mapping if source/destination exists
        if 'source' in self.rides_df.columns and 'destination' in self.rides_df.columns:
            self._create_location_mapping()

    def _clean_weather_data(self):
        """Clean and preprocess the weather data."""
        if self.weather_df is None:
            return
        
        # Fill missing precipitation types with 'clear'
        if 'precipitation_type' in self.weather_df.columns:
            self.weather_df['precipitation_type'] = self.weather_df['precipitation_type'].fillna('clear')
        
        # Convert categorical weather data to encoded values
        cat_columns = ['precipitation_type']
        for col in cat_columns:
            if col in self.weather_df.columns:
                self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                encoded = self.encoders[col].fit_transform(self.weather_df[[col]])
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]],
                    index=self.weather_df.index
                )
                self.weather_df = pd.concat([self.weather_df, encoded_df], axis=1)

    def _create_location_mapping(self):
        """Create mapping from location names to coordinates (for demonstration)."""
        # In a real application, you would use geocoding APIs to get actual coordinates
        unique_locations = list(set(
            self.rides_df['source'].dropna().unique().tolist() + 
            self.rides_df['destination'].dropna().unique().tolist()
        ))
        
        # Create a simple mapping with made-up coordinates
        np.random.seed(42)  # For reproducibility
        base_lat, base_lon = 42.3601, -71.0589  # Boston coordinates
        
        self.location_mapping = {}
        for i, loc in enumerate(unique_locations):
            # Generate coordinates within a small radius of Boston
            lat_offset = np.random.uniform(-0.05, 0.05)
            lon_offset = np.random.uniform(-0.05, 0.05)
            self.location_mapping[loc] = (base_lat + lat_offset, base_lon + lon_offset)

    def create_synthetic_weather_data(self):
        """Create synthetic weather data when real data is not available."""
        if self.rides_df is not None:
            # Get unique timestamps from rides data, subsample for efficiency
            unique_timestamps = self.rides_df['timestamp'].drop_duplicates().sort_values()
            if len(unique_timestamps) > 1000:
                unique_timestamps = unique_timestamps.sample(1000)
            
            # Generate random weather data
            precipitation_types = ['clear', 'rain', 'snow', None]
            weather_data = {
                'timestamp': unique_timestamps,
                'temperature': np.random.uniform(0, 35, size=len(unique_timestamps)),
                'humidity': np.random.uniform(30, 100, size=len(unique_timestamps)),
                'wind_speed': np.random.uniform(0, 30, size=len(unique_timestamps)),
                'precipitation_type': [np.random.choice(precipitation_types) for _ in range(len(unique_timestamps))]
            }
            
            self.weather_df = pd.DataFrame(weather_data)
            
            # Clean the synthetic weather data
            self._clean_weather_data()
            
            # Save synthetic data for future use
            if not os.path.exists('weather.csv'):
                self.weather_df.to_csv('weather.csv', index=False)
                print("Created synthetic weather data and saved to weather.csv")
        else:
            raise ValueError("Rides data must be loaded before creating synthetic weather data")

    def merge_datasets(self, time_window: str = '1h') -> pd.DataFrame:
        """Merge rides and weather data based on timestamp proximity."""
        if self.weather_df is None:
            # If no weather data, just use the rides data and add dummy weather columns
            self.merged_df = self.rides_df.copy()
            self.merged_df['temperature'] = 20.0  # default temperature
            self.merged_df['humidity'] = 50.0  # default humidity
            self.merged_df['wind_speed'] = 5.0  # default wind speed
            self.merged_df['precipitation_type'] = 'clear'  # default precipitation
            return self.merged_df
            
        # Ensure both dataframes have timestamp as datetime
        if 'timestamp' in self.rides_df.columns and self.rides_df['timestamp'].dtype != 'datetime64[ns]':
            self.rides_df['timestamp'] = pd.to_datetime(self.rides_df['timestamp'])
            
        if 'timestamp' in self.weather_df.columns and self.weather_df['timestamp'].dtype != 'datetime64[ns]':
            self.weather_df['timestamp'] = pd.to_datetime(self.weather_df['timestamp'])
            
        # Sort both dataframes by timestamp
        self.rides_df = self.rides_df.sort_values('timestamp')
        self.weather_df = self.weather_df.sort_values('timestamp')
        
        # Merge using pandas merge_asof (nearest match within time window)
        self.merged_df = pd.merge_asof(
            self.rides_df,
            self.weather_df,
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(time_window)
        )
        
        return self.merged_df

    def engineer_features(self) -> pd.DataFrame:
        """Create new features for the model."""
        df = self.merged_df.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                             (df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
        
        # Time cyclic features (to capture daily and weekly cycles)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Weather-based features
        if 'precipitation_type' in df.columns:
            df['has_precipitation'] = (~df['precipitation_type'].isin(['clear', 'none', None])).astype(int)
            
            # Weather severity score (higher for rain/snow)
            weather_severity = {'clear': 0, 'none': 0, None: 0, 'rain': 1, 'snow': 2}
            df['weather_severity'] = df['precipitation_type'].map(lambda x: weather_severity.get(x, 0))
        
        # Weather interaction with time
        if all(col in df.columns for col in ['has_precipitation', 'is_rush_hour']):
            df['rain_during_rush'] = df['has_precipitation'] * df['is_rush_hour']
        
        if all(col in df.columns for col in ['temperature', 'is_weekend']):
            # Temperature interaction with weekend (hot weekends might have higher demand)
            df['temp_weekend_interaction'] = df['temperature'] * df['is_weekend']
        
        # Distance-based features
        if 'distance' in df.columns:
            df['distance_km'] = df['distance'] * 1.60934  # Convert miles to km
            
            # Distance categories
            df['distance_category'] = pd.cut(
                df['distance_km'], 
                bins=[0, 2, 5, 10, 20, 100], 
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        # Location-based features
        self._add_location_features(df)
        
        # Ride type features
        if 'cab_type' in df.columns:
            df['is_uber'] = (df['cab_type'] == 'Uber').astype(int)
            df['is_lyft'] = (df['cab_type'] == 'Lyft').astype(int)
        
        # Product type features
        if 'product_id' in df.columns:
            # One-hot encode product types
            product_dummies = pd.get_dummies(df['product_id'], prefix='product')
            df = pd.concat([df, product_dummies], axis=1)
        
        # Historical demand features
        if 'timestamp' in df.columns:
            # Group by hour and day of week to get average surge for similar times
            hourly_surge = df.groupby(['hour', 'day_of_week'])['surge_multiplier'].mean().reset_index()
            hourly_surge.columns = ['hour', 'day_of_week', 'avg_historical_surge']
            df = pd.merge(df, hourly_surge, on=['hour', 'day_of_week'], how='left')
        
        return df
    
    def _add_location_features(self, df: pd.DataFrame) -> None:
        """Add location-based features to the dataframe."""
        # Add pickup and dropoff clusters if coordinates are available
        if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude']):
            pickup_coords = df[['pickup_latitude', 'pickup_longitude']].dropna()
            if len(pickup_coords) > 0:
                df['pickup_cluster'] = self._cluster_locations(pickup_coords)
            
        if all(col in df.columns for col in ['dropoff_latitude', 'dropoff_longitude']):
            dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']].dropna()
            if len(dropoff_coords) > 0:
                df['dropoff_cluster'] = self._cluster_locations(dropoff_coords)
        
        # Handle location names (source/destination) by mapping to synthetic coordinates
        elif all(col in df.columns for col in ['source', 'destination']):
            if self.location_mapping is None:
                self._create_location_mapping()
            
            # Add source and destination clusters based on the mapping
            if 'source' in df.columns:
                df['source_cluster'] = df['source'].map(
                    lambda x: hash(str(self.location_mapping.get(x, (0, 0)))) % 10
                )
            
            if 'destination' in df.columns:
                df['destination_cluster'] = df['destination'].map(
                    lambda x: hash(str(self.location_mapping.get(x, (0, 0)))) % 10
                )
                
            # Calculate approximate distances between source and destination
            if 'source' in df.columns and 'destination' in df.columns and 'distance' not in df.columns:
                df['distance_km'] = df.apply(
                    lambda row: self._calculate_distance(
                        self.location_mapping.get(row['source'], (0, 0)),
                        self.location_mapping.get(row['destination'], (0, 0))
                    ),
                    axis=1
                )

    def _cluster_locations(self, coords: pd.DataFrame, n_clusters: int = 10) -> np.ndarray:
        """Cluster locations into regions using K-means."""
        if len(coords) < n_clusters:
            n_clusters = max(1, len(coords) // 2)
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Store the clustering model for future use
        self.location_clusters = kmeans
        
        # Return cluster for each location
        clusters = kmeans.fit_predict(coords)
        
        # Fill any missing values with the most common cluster
        if len(clusters) < len(coords):
            most_common = np.bincount(clusters).argmax()
            clusters = np.append(clusters, [most_common] * (len(coords) - len(clusters)))
            
        return clusters
    
    def _calculate_distance(self, source_coords: Tuple[float, float], 
                          dest_coords: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers."""
        try:
            return geodesic(source_coords, dest_coords).kilometers
        except:
            return 5.0  # Default distance if calculation fails

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training."""
        if self.merged_df is None:
            raise ValueError("Data not merged yet. Call merge_datasets() first.")
        
        # Engineer features if not already done
        if 'hour' not in self.merged_df.columns:
            self.engineer_features()
        
        # Create a copy of the dataframe for training
        df = self.merged_df.copy()
        
        # Get target variable
        if 'surge_multiplier' not in df.columns:
            raise ValueError("Target variable 'surge_multiplier' not found in data.")
        
        y = df['surge_multiplier']
        
        # Drop columns not useful for training
        drop_cols = ['timestamp', 'surge_multiplier', 'price']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Handle categorical columns with one-hot encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Convert categorical columns to string first to avoid CategoricalDtype issues
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        # Apply one-hot encoding to categorical columns
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Fill any remaining NA values with appropriate values
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean() if df[col].mean() is not np.nan else 0)
        
        # Ensure all columns have valid numeric data
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype('float64')
        
        return df, y

    def prepare_prediction_data(self, 
                              timestamp: datetime,
                              pickup_lat: Optional[float] = None,
                              pickup_lon: Optional[float] = None,
                              distance: float = 5.0,
                              source: Optional[str] = None,
                              destination: Optional[str] = None,
                              cab_type: Optional[str] = 'Uber',
                              product_id: Optional[str] = None,
                              weather_data: Optional[dict] = None) -> pd.DataFrame:
        """Prepare data for making a prediction."""
        data = {
            'timestamp': [timestamp],
            'distance': [distance]
        }
        
        # Add ride type if provided
        if cab_type:
            data['cab_type'] = [cab_type]
        
        # Add product if provided
        if product_id:
            data['product_id'] = [product_id]
        
        # Add location data if provided
        if pickup_lat is not None and pickup_lon is not None:
            data['pickup_latitude'] = [pickup_lat]
            data['pickup_longitude'] = [pickup_lon]
        elif source is not None and destination is not None:
            data['source'] = [source]
            data['destination'] = [destination]
            
            # Calculate distance if not provided
            if self.location_mapping and source in self.location_mapping and destination in self.location_mapping:
                source_coords = self.location_mapping[source]
                dest_coords = self.location_mapping[destination]
                data['distance'] = [self._calculate_distance(source_coords, dest_coords) / 1.60934]  # Convert to miles
        
        # Add weather data if provided
        if weather_data:
            for key, value in weather_data.items():
                data[key] = [value]
                
        # Create a temporary dataframe
        temp_df = pd.DataFrame(data)
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        
        # Add the same engineered features as training data
        temp_df['hour'] = temp_df['timestamp'].dt.hour
        temp_df['day_of_week'] = temp_df['timestamp'].dt.dayofweek
        temp_df['month'] = temp_df['timestamp'].dt.month
        temp_df['is_weekend'] = temp_df['day_of_week'].isin([5, 6]).astype(int)
        temp_df['is_rush_hour'] = ((temp_df['hour'] >= 7) & (temp_df['hour'] <= 9) | 
                                 (temp_df['hour'] >= 16) & (temp_df['hour'] <= 18)).astype(int)
        temp_df['is_night'] = ((temp_df['hour'] >= 22) | (temp_df['hour'] <= 4)).astype(int)
        
        # Time cyclic features
        temp_df['hour_sin'] = np.sin(2 * np.pi * temp_df['hour'] / 24)
        temp_df['hour_cos'] = np.cos(2 * np.pi * temp_df['hour'] / 24)
        temp_df['day_sin'] = np.sin(2 * np.pi * temp_df['day_of_week'] / 7)
        temp_df['day_cos'] = np.cos(2 * np.pi * temp_df['day_of_week'] / 7)
        
        # Distance features
        temp_df['distance_km'] = temp_df['distance'] * 1.60934
        
        # Weather features
        if weather_data and 'precipitation_type' in weather_data:
            temp_df['has_precipitation'] = 1 if weather_data['precipitation_type'] not in [None, 'clear', 'none'] else 0
            
            # Weather severity
            weather_severity = {'clear': 0, 'none': 0, None: 0, 'rain': 1, 'snow': 2}
            temp_df['weather_severity'] = weather_severity.get(weather_data['precipitation_type'], 0)
            
            # Weather interactions
            if 'is_rush_hour' in temp_df.columns:
                temp_df['rain_during_rush'] = temp_df['has_precipitation'] * temp_df['is_rush_hour']
                
            if 'temperature' in weather_data and 'is_weekend' in temp_df.columns:
                temp_df['temp_weekend_interaction'] = weather_data['temperature'] * temp_df['is_weekend']
        
        # Location features
        if source is not None and destination is not None and self.location_mapping:
            if source in self.location_mapping:
                temp_df['source_cluster'] = hash(str(self.location_mapping[source])) % 10
            else:
                temp_df['source_cluster'] = 0
                
            if destination in self.location_mapping:
                temp_df['destination_cluster'] = hash(str(self.location_mapping[destination])) % 10
            else:
                temp_df['destination_cluster'] = 0
        
        # Ride type features
        if 'cab_type' in temp_df.columns:
            temp_df['is_uber'] = (temp_df['cab_type'] == 'Uber').astype(int)
            temp_df['is_lyft'] = (temp_df['cab_type'] == 'Lyft').astype(int)
        
        # Product type features
        if 'product_id' in temp_df.columns and self.merged_df is not None:
            # Get all product types from training data
            if 'product_id' in self.merged_df.columns:
                product_types = self.merged_df['product_id'].unique()
                for product in product_types:
                    temp_df[f'product_{product}'] = (temp_df['product_id'] == product).astype(int)
        
        # Get historical average surge for this hour and day of week
        if self.merged_df is not None:
            hour = temp_df['hour'].iloc[0]
            day_of_week = temp_df['day_of_week'].iloc[0]
            
            # Calculate average surge for this hour and day
            mask = (self.merged_df['hour'] == hour) & (self.merged_df['day_of_week'] == day_of_week)
            if mask.sum() > 0:
                avg_surge = self.merged_df.loc[mask, 'surge_multiplier'].mean()
                temp_df['avg_historical_surge'] = avg_surge
            else:
                temp_df['avg_historical_surge'] = 1.0
        
        # Get the required features from our training data
        X_cols = self.prepare_training_data()[0].columns
        
        # Ensure all required columns exist
        for col in X_cols:
            if col not in temp_df.columns:
                # Add default values for missing columns
                if col.startswith('product_'):
                    temp_df[col] = 0
                elif col in ['temperature', 'avg_historical_surge']:
                    temp_df[col] = 20.0 if col == 'temperature' else 1.0
                elif col in ['humidity', 'wind_speed']:
                    temp_df[col] = 50.0 if col == 'humidity' else 5.0
                elif col in ['has_precipitation', 'weather_severity', 'rain_during_rush', 
                           'pickup_cluster', 'dropoff_cluster', 'source_cluster', 'destination_cluster']:
                    temp_df[col] = 0
                else:
                    temp_df[col] = 0
        
        return temp_df[X_cols]
    
    def get_location_mapping(self) -> Dict[str, Tuple[float, float]]:
        """Get the mapping of locations to coordinates."""
        if self.location_mapping is None:
            self._create_location_mapping()
        return self.location_mapping
    
    def get_historical_surge_patterns(self) -> pd.DataFrame:
        """Get historical surge patterns by hour, day, and weather."""
        if self.merged_df is None:
            return pd.DataFrame()
        
        # Group by hour and day of week
        hourly_surge = self.merged_df.groupby(['hour', 'day_of_week'])['surge_multiplier'].mean().reset_index()
        
        # Group by weather type
        weather_surge = self.merged_df.groupby('precipitation_type')['surge_multiplier'].mean().reset_index()
        
        # Group by hour and weather
        hour_weather_surge = self.merged_df.groupby(['hour', 'precipitation_type'])['surge_multiplier'].mean().reset_index()
        
        return {
            'hourly': hourly_surge,
            'weather': weather_surge,
            'hour_weather': hour_weather_surge
        } 