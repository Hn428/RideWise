import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, Tuple, List, Union
from geopy.distance import geodesic
import os


class FeatureEngineer:
    """
    Class for creating and transforming features from the merged ride data.
    """
    
    def __init__(self):
        """Initialize the feature engineer with necessary encoders and transformers."""
        self.categorical_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.location_mapping = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the merged dataframe.
        
        Args:
            df: Merged dataframe with ride and weather data
            
        Returns:
            DataFrame with engineered features
        """
        print("Starting feature engineering process...")
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Create time-based features
        data = self._create_time_features(data)
        
        # Create weather-based features
        data = self._create_weather_features(data)
        
        # Create location-based features
        data = self._create_location_features(data)
        
        # Create ride-based features
        data = self._create_ride_features(data)
        
        # Create interaction features
        data = self._create_interaction_features(data)
        
        print(f"Feature engineering complete. Created {len(data.columns) - len(df.columns)} new features.")
        return data
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp."""
        if 'timestamp' not in df.columns:
            print("Warning: No timestamp column found. Skipping time features.")
            return df
        
        print("Creating time-based features...")
        
        # Basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Special time indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Cyclical time features (to capture daily and weekly cycles)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and transform weather-related features."""
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation_type']
        if not any(col in df.columns for col in weather_cols):
            print("Warning: No weather columns found. Skipping weather features.")
            return df
        
        print("Creating weather-based features...")
        
        # Handle precipitation type
        if 'precipitation_type' in df.columns:
            # Create a binary indicator for precipitation
            df['has_precipitation'] = (~df['precipitation_type'].isin(['clear', 'none', np.nan])).astype(int)
            
            # Create a weather severity score
            weather_severity_map = {
                'clear': 0,
                'none': 0,
                'drizzle': 1,
                'rain': 2,
                'snow': 3,
                'sleet': 3,
                'hail': 4,
                'thunderstorm': 4
            }
            df['weather_severity'] = df['precipitation_type'].map(
                lambda x: weather_severity_map.get(x, 0) if pd.notna(x) else 0
            )
            
            # One-hot encode precipitation type
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            precip_types = df['precipitation_type'].fillna('missing').values.reshape(-1, 1)
            precip_encoded = encoder.fit_transform(precip_types)
            
            # Create DataFrame with encoded columns
            encoded_df = pd.DataFrame(
                precip_encoded,
                columns=[f'precip_{cat}' for cat in encoder.categories_[0][1:]],
                index=df.index
            )
            
            # Concatenate with original dataframe
            df = pd.concat([df, encoded_df], axis=1)
            
            # Store encoder for future use
            self.categorical_encoders['precipitation_type'] = encoder
        
        # Handle temperature
        if 'temperature' in df.columns:
            # Create temperature categories
            df['temp_category'] = pd.cut(
                df['temperature'],
                bins=[-float('inf'), 0, 10, 20, 30, float('inf')],
                labels=['freezing', 'cold', 'moderate', 'warm', 'hot']
            )
        
        # Handle humidity
        if 'humidity' in df.columns:
            # Create humidity categories
            df['is_humid'] = (df['humidity'] > 70).astype(int)
        
        # Handle wind speed
        if 'wind_speed' in df.columns:
            # Create wind categories
            df['is_windy'] = (df['wind_speed'] > 15).astype(int)
        
        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and transform location-related features."""
        location_cols = ['pickup_latitude', 'pickup_longitude', 'source', 'destination']
        if not any(col in df.columns for col in location_cols):
            print("Warning: No location columns found. Skipping location features.")
            return df
        
        print("Creating location-based features...")
        
        # Handle coordinate-based locations
        if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude']):
            # Create clusters for similar pickup locations
            if len(df) > 10:
                from sklearn.cluster import KMeans
                
                # Sample down for efficiency if needed
                sample_size = min(10000, len(df))
                coords = df[['pickup_latitude', 'pickup_longitude']].dropna().sample(sample_size) if len(df) > sample_size else df[['pickup_latitude', 'pickup_longitude']].dropna()
                
                # Determine appropriate number of clusters
                n_clusters = min(10, len(coords) // 100) if len(coords) >= 100 else 2
                
                if len(coords) >= n_clusters:
                    # Cluster pickup locations
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(coords)
                    
                    # Predict cluster for all points
                    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']].fillna(coords.mean()))
        
        # Handle named locations (source/destination)
        if all(col in df.columns for col in ['source', 'destination']):
            # Create unique IDs for each source and destination
            sources = sorted(df['source'].unique())
            destinations = sorted(df['destination'].unique())
            
            source_mapping = {source: idx for idx, source in enumerate(sources)}
            dest_mapping = {dest: idx for idx, dest in enumerate(destinations)}
            
            df['source_id'] = df['source'].map(source_mapping)
            df['destination_id'] = df['destination'].map(dest_mapping)
            
            # Store mappings for future use
            self.location_mapping = {
                'source': source_mapping,
                'destination': dest_mapping
            }
            
            # Create route_id for source-destination pairs
            df['route_id'] = df['source'] + '_to_' + df['destination']
            
            # Calculate route demand (number of rides per route)
            route_counts = df['route_id'].value_counts().to_dict()
            df['route_demand'] = df['route_id'].map(route_counts)
            
            # Normalize to 0-1 scale
            max_demand = df['route_demand'].max()
            if max_demand > 0:
                df['route_demand_normalized'] = df['route_demand'] / max_demand
        
        return df
    
    def _create_ride_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and transform ride-related features."""
        ride_cols = ['cab_type', 'product_id', 'distance', 'price']
        if not any(col in df.columns for col in ride_cols):
            print("Warning: Few ride-specific columns found. Skipping some ride features.")
            return df
        
        print("Creating ride-based features...")
        
        # Handle distance
        if 'distance' in df.columns:
            # Convert to kilometers if not already
            if 'distance_km' not in df.columns:
                df['distance_km'] = df['distance'] * 1.60934  # Convert miles to km
            
            # Create distance categories
            df['distance_category'] = pd.cut(
                df['distance_km'],
                bins=[0, 2, 5, 10, 20, float('inf')],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        # Handle ride type (cab_type)
        if 'cab_type' in df.columns:
            df['is_uber'] = (df['cab_type'] == 'Uber').astype(int)
            df['is_lyft'] = (df['cab_type'] == 'Lyft').astype(int)
            
            # One-hot encode cab_type
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            cab_types = df['cab_type'].fillna('unknown').values.reshape(-1, 1)
            cab_encoded = encoder.fit_transform(cab_types)
            
            # Create DataFrame with encoded columns
            encoded_df = pd.DataFrame(
                cab_encoded,
                columns=[f'cab_{cat}' for cat in encoder.categories_[0][1:]],
                index=df.index
            )
            
            # Concatenate with original dataframe
            df = pd.concat([df, encoded_df], axis=1)
            
            # Store encoder for future use
            self.categorical_encoders['cab_type'] = encoder
        
        # Handle product type
        if 'product_id' in df.columns or 'name' in df.columns:
            # Use either product_id or name
            product_col = 'product_id' if 'product_id' in df.columns else 'name'
            
            # One-hot encode product
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            products = df[product_col].fillna('unknown').values.reshape(-1, 1)
            product_encoded = encoder.fit_transform(products)
            
            # Create DataFrame with encoded columns
            encoded_df = pd.DataFrame(
                product_encoded,
                columns=[f'product_{cat}' for cat in encoder.categories_[0][1:]],
                index=df.index
            )
            
            # Concatenate with original dataframe
            df = pd.concat([df, encoded_df], axis=1)
            
            # Store encoder for future use
            self.categorical_encoders[product_col] = encoder
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture interactions between different factors."""
        print("Creating interaction features...")
        
        # Time and Weather interactions
        if all(col in df.columns for col in ['is_rush_hour', 'has_precipitation']):
            df['rain_during_rush'] = df['is_rush_hour'] * df['has_precipitation']
        
        if all(col in df.columns for col in ['is_weekend', 'temperature']):
            df['temp_weekend_interaction'] = df['is_weekend'] * df['temperature']
        
        # Time and Location interactions
        if 'is_rush_hour' in df.columns and any(col in df.columns for col in ['pickup_cluster', 'source_id']):
            location_col = 'pickup_cluster' if 'pickup_cluster' in df.columns else 'source_id'
            df['loc_rush_hour'] = df['is_rush_hour'].astype(str) + '_' + df[location_col].astype(str)
        
        # Historical demand features
        if all(col in df.columns for col in ['hour', 'day_of_week', 'surge_multiplier']):
            # Calculate average surge by hour and day
            hour_day_surge = df.groupby(['hour', 'day_of_week'])['surge_multiplier'].mean().reset_index()
            hour_day_surge.columns = ['hour', 'day_of_week', 'avg_historical_surge']
            
            # Merge back with original data
            df = pd.merge(df, hour_day_surge, on=['hour', 'day_of_week'], how='left')
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame = None) -> List[str]:
        """
        Return list of feature columns to use for training/prediction.
        
        Args:
            df: DataFrame to extract feature names from (optional)
            
        Returns:
            List of feature column names
        """
        # If a dataframe is provided, use it to identify feature columns
        if df is not None:
            # Basic time features
            time_features = [
                'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_night',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            time_features = [f for f in time_features if f in df.columns]
            
            # Weather features
            weather_features = [
                'temperature', 'humidity', 'wind_speed', 'has_precipitation',
                'weather_severity', 'is_humid', 'is_windy'
            ]
            weather_features = [f for f in weather_features if f in df.columns]
            
            # Location features
            location_features = [
                'pickup_cluster', 'source_id', 'destination_id', 'route_demand_normalized'
            ]
            location_features = [f for f in location_features if f in df.columns]
            
            # Ride features
            ride_features = ['distance_km', 'is_uber', 'is_lyft']
            ride_features = [f for f in ride_features if f in df.columns]
            
            # Interaction features
            interaction_features = [
                'rain_during_rush', 'temp_weekend_interaction', 'avg_historical_surge'
            ]
            interaction_features = [f for f in interaction_features if f in df.columns]
            
            # One-hot encoded categorical features
            categorical_features = [col for col in df.columns if col.startswith(('precip_', 'cab_', 'product_'))]
            
            # Combine all features
            feature_columns = (
                time_features + weather_features + location_features + 
                ride_features + interaction_features + categorical_features
            )
            
            self.feature_columns = feature_columns
        
        return self.feature_columns
    
    def prepare_features_for_model(self, df: pd.DataFrame, scale: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare the engineered features for model training/prediction.
        
        Args:
            df: DataFrame with engineered features
            scale: Whether to scale the numerical features
            
        Returns:
            Tuple containing:
                - NumPy array of prepared features
                - List of feature column names
        """
        # Get feature columns if not already set
        if not self.feature_columns:
            self.get_feature_columns(df)
        
        # Ensure all feature columns exist in the dataframe
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing feature columns: {missing_cols}")
            # Subset to only available columns
            available_features = [col for col in self.feature_columns if col in df.columns]
            X = df[available_features].copy()
        else:
            X = df[self.feature_columns].copy()
        
        # Fill any missing values
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode().iloc[0])
        
        # Scale numerical features if requested
        if scale:
            # Identify numerical columns
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if num_cols:
                # Fit scaler if not already fit
                X_num = X[num_cols].values
                
                if not hasattr(self.scaler, 'n_features_in_'):
                    self.scaler.fit(X_num)
                
                # Transform the data
                X_scaled = self.scaler.transform(X_num)
                
                # Replace original values with scaled values
                for i, col in enumerate(num_cols):
                    X[col] = X_scaled[:, i]
        
        return X.values, X.columns.tolist()


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    rides_path = os.path.join("data", "cab_rides.csv")
    weather_path = os.path.join("data", "weather.csv")
    
    # Preprocess data
    preprocessor = DataPreprocessor(rides_path, weather_path)
    preprocessor.load_data()
    preprocessor.clean_rides_data()
    preprocessor.clean_weather_data()
    merged_data = preprocessor.merge_datasets()
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(merged_data)
    
    # Get feature columns for model
    feature_cols = engineer.get_feature_columns(features_df)
    print(f"\nSelected {len(feature_cols)} features for modeling:")
    print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    # Prepare features for model
    X, feature_names = engineer.prepare_features_for_model(features_df)
    print(f"\nPrepared feature matrix with shape: {X.shape}")
    
    # Show sample of prepared features
    sample_df = pd.DataFrame(X[:5], columns=feature_names)
    print("\nSample of prepared features:")
    print(sample_df) 