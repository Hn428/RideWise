import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class FeatureEngineer:
    def __init__(self, save_dir='models'):
        self.save_dir = save_dir
        self.transformer = None
        self.features = None
        os.makedirs(save_dir, exist_ok=True)
    
    def extract_features(self, df):
        """
        Extract and engineer features from the preprocessed dataframe
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Extract hour of day and day of week
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        
        # Create rush hour feature (typical rush hours: 7-9 AM and 4-7 PM)
        data['is_rush_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9)) | \
                               ((data['hour'] >= 16) & (data['hour'] <= 19))
        
        # Create weekend feature
        data['is_weekend'] = data['day_of_week'] >= 5
        
        # Calculate distance in km using Haversine formula
        data['distance'] = self.calculate_distance(
            data['source_lat'], data['source_lon'],
            data['destination_lat'], data['destination_lon']
        )
        
        # Weather impact features
        data['is_poor_weather'] = (data['rain'] >= 0.1) | (data['snow'] >= 0.1)
        
        # Base features for model
        self.features = [
            'distance', 'hour', 'day_of_week', 'is_rush_hour', 'is_weekend',
            'temperature', 'rain', 'snow', 'is_poor_weather', 'cab_type'
        ]
        
        return data
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate Haversine distance between two sets of coordinates
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def prepare_features(self, data):
        """
        Prepare features for model training/prediction by encoding and scaling
        """
        # Select relevant columns
        X = data[self.features].copy()
        
        if self.transformer is None:
            # Define preprocessing for numerical features
            numeric_features = [
                'distance', 'hour', 'day_of_week', 'temperature', 'rain', 'snow'
            ]
            numeric_transformer = StandardScaler()
            
            # Define preprocessing for categorical features
            categorical_features = ['cab_type']
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            # Boolean features
            boolean_features = ['is_rush_hour', 'is_weekend', 'is_poor_weather']
            
            # Create preprocessor
            self.transformer = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features),
                    # Boolean features are passed through
                    ('bool', 'passthrough', boolean_features)
                ]
            )
            
            # Fit the transformer
            self.transformer.fit(X)
            
            # Save the transformer
            joblib.dump(self.transformer, os.path.join(self.save_dir, 'feature_transformer.joblib'))
        
        # Transform the features
        X_processed = self.transformer.transform(X)
        
        return X_processed
    
    def load_transformer(self):
        """
        Load a previously saved transformer
        """
        transformer_path = os.path.join(self.save_dir, 'feature_transformer.joblib')
        if os.path.exists(transformer_path):
            self.transformer = joblib.load(transformer_path)
            return True
        return False 