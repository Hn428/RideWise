import pandas as pd
import numpy as np
import pickle
import base64
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from database import get_data_as_dataframe, store_model, get_model

def preprocess_features(df):
    """
    Preprocess the features for model training
    
    Args:
        df (DataFrame): Input data
        
    Returns:
        DataFrame: Processed feature data
    """
    if df.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Handle missing values
    df_features = df_features.fillna({
        'temperature': df_features['temperature'].median(),
        'congestion_level': df_features['congestion_level'].median(),
        'current_speed': df_features['current_speed'].median(),
        'event_count': 0,
        'uber_price': df_features['uber_price'].median(),
        'uber_surge': 1.0,
        'lyft_price': df_features['lyft_price'].median(),
        'lyft_surge': 1.0
    })
    
    # One-hot encode categorical features
    if 'weather_description' in df_features.columns:
        # Group similar weather conditions
        weather_mapping = {
            'clear': 'clear',
            'sunny': 'clear',
            'partly': 'partly_cloudy',
            'cloudy': 'cloudy',
            'overcast': 'cloudy',
            'rain': 'rain',
            'drizzle': 'rain',
            'shower': 'rain',
            'thunderstorm': 'storm',
            'storm': 'storm',
            'snow': 'snow',
            'sleet': 'snow',
            'fog': 'fog',
            'mist': 'fog',
            'haze': 'fog'
        }
        
        # Apply mapping to standardize weather descriptions
        df_features['weather_category'] = df_features['weather_description'].str.lower().apply(
            lambda x: next((v for k, v in weather_mapping.items() if k in str(x)), 'other')
        )
        
        # One-hot encode the weather category
        weather_dummies = pd.get_dummies(df_features['weather_category'], prefix='weather')
        df_features = pd.concat([df_features, weather_dummies], axis=1)
    
    # Create time-based features
    df_features['sin_hour'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['cos_hour'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['sin_day'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['cos_day'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # Drop columns not used for modeling
    columns_to_drop = ['timestamp', 'city', 'weather_description', 'weather_category',
                       'uber_price', 'lyft_price', 'uber_surge', 'lyft_surge']
    
    features = df_features.drop(columns=[col for col in columns_to_drop if col in df_features.columns])
    
    return features

def train_surge_model(df):
    """
    Train a model to predict surge probability
    
    Args:
        df (DataFrame): Training data
        
    Returns:
        model: Trained surge prediction model
    """
    if df.empty or len(df) < 10:  # Need minimum amount of data
        return None
    
    # Prepare features
    features = preprocess_features(df)
    
    # Define surge as a binary target (1 if surge >= 1.2, else 0)
    y_uber = (df['uber_surge'] >= 1.2).astype(int)
    y_lyft = (df['lyft_surge'] >= 1.2).astype(int)
    
    # Combine targets (1 if either service has surge)
    y = (y_uber | y_lyft).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Surge Model - Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    return model

def train_price_model(df, service='uber'):
    """
    Train a model to predict ride prices
    
    Args:
        df (DataFrame): Training data
        service (str): 'uber' or 'lyft'
        
    Returns:
        model: Trained price prediction model
    """
    if df.empty or len(df) < 10:  # Need minimum amount of data
        return None
    
    # Prepare features
    features = preprocess_features(df)
    
    # Target variable
    target_col = f"{service}_price"
    
    if target_col not in df.columns:
        return None
    
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    
    print(f"{service.capitalize()} Price Model - Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
    
    return model

def train_models(historical_data):
    """
    Train all models based on historical data
    
    Args:
        historical_data (list): Historical ride data
    """
    # Convert to DataFrame if it's not already
    if isinstance(historical_data, list):
        # Create a DataFrame from the raw data
        df = get_data_as_dataframe(city=None)
    else:
        df = historical_data
    
    if df.empty:
        print("Not enough data to train models")
        return False
    
    # Train surge probability model
    surge_model = train_surge_model(df)
    if surge_model:
        # Serialize and store model
        model_data = pickle.dumps(surge_model)
        store_model("surge_model", model_data)
    
    # Train price prediction models
    uber_model = train_price_model(df, 'uber')
    if uber_model:
        model_data = pickle.dumps(uber_model)
        store_model("uber_price_model", model_data)
    
    lyft_model = train_price_model(df, 'lyft')
    if lyft_model:
        model_data = pickle.dumps(lyft_model)
        store_model("lyft_price_model", model_data)
    
    return True

def predict_surge(city, hour=None, day_of_week=None, weather=None, events=0, traffic=None):
    """
    Predict surge probability for a given set of conditions
    
    Args:
        city (str): City name
        hour (int, optional): Hour of day (0-23)
        day_of_week (int, optional): Day of week (0=Monday, 6=Sunday)
        weather (str, optional): Weather description
        events (int, optional): Number of nearby events
        traffic (float, optional): Traffic congestion level
        
    Returns:
        float: Probability of surge pricing
    """
    # Get the surge model
    model_data = get_model("surge_model")
    if not model_data:
        return 0.5  # Default if no model is available
    
    model = pickle.loads(model_data)
    
    # Use current time if not specified
    now = datetime.now()
    if hour is None:
        hour = now.hour
    if day_of_week is None:
        day_of_week = now.weekday()
    
    # Create feature vector
    features = {
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [1 if day_of_week >= 5 else 0],
        'is_rush_hour': [1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0],
        'event_count': [events if events is not None else 0],
        'sin_hour': [np.sin(2 * np.pi * hour / 24)],
        'cos_hour': [np.cos(2 * np.pi * hour / 24)],
        'sin_day': [np.sin(2 * np.pi * day_of_week / 7)],
        'cos_day': [np.cos(2 * np.pi * day_of_week / 7)]
    }
    
    # Add weather features if provided
    if weather:
        weather_mapping = {
            'clear': 'clear',
            'sunny': 'clear',
            'partly': 'partly_cloudy',
            'cloudy': 'cloudy',
            'overcast': 'cloudy',
            'rain': 'rain',
            'drizzle': 'rain',
            'shower': 'rain',
            'thunderstorm': 'storm',
            'storm': 'storm',
            'snow': 'snow',
            'sleet': 'snow',
            'fog': 'fog',
            'mist': 'fog',
            'haze': 'fog'
        }
        
        weather_category = next((v for k, v in weather_mapping.items() if k in weather.lower()), 'other')
        
        # Add one-hot encoded weather features
        for category in ['clear', 'partly_cloudy', 'cloudy', 'rain', 'storm', 'snow', 'fog', 'other']:
            features[f'weather_{category}'] = [1 if category == weather_category else 0]
    
    # Add traffic features if provided
    if traffic is not None:
        features['congestion_level'] = [traffic]
        # Estimate speed based on congestion
        features['current_speed'] = [60 * (1 - traffic)]
    
    # Create DataFrame
    X = pd.DataFrame(features)
    
    # Make prediction
    surge_prob = model.predict_proba(X)[0][1]
    
    return surge_prob

def predict_price(service, city, hour=None, day_of_week=None, weather=None, events=0, traffic=None):
    """
    Predict price for a given service and conditions
    
    Args:
        service (str): 'uber' or 'lyft'
        city (str): City name
        hour (int, optional): Hour of day (0-23)
        day_of_week (int, optional): Day of week (0=Monday, 6=Sunday)
        weather (str, optional): Weather description
        events (int, optional): Number of nearby events
        traffic (float, optional): Traffic congestion level
        
    Returns:
        float: Predicted price
    """
    # Get the price model
    model_data = get_model(f"{service}_price_model")
    if not model_data:
        # Default baseline prices if no model is available
        base_prices = {"uber": 15.0, "lyft": 14.0}
        return base_prices.get(service, 15.0)
    
    model = pickle.loads(model_data)
    
    # Use current time if not specified
    now = datetime.now()
    if hour is None:
        hour = now.hour
    if day_of_week is None:
        day_of_week = now.weekday()
    
    # Create feature vector (same as for surge prediction)
    features = {
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [1 if day_of_week >= 5 else 0],
        'is_rush_hour': [1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0],
        'event_count': [events if events is not None else 0],
        'sin_hour': [np.sin(2 * np.pi * hour / 24)],
        'cos_hour': [np.cos(2 * np.pi * hour / 24)],
        'sin_day': [np.sin(2 * np.pi * day_of_week / 7)],
        'cos_day': [np.cos(2 * np.pi * day_of_week / 7)]
    }
    
    # Add weather features if provided
    if weather:
        weather_mapping = {
            'clear': 'clear',
            'sunny': 'clear',
            'partly': 'partly_cloudy',
            'cloudy': 'cloudy',
            'overcast': 'cloudy',
            'rain': 'rain',
            'drizzle': 'rain',
            'shower': 'rain',
            'thunderstorm': 'storm',
            'storm': 'storm',
            'snow': 'snow',
            'sleet': 'snow',
            'fog': 'fog',
            'mist': 'fog',
            'haze': 'fog'
        }
        
        weather_category = next((v for k, v in weather_mapping.items() if k in weather.lower()), 'other')
        
        # Add one-hot encoded weather features
        for category in ['clear', 'partly_cloudy', 'cloudy', 'rain', 'storm', 'snow', 'fog', 'other']:
            features[f'weather_{category}'] = [1 if category == weather_category else 0]
    
    # Add traffic features if provided
    if traffic is not None:
        features['congestion_level'] = [traffic]
        # Estimate speed based on congestion
        features['current_speed'] = [60 * (1 - traffic)]
    
    # Create DataFrame
    X = pd.DataFrame(features)
    
    # Make prediction
    price = model.predict(X)[0]
    
    return price

def predict_best_times(city, origin, destination):
    """
    Predict the best times to book a ride in the next 24 hours
    
    Args:
        city (str): City name
        origin (str): Starting location
        destination (str): Ending location
        
    Returns:
        dict: Best times and prices for each service
    """
    # Get the latest weather and events data
    # In a real implementation, this would fetch the forecast for the next 24 hours
    
    # For now, use simple approximations
    
    # Get current time
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()
    
    # Generate hourly predictions for the next 24 hours
    hourly_predictions = []
    
    for hour_offset in range(24):
        # Calculate future hour and day
        future_datetime = now + timedelta(hours=hour_offset)
        future_hour = future_datetime.hour
        future_day = future_datetime.weekday()
        
        # Predict prices for each service
        # For simplicity, we'll use the same weather and traffic conditions
        # In a real app, we would use forecasts
        
        # Estimate traffic based on time patterns
        is_rush_hour = (7 <= future_hour <= 9) or (16 <= future_hour <= 19)
        is_weekend = future_day >= 5
        
        if is_rush_hour and not is_weekend:
            traffic = 0.7
        elif is_weekend and future_hour >= 10 and future_hour <= 18:
            traffic = 0.5
        elif future_hour >= 22 or future_hour <= 5:
            traffic = 0.2
        else:
            traffic = 0.4
            
        # Predict prices
        uber_price = predict_price('uber', city, future_hour, future_day, None, 0, traffic)
        lyft_price = predict_price('lyft', city, future_hour, future_day, None, 0, traffic)
        
        hourly_predictions.append({
            'hour': future_hour,
            'day': future_day,
            'datetime': future_datetime,
            'uber_price': uber_price,
            'lyft_price': lyft_price
        })
    
    # Find the best time for each service
    hourly_df = pd.DataFrame(hourly_predictions)
    
    # Best time for Uber
    uber_best_idx = hourly_df['uber_price'].idxmin()
    uber_best_time = hourly_df.iloc[uber_best_idx]['datetime']
    uber_lowest_price = hourly_df.iloc[uber_best_idx]['uber_price']
    
    # Best time for Lyft
    lyft_best_idx = hourly_df['lyft_price'].idxmin()
    lyft_best_time = hourly_df.iloc[lyft_best_idx]['datetime']
    lyft_lowest_price = hourly_df.iloc[lyft_best_idx]['lyft_price']
    
    # Format results
    result = {
        'uber': {
            'best_time': uber_best_time,
            'lowest_price': uber_lowest_price
        },
        'lyft': {
            'best_time': lyft_best_time,
            'lowest_price': lyft_lowest_price
        },
        'hourly_predictions': hourly_df.to_dict('records')
    }
    
    return result
