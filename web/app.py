import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import urllib3
import ssl
import certifi
import sys

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("weather_api")

# Fix for SSL certificate verification issues - FOR TESTING ONLY
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

# Direct Nominatim geocoding function that bypasses SSL verification
def geocode_address(address):
    """Directly geocode an address using Nominatim API with SSL verification disabled"""
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
    try:
        response = requests.get(url, verify=False, headers={'User-Agent': 'surge_price_predictor'})
        data = response.json()
        if data and len(data) > 0:
            return {
                'lat': float(data[0]['lat']),
                'lon': float(data[0]['lon']),
                'display_name': data[0]['display_name']
            }
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
    return None

def reverse_geocode(lat, lon):
    """Directly reverse geocode coordinates using Nominatim API with SSL verification disabled"""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    try:
        response = requests.get(url, verify=False, headers={'User-Agent': 'surge_price_predictor'})
        data = response.json()
        if 'display_name' in data:
            return data['display_name']
    except Exception as e:
        st.error(f"Reverse geocoding error: {str(e)}")
    return None

def get_weather_emoji(weather_type):
    """Get appropriate emoji for weather type"""
    weather_emojis = {
        'clear': "☀️",
        'rain': "🌧️",
        'snow': "❄️",
        'sleet': "🌨️",
        'hail': "🌨️",
        'thunderstorm': "⛈️",
        'fog': "🌫️",
        'clouds': "☁️",
        'mist': "🌫️"
    }
    
    if weather_type.lower() in weather_emojis:
        return weather_emojis[weather_type.lower()]
    else:
        return "🌡️"  # Default emoji

def get_weather_background_css(hour, weather_type):
    """
    Generate CSS for Uber-like styling
    
    Args:
        hour: Current hour (0-23)
        weather_type: Type of weather (clear, rain, etc.)
        
    Returns:
        CSS string for styling
    """
    # Use Uber's color scheme regardless of time/weather
    bg_colors = ["#000000", "#121212"]  # Black to very dark gray
    accent_color = "#27B666"  # Uber green
    
    # Create CSS for styling
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {bg_colors[0]}, {bg_colors[1]});
    }}
    .css-6qob1r {{
        background-color: rgba(34, 34, 34, 0.8) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(39, 182, 102, 0.2);
    }}
    .stDateInput, .stTimeInput, .stNumberInput, .stSelectbox, .stTextInput {{
        background-color: rgba(34, 34, 34, 0.8);
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid rgba(39, 182, 102, 0.3);
    }}
    /* Additional Uber-like styling */
    .stButton > button {{
        background-color: {accent_color};
        color: white;
        border: none;
        font-weight: 500;
    }}
    /* Make success messages green like Uber */
    .element-container .stAlert.st-ae.st-af {{
        border-left-color: {accent_color} !important;
    }}
    /* Improve tabs appearance */
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(34, 34, 34, 0.5);
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {accent_color};
    }}
    </style>
    """
    return css

# Function to fetch weather data
def fetch_weather_data(lat, lon):
    """
    Fetch current weather data for a specific location using OpenWeatherMap API.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        
    Returns:
        Dict containing weather data or None if API call fails
    """
    try:
        # API endpoint for current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={WEATHER_API_KEY}"
        
        # Make the API request
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant weather information
            weather_info = {
                'temperature': data['main']['temp'],          # in Celsius
                'humidity': data['main']['humidity'],         # in %
                'wind_speed': data['wind']['speed'],          # in m/s, convert to km/h
                'precipitation_type': get_precipitation_type(data),
                'pressure': data['main']['pressure'],         # in hPa
                'feels_like': data['main']['feels_like'],     # in Celsius
                'visibility': data.get('visibility', 10000),  # in meters
                'weather_description': data['weather'][0]['description']
            }
            
            # Convert wind speed from m/s to km/h
            weather_info['wind_speed'] = weather_info['wind_speed'] * 3.6
            
            return weather_info
        else:
            st.error(f"Error fetching weather data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception while fetching weather data: {str(e)}")
        return None

def get_precipitation_type(weather_data):
    """
    Determine the precipitation type from OpenWeatherMap data.
    
    Args:
        weather_data: Raw weather data from OpenWeatherMap API
        
    Returns:
        String representing the precipitation type
    """
    if 'weather' not in weather_data or not weather_data['weather']:
        return 'clear'
    
    # Get the main weather condition and id
    weather_id = weather_data['weather'][0]['id']
    main_condition = weather_data['weather'][0]['main'].lower()
    
    # Map OpenWeatherMap conditions to our model's precipitation types
    if weather_id >= 200 and weather_id < 300:  # Thunderstorm
        return 'thunderstorm'
    elif weather_id >= 300 and weather_id < 400:  # Drizzle
        return 'rain'
    elif weather_id >= 500 and weather_id < 600:  # Rain
        return 'rain'
    elif weather_id >= 600 and weather_id < 700:  # Snow
        return 'snow' 
    elif weather_id == 611 or weather_id == 612 or weather_id == 613:  # Sleet
        return 'sleet'
    elif weather_id == 771 or weather_id == 781:  # Extreme weather
        return 'thunderstorm'
    elif 'clear' in main_condition:
        return 'clear'
    else:
        return 'clear'  # Default to clear for fog, mist, etc.

# Import custom modules
from src.data_processor import DataProcessor
from src.model import SurgePredictor

# Import utility functions from src.utils or define them here
def calculate_base_price(distance: float, product_id: str, is_rush_hour: bool = False) -> float:
    """
    Calculate base price for a ride, considering service level.
    
    Args:
        distance: Distance in kilometers
        product_id: Service level (e.g., 'UberX', 'Black', 'Lux')
        is_rush_hour: Whether the ride is during rush hour
        
    Returns:
        Base price for the ride
    """
    # Base rates and multipliers per service level (example values)
    service_level_modifiers = {
        # Uber
        "UberX":      {'base': 2.50, 'per_km': 1.75},
        "UberXL":     {'base': 3.50, 'per_km': 2.10},
        "UberPOOL":   {'base': 2.00, 'per_km': 1.50}, # Hypothetical
        "Black":      {'base': 7.00, 'per_km': 3.25},
        "SUV":        {'base': 14.00, 'per_km': 4.00}, # Assuming Black XL is SUV
        # Lyft
        "Lyft":       {'base': 2.50, 'per_km': 1.70},
        "Lyft XL":    {'base': 3.50, 'per_km': 2.05},
        "Lux":        {'base': 6.50, 'per_km': 3.10},
        "Lux Black":  {'base': 12.00, 'per_km': 3.80},
        "Lux Black XL":{'base': 15.00, 'per_km': 4.20},
        # Add defaults for any missing/unrecognized IDs
        "Shared":     {'base': 2.00, 'per_km': 1.50},
        "Default":    {'base': 2.50, 'per_km': 1.75}
    }
    
    # Get modifiers for the selected product_id, fallback to default
    modifiers = service_level_modifiers.get(product_id, service_level_modifiers["Default"])
    base_rate = modifiers['base']
    per_km_rate = modifiers['per_km']
    
    # Rush hour multiplier
    rush_hour_multiplier = 1.2 if is_rush_hour else 1.0
    
    # Calculate final base price
    calculated_price = (base_rate + (distance * per_km_rate)) * rush_hour_multiplier
    
    # Ensure minimum price (e.g., $5)
    min_price = 5.0
    return max(calculated_price, min_price)

def format_prediction_output(predicted_surge: float, base_price: float) -> dict:
    """
    Format prediction output with calculated prices.
    
    Args:
        predicted_surge: Predicted surge multiplier
        base_price: Base price for the ride
        
    Returns:
        Dictionary containing formatted prediction results
    """
    # Ensure surge multiplier is at least 1.0
    predicted_surge = max(1.0, predicted_surge)
    
    return {
        'surge_multiplier': round(predicted_surge, 2),
        'base_price': round(base_price, 2),
        'total_price': round(base_price * predicted_surge, 2)
    }

def check_data_files() -> dict:
    """
    Check if required data files exist.
    
    Returns:
        Dictionary indicating which files exist
    """
    data_dir = os.path.join(os.getcwd(), 'data')
    rides_path = os.path.join(data_dir, 'cab_rides.csv')
    weather_path = os.path.join(data_dir, 'weather.csv')
    
    return {
        'rides_file': os.path.exists(rides_path),
        'weather_file': os.path.exists(weather_path),
        'data_dir': os.path.exists(data_dir)
    }

def check_model_file() -> bool:
    """
    Check if a trained model file exists.
    
    Returns:
        Boolean indicating whether model file exists
    """
    model_path = os.path.join(os.getcwd(), 'models', 'surge_model.joblib')
    return os.path.exists(model_path)

# --- New Helper Function --- 
def find_cheapest_time_today(model, input_features, original_timestamp, distance, product_id):
    """
    Predicts surge price for remaining hours of the day to find the cheapest time.

    Args:
        model: The loaded SurgePredictor model.
        input_features: Dictionary of features for the original prediction (used as a base).
        original_timestamp: The timestamp selected by the user.
        distance: Ride distance.
        product_id: Selected service level.

    Returns:
        Tuple: (best_hour, best_price) or (None, None) if prediction fails.
    """
    best_time = None
    best_price = float('inf')
    predictions = []

    # Start from the selected time and check the next 24 hours
    start_time = original_timestamp.replace(minute=original_timestamp.minute // 30 * 30, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=24)
    
    # Check every 30 minutes
    time_slot = start_time
    time_increment = timedelta(minutes=30)
    
    # Create a list to store all predictions for visualization
    all_predictions = []
    
    while time_slot < end_time:
        try:
            # --- Prepare features for this time slot ---
            temp_features = input_features.copy()  # Start with original features
            hour = time_slot.hour
            minute = time_slot.minute
            
            # Update time-related features
            temp_features['hour'] = hour
            temp_features['day_of_week'] = time_slot.weekday()
            temp_features['month'] = time_slot.month
            temp_features['is_weekend'] = 1 if time_slot.weekday() >= 5 else 0
            temp_features['is_rush_hour'] = 1 if (((hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18)) and time_slot.weekday() < 5) else 0
            temp_features['is_night'] = 1 if (hour >= 22 or hour <= 4) else 0
            
            # More precise time features using actual minutes
            hour_fraction = hour + (minute / 60.0)
            temp_features['hour_sin'] = np.sin(2 * np.pi * hour_fraction / 24)
            temp_features['hour_cos'] = np.cos(2 * np.pi * hour_fraction / 24)
            temp_features['day_sin'] = np.sin(2 * np.pi * time_slot.weekday() / 7)
            temp_features['day_cos'] = np.cos(2 * np.pi * time_slot.weekday() / 7)
            
            # Note: Using original weather data for all predictions
            # We assume 'temperature', 'humidity', 'wind_speed', 'precipitation_type', 
            # 'has_precipitation' are already in temp_features from the input_features copy
            
            # Create DataFrame
            temp_input_df = pd.DataFrame([temp_features])
            
            temp_input_df_final = None # Initialize
            # --- Align features and handle categoricals (same logic as main prediction) ---
            if hasattr(model, 'feature_names') and model.feature_names:
                # --- Cab Type Encoding --- 
                if 'is_uber' in model.feature_names:
                    temp_input_df['is_uber'] = 1 if input_features.get('cab_type') == 'Uber' else 0 # Use original cab_type
                if 'is_lyft' in model.feature_names:
                    temp_input_df['is_lyft'] = 1 if input_features.get('cab_type') == 'Lyft' else 0
                if 'is_uber' in temp_input_df.columns or 'is_lyft' in temp_input_df.columns:
                    if 'cab_type' in temp_input_df.columns: temp_input_df = temp_input_df.drop(columns=['cab_type'])

                # --- Product ID Encoding --- 
                expected_product_cols = [f for f in model.feature_names if f.startswith('product_id_')]
                for col in expected_product_cols:
                    expected_product_name = col.replace('product_id_', '')
                    temp_input_df[col] = 1 if product_id == expected_product_name else 0
                if expected_product_cols and 'product_id' in temp_input_df.columns: temp_input_df = temp_input_df.drop(columns=['product_id'])

                # --- Precipitation Type Encoding --- 
                expected_precip_cols = [f for f in model.feature_names if f.startswith('precipitation_type_')]
                for col in expected_precip_cols:
                    expected_precip_name = col.replace('precipitation_type_', '')
                    temp_input_df[col] = 1 if input_features.get('precipitation_type') == expected_precip_name else 0 # Use original precip_type
                if expected_precip_cols and 'precipitation_type' in temp_input_df.columns: temp_input_df = temp_input_df.drop(columns=['precipitation_type'])

                # --- Alignment --- 
                aligned_df = pd.DataFrame(columns=model.feature_names)
                for col in model.feature_names:
                    if col in temp_input_df.columns: aligned_df[col] = temp_input_df[col]
                    else: aligned_df[col] = 0
                temp_input_df_final = aligned_df.fillna(0)
            
            else: # Case where model.feature_names is not available
                cols_to_drop = [c for c in ['cab_type', 'product_id', 'precipitation_type'] if c in temp_input_df.columns]
                temp_input_df_final = temp_input_df.drop(columns=cols_to_drop)
            # --- End Feature Alignment/Handling --- 
            
            # Ensure we have a DataFrame to predict on
            if temp_input_df_final is None:
                time_slot += time_increment
                continue # Skip this time slot if feature prep failed

            # --- Predict Surge --- 
            predicted_surge = 1.0 # Default surge
            if hasattr(model, 'predict') and callable(model.predict):
                 temp_prediction_result = model.predict(temp_input_df_final)
                 if isinstance(temp_prediction_result, (np.ndarray, list)) and len(temp_prediction_result) > 0:
                     predicted_surge = temp_prediction_result[0]
                 elif isinstance(temp_prediction_result, (float, np.floating)):
                     predicted_surge = temp_prediction_result
                 else:
                     time_slot += time_increment
                     continue # Skip this time slot if prediction failed
            else:
                time_slot += time_increment
                continue # Skip if model has no predict method

            # --- Calculate Price for this time slot ---
            temp_is_rush_hour = temp_features['is_rush_hour'] == 1
            temp_base_price = calculate_base_price(distance, product_id, temp_is_rush_hour)
            temp_total_price = temp_base_price * max(1.0, predicted_surge) # Ensure surge >= 1
            
            # Store prediction in the format (timestamp, price, surge)
            all_predictions.append((time_slot, temp_total_price, predicted_surge))

            # Update best time if this is cheaper
            if temp_total_price < best_price:
                best_price = temp_total_price
                best_time = time_slot
                
        except Exception as e:
            # Log errors for debugging
            # print(f"Error processing time {time_slot}: {e}")
            pass
            
        # Move to next time slot
        time_slot += time_increment
            
    if best_time is not None:
        return best_time, best_price, all_predictions
    else:
        return None, None, []
# --- End New Helper Function --- 

# Set page config
st.set_page_config(
    page_title="Surge Price Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .st-emotion-cache-16txtl3 h1 {
        text-align: center;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #27B666;
        color: white;
    }
    .prediction-box {
        background-color: rgba(34, 34, 34, 0.8);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .st-bq {
        border-left-color: rgba(39, 182, 102, 0.5) !important;
    }
    .stButton > button {
        background-color: #27B666;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #1E9E57;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background-color: #27B666;
    }
    .stButton > button[data-baseweb="button"][kind="primary"]:hover {
        background-color: #1E9E57;
    }
    @media (max-width: 768px) {
        .mobile-friendly {
            font-size: 0.9rem;
        }
    }
    .location-card {
        background-color: rgba(34, 34, 34, 0.8);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .location-title {
        text-align: center;
        padding-bottom: 10px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: bold;
        font-size: 1.2rem;
    }
    .location-box {
        background-color: rgba(34, 34, 34, 0.8);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .compact-location-selector {
        padding: 5px;
        margin-bottom: 0;
    }
    .map-container {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .weather-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 20px;
    }
    .weather-card {
        background-color: rgba(34, 34, 34, 0.8);
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .input-section {
        margin-bottom: 15px;
    }
    .ride-options {
        background-color: rgba(34, 34, 34, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-top: 0;
        margin-bottom: 0;
    }
    .time-weather-section {
        background-color: rgba(34, 34, 34, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-top: 0;
        margin-bottom: 0;
        border: 1px solid rgba(39, 182, 102, 0.3);
    }
    .prediction-button {
        margin-top: 10px;
        margin-bottom: 0;
        text-align: center;
    }
    .prediction-results {
        background-color: rgba(34, 34, 34, 0.8);
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid rgba(39, 182, 102, 0.5);
    }
</style>
""", unsafe_allow_html=True)


def get_available_models():
    """
    Get a list of available model files in the models directory
    
    Returns:
        List of model filenames
    """
    model_files = []
    models_dir = os.path.join(os.getcwd(), 'models')
    
    # Check if directory exists
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        # List all files with .joblib extension
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    return model_files


@st.cache_resource
def load_model(model_filename='surge_model.joblib'):
    """Load the trained model or return None if not found
    
    Args:
        model_filename: Name of the model file to load
        
    Returns:
        Loaded model or None if not found
    """
    # Clear any existing CSS conflicts or dividers
    # Remove extra dividers
    hide_divider_css = """
    <style>
    /* Remove the empty divider between model selection and location box */
    .css-18ni7ap.e8zbici2, .css-9uwa36.e8zbici2, .e8zbici2, .e1g8pov61 {
        display: none !important;
    }
    /* Hide any default dividers */
    .streamlit-expanderHeader, .streamlit-expanderContent {
        margin-bottom: 0 !important;
    }
    /* Hide other potential dividers */
    .css-12y0wxx.e1g8pov61, .e1g8pov61 {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        height: 0px !important;
    }
    /* Remove additional margins that may cause empty space */
    .css-1544g2n.eczjsme4, .eczjsme4, .e1tzin5v0, .css-1r6slb0.e1tzin5v2 {
        margin-top: 0px !important;
        padding-top: 0px !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    /* Make sure sections connect */
    .location-box {
        margin-top: 0 !important;
    }
    /* Remove any vertical spacing between elements */
    .stExpander, .css-1kyxreq.etr89bj2 {
        margin-bottom: 0 !important;
    }
    </style>
    """
    st.markdown(hide_divider_css, unsafe_allow_html=True)
    
    model_path = os.path.join('models', model_filename)
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            
            # Create a SurgePredictor instance if needed
            if not isinstance(model_data, SurgePredictor):
                predictor = SurgePredictor()
                
                # If it's a dictionary containing model components
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        predictor.model = model_data['model']
                        if 'scaler' in model_data:
                            predictor.scaler = model_data['scaler']
                        if 'feature_names' in model_data:
                            predictor.feature_names = model_data['feature_names']
                        return predictor
                    else:
                        return model_data  # Return as is
                else:
                    # Assume it's just the model itself
                    predictor.model = model_data
                    return predictor
            else:
                # It's already a SurgePredictor
                return model_data
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    # No model found or error loading
    return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and preprocess data for visualization"""
    # Check if data files exist
    file_status = check_data_files()
    
    if not file_status['rides_file'] or not file_status['weather_file']:
        return None, None, False
    
    # Load data using the processor
    processor = DataProcessor(
        rides_path='data/cab_rides.csv',
        weather_path='data/weather.csv'
    )
    
    # Load and clean data
    rides_df, weather_df = processor.load_data()
    
    # Merge the datasets
    merged_df = processor.merge_datasets(time_window='1h')
    
    # Engineer features (either using the processor's engineer_features method
    # or manually)
    try:
        # Try using the processor's engineer_features method
        processed_df = processor.engineer_features()
        if processed_df is not None:
            merged_df = processed_df
    except Exception as e:
        st.warning(f"Could not engineer features using processor: {str(e)}")
        # If that fails, extract basic features manually
        if 'timestamp' in merged_df.columns:
            merged_df['hour'] = merged_df['timestamp'].dt.hour
            merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    
    return merged_df, processor, True


def create_surge_trend_plot(df, time_group='hour'):
    """Create a plot showing surge trends over time."""
    if time_group == 'hour':
        # Group by hour - use string to avoid KeyError
        group_data = df.groupby('hour')['surge_multiplier'].mean().reset_index()
        x_label = 'Hour of Day'
        x_col = 'hour'
    else:
        # Group by day of week
        group_data = df.groupby('day_of_week')['surge_multiplier'].mean().reset_index()
        day_mapping = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        group_data['day_name'] = group_data['day_of_week'].map(day_mapping)
        x_label = 'Day of Week'
        x_col = 'day_name'
    
    # Create the plot
    fig = px.line(
        group_data, 
        x=x_col, 
        y='surge_multiplier',
        title=f'Average Surge Multiplier by {x_label}',
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title='Average Surge Multiplier'
    )
    
    return fig

def create_weather_impact_plot(df):
    """Create a plot showing weather impact on surge pricing."""
    if 'precipitation_type' not in df.columns:
        return None
    
    # Group by precipitation type - use string to avoid KeyError
    weather_impact = df.groupby('precipitation_type')['surge_multiplier'].mean().reset_index()
    
    # Create the plot
    fig = px.bar(
        weather_impact,
        x='precipitation_type',
        y='surge_multiplier',
        title='Impact of Weather Conditions on Surge Pricing',
        color='surge_multiplier'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Weather Condition',
        yaxis_title='Average Surge Multiplier'
    )
    
    return fig

def create_price_history_plot(df, time_window=24):
    """Create a plot showing price history."""
    if 'timestamp' not in df.columns:
        return None
    
    # Create a proper copy to avoid SettingWithCopyWarning
    df_window = df.copy()
    
    # Filter to the time window
    if time_window:
        latest_time = df_window['timestamp'].max()
        cutoff_time = latest_time - pd.Timedelta(hours=time_window)
        df_window = df_window[df_window['timestamp'] >= cutoff_time].copy()
    
    # Group by hour - use 'h' instead of 'H' to avoid deprecation warning
    df_window.loc[:, 'hour_bin'] = df_window['timestamp'].dt.floor('h')
    
    # Check if cab_type exists
    if 'cab_type' in df_window.columns:
        hourly_data = df_window.groupby(['hour_bin', 'cab_type']).agg({
            'surge_multiplier': 'mean'
        }).reset_index()
        
        # Create the plot
        fig = px.line(
            hourly_data, 
            x='hour_bin', 
            y='surge_multiplier', 
            color='cab_type',
            title=f'Surge Multiplier History (Past {time_window} Hours)'
        )
    else:
        hourly_data = df_window.groupby('hour_bin').agg({
            'surge_multiplier': 'mean'
        }).reset_index()
        
        # Create the plot
        fig = px.line(
            hourly_data, 
            x='hour_bin', 
            y='surge_multiplier',
            title=f'Surge Multiplier History (Past {time_window} Hours)'
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Average Surge Multiplier'
    )
    
    return fig

def display_prediction_tab():
    """Display the prediction tab content"""
    st.header("Predict Surge Pricing", anchor=False)
    
    # Get available models
    available_models = get_available_models()
    
    if len(available_models) == 0:
        st.warning("⚠️ No trained models found. Please go to the Training tab to train a model first.")
        return
    
    # Model selection
    with st.expander("Model Selection", expanded=False):
        st.info("Select a model to use for prediction. If you've trained multiple models, you can choose which one to use.")
        
        # Initialize model selection in session state if needed
        if 'selected_model' not in st.session_state or st.session_state.selected_model not in available_models:
            st.session_state.selected_model = available_models[0]
        
        # Model dropdown
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            index=available_models.index(st.session_state.selected_model),
            key="model_selector"
        )
        
        # Update session state if changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            # Force Streamlit to clear the model cache for the new selection
            st.cache_resource.clear()
        
        st.success(f"Using model: {selected_model}")
        
        # Show model details if available
        model_details_btn = st.button("Show Model Details")
        if model_details_btn:
            try:
                model_path = os.path.join('models', selected_model)
                model_obj = joblib.load(model_path)
                
                st.subheader("Model Details")
                
                # Check if model has metadata
                if hasattr(model_obj, 'metadata') and model_obj.metadata:
                    metadata = model_obj.metadata
                    
                    # Show basic info
                    st.markdown(f"**Model Type**: {metadata.get('model_type', 'Unknown')}")
                    st.markdown(f"**Created**: {metadata.get('timestamp', 'Unknown')}")
                    st.markdown(f"**Features Used**: {metadata.get('feature_count', 'Unknown')}")
                    
                    # Show metrics if available
                    if 'metrics' in metadata:
                        metrics = metadata['metrics']
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("MSE", f"{metrics.get('mse', 'N/A'):.4f}")
                        with metrics_cols[1]:
                            st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.4f}")
                        with metrics_cols[2]:
                            st.metric("MAE", f"{metrics.get('mae', 'N/A'):.4f}")
                        with metrics_cols[3]:
                            st.metric("R²", f"{metrics.get('r2', 'N/A'):.4f}")
                    
                    # Show feature list in an expander
                    if 'features' in metadata:
                        with st.expander("Features Used"):
                            st.write(metadata['features'])
                    
                    # Feature importance if available
                    if hasattr(model_obj, 'get_feature_importance'):
                        with st.expander("Feature Importance"):
                            try:
                                feature_importance = model_obj.get_feature_importance()
                                st.dataframe(feature_importance.head(10))
                            except:
                                st.info("Feature importance not available.")
                else:
                    # Basic model info if no metadata
                    st.markdown(f"**Model File**: {selected_model}")
                    st.markdown(f"**Model Type**: {type(model_obj).__name__}")
                    st.info("This model doesn't have detailed metadata available.")
                    
            except Exception as e:
                st.error(f"Error loading model details: {str(e)}")
    
    # Load the selected model
    model = load_model(st.session_state.selected_model)
    
    if model is None:
        st.warning(f"⚠️ Error loading selected model. Please try another model or train a new one.")
        return
    
    # Get current hour for dynamic styling
    current_hour = datetime.now().hour
    
    # Apply dynamic styling after loading weather data
    weather_type = 'clear'  # Default
    
    # --- Helper Function for Compact Location Selection ---
    def compact_location_selection(type_prefix, default_lat, default_long, default_name):
        # Predefined locations and their coordinates
        location_coords = {
            "Current Location": None,  # Special value for current location
            "Times Square, New York": (40.7580, -73.9855),
            "Empire State Building, New York": (40.7484, -73.9857),
            "Central Park, New York": (40.7812, -73.9665),
            "Wall Street, New York": (40.7068, -74.0090),
            "Brooklyn Bridge, New York": (40.7061, -73.9969),
            "Grand Central Station, New York": (40.7527, -73.9772),
            "Statue of Liberty, New York": (40.6892, -74.0445),
            "Rockefeller Center, New York": (40.7587, -73.9787),
            "New York City Center (Default)": (40.7128, -74.0060)
        }
        
        location_list = list(location_coords.keys())
        default_index = location_list.index(default_name) if default_name in location_list else 0
        
        title_emoji = "🚩" if type_prefix == 'source' else "🏁"
        
        # Create three radio options but only show the selected one's content
        selection_method = st.radio(
            f"{title_emoji} {type_prefix.capitalize()} Location:",
            ("Select from List", "Search Address", "Enter Coordinates"),
            key=f"{type_prefix}_method",
            horizontal=True
        )
        
        lat, lon = default_lat, default_long
        location_name = default_name
        
        # Show appropriate input based on selection
        if selection_method == "Select from List":
            selected_location = st.selectbox(
                f"Select {type_prefix}:",
                location_list,
                index=0,  # Default to "Current Location"
                key=f"{type_prefix}_select"
            )
            
            # Handle current location specially
            if selected_location == "Current Location":
                with st.spinner("Fetching current location..."):
                    try:
                        # Get IP-based location (simplified version)
                        # In a real app, you might use browser geolocation API or more accurate methods
                        ip_location = requests.get('https://ipinfo.io/json', verify=False).json()
                        location_str = f"{ip_location.get('city', '')}, {ip_location.get('region', '')}, {ip_location.get('country', '')}"
                        
                        # Use our direct geocoding function
                        location_result = geocode_address(location_str)
                        
                        if location_result:
                            lat = location_result['lat']
                            lon = location_result['lon']
                            location_display = location_result['display_name'].split(',')[0]
                            location_name = f"Current Location ({location_display})"
                            st.success(f"📍 Located at: {location_display}")
                        else:
                            # Fallback to a default location if geolocation fails
                            st.warning("Could not determine current location. Using New York City as default.")
                            lat, lon = 40.7128, -74.0060  # NYC default
                            location_name = "NYC (Current Location Unavailable)"
                    except Exception as e:
                        st.error(f"Error getting location: {str(e)}")
                        # Fallback to default
                        lat, lon = 40.7128, -74.0060
                        location_name = "NYC (Current Location Unavailable)"
            else:
                # For predefined locations
                lat, lon = location_coords[selected_location]
                location_name = selected_location
                
            st.session_state[f"{type_prefix}_location_name"] = location_name
            
        elif selection_method == "Search Address":
            col1, col2 = st.columns([3, 1])
            with col1:
                address = st.text_input(
                    f"Enter {type_prefix} address",
                    key=f"{type_prefix}_address_input"
                )
            with col2:
                search_button = st.button(f"🔍 Find", key=f"{type_prefix}_search_button")
            
            if search_button and address:
                with st.spinner("Finding location..."):
                    try:
                        # Use our direct geocoding function
                        location_result = geocode_address(address)
                        
                        if location_result:
                            lat = location_result['lat']
                            lon = location_result['lon']
                            location_display = location_result['display_name'].split(',')[0]
                            location_name = location_display
                            st.success(f"Found: {location_name}")
                            st.session_state[f"{type_prefix}_location_name"] = location_name
                        else:
                            # Fallback to simple matching if geocoding fails
                            address_lower = address.lower()
                            found = False
                            for name, coords in location_coords.items():
                                if coords and any(term in address_lower for term in name.lower().split(',')[0].split()):
                                    lat, lon = coords
                                    location_name = name
                                    st.success(f"Found: {location_name}")
                                    st.session_state[f"{type_prefix}_location_name"] = location_name
                                    found = True
                                    break
                            if not found:
                                # Absolute fallback
                                lat, lon = 40.7128, -74.0060  # NYC Center
                                location_name = f"Approx. for '{address}'"
                                st.info(f"Could not find exact match. Using approximate location.")
                                st.session_state[f"{type_prefix}_location_name"] = location_name
                    except Exception as e:
                        st.error(f"Error finding location: {str(e)}")
                        # Fallback if geocoding errors out
                        address_lower = address.lower()
                        found = False
                        for name, coords in location_coords.items():
                            if coords and any(term in address_lower for term in name.lower().split(',')[0].split()):
                                lat, lon = coords
                                location_name = name
                                st.session_state[f"{type_prefix}_location_name"] = location_name
                                found = True
                                break
                        if not found:
                            lat, lon = 40.7128, -74.0060  # NYC Center
                            location_name = f"Approx. for '{address}'"
                            st.session_state[f"{type_prefix}_location_name"] = location_name
                
        elif selection_method == "Enter Coordinates":
            coord_col1, coord_col2 = st.columns(2)
            with coord_col1:
                lat_input = st.number_input(f"Latitude", value=lat, format="%.4f", step=0.0001, key=f"{type_prefix}_lat_input")
            with coord_col2:
                lon_input = st.number_input(f"Longitude", value=lon, format="%.4f", step=0.0001, key=f"{type_prefix}_long_input")
            
            if lat_input != lat or lon_input != lon:
                lat, lon = lat_input, lon_input
                # Try to get location name from coordinates
                try:
                    location_display = reverse_geocode(lat, lon)
                    if location_display:
                        location_name = location_display.split(',')[0]
                    else:
                        location_name = f"Manual Coordinates ({lat:.4f}, {lon:.4f})"
                except Exception as e:
                    st.error(f"Error with reverse geocoding: {str(e)}")
                    location_name = f"Manual Coordinates ({lat:.4f}, {lon:.4f})"
                    
                st.session_state[f"{type_prefix}_location_name"] = location_name
                
        return lat, lon, location_name
    # --- End Helper Function ---
    
    # Default location (New York City center)
    default_lat, default_long = 40.7128, -74.0060
    
    # Initialize session state for location names if not present
    if 'source_location_name' not in st.session_state:
        st.session_state.source_location_name = "New York City Center (Default)"
    if 'dropoff_location_name' not in st.session_state:
        st.session_state.dropoff_location_name = "Empire State Building (Default)"
        
    # Create a compact box for location selection
    #st.markdown("<div class='location-box'>", unsafe_allow_html=True)
    
    # Two columns layout within the box
    loc_col1, loc_col2 = st.columns(2)
    
    with loc_col1:
        source_lat, source_long, source_location_name = compact_location_selection(
            'source', 
            default_lat, 
            default_long, 
            st.session_state.source_location_name
        )
        
    with loc_col2:
        dest_lat, dest_long, dropoff_location_name = compact_location_selection(
            'dropoff', 
            40.7484,  # Default dropoff Empire State Building
            -73.9857,
            st.session_state.dropoff_location_name
        )
    
    try:
        # Calculate distance (simple approximation)
        distance = np.sqrt((dest_lat - source_lat)**2 + (dest_long - source_long)**2) * 111  # Approx km
        
        # Show map with markers for pickup and dropoff
        st.markdown(f"**From:** {st.session_state.source_location_name} 🚩 **To:** {st.session_state.dropoff_location_name} 🏁")
        
        # Create a DataFrame for map locations
        map_data = pd.DataFrame({
            'lat': [source_lat, dest_lat],
            'lon': [source_long, dest_long],
            'location': ['Pickup', 'Dropoff'] # Simple labels for tooltip
        })
        
        # Get routing directions between points
        def get_route_coordinates(source_lat, source_long, dest_lat, dest_long):
            try:
                # Using Open Source Routing Machine (OSRM) which doesn't require API key
                url = f"https://router.project-osrm.org/route/v1/driving/{source_long},{source_lat};{dest_long},{dest_lat}?overview=full&geometries=geojson"
                response = requests.get(url, timeout=10)  # Increased timeout to 10 seconds
                if response.status_code == 200:
                    data = response.json()
                    if data["code"] == "Ok":
                        # Extract coordinates from the response
                        coordinates = data["routes"][0]["geometry"]["coordinates"]
                        # OSRM returns [lon, lat] pairs, so we need to swap them for Plotly
                        route_lons = [coord[0] for coord in coordinates]
                        route_lats = [coord[1] for coord in coordinates]
                        return route_lats, route_lons
            except Exception as e:
                st.warning(f"Using direct route. Could not fetch detailed directions: {str(e)}")
            
            # Fallback to straight line if API fails
            return [source_lat, dest_lat], [source_long, dest_long]
        
        # Get route coordinates
        with st.spinner("Getting route directions..."):
            route_lats, route_lons = get_route_coordinates(source_lat, source_long, dest_lat, dest_long)
        
        # Create a map with route directions using Plotly
        fig = go.Figure()
        
        # Add the route line (using the fetched coordinates)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=route_lons,
            lat=route_lats,
            line=dict(width=3, color="#27B666"),
            name="Route"
        ))
        
        # Add pickup marker
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[source_long],
            lat=[source_lat],
            marker=dict(size=12, color="#FF0000", symbol="circle"),
            name="Pickup",
            text=["Pickup"]
        ))
        
        # Add dropoff marker
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[dest_long],
            lat=[dest_lat],
            marker=dict(size=12, color="#0000FF", symbol="circle"),
            name="Dropoff",
            text=["Dropoff"]
        ))
        
        # Set map center and zoom
        center_lat = (source_lat + dest_lat) / 2
        center_lon = (source_long + dest_long) / 2
        
        # Configure the layout - fix the invalid properties
        fig.update_layout(
            # Main layout configuration
            dragmode="pan",  # Set drag mode at the top level
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            showlegend=False,
            
            # Mapbox-specific configuration
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
                uirevision=True
            ),
            
            # Add modebar buttons including zoom controls
            modebar_add=["zoomIn", "zoomOut", "resetViewMapbox", "toImage"]
        )
        
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
        
        # Display route information
        route_col1, route_col2, route_col3 = st.columns(3)
        with route_col1:
            st.metric("Distance", f"{distance:.2f} km")
        with route_col2:
            # Estimate time (assuming average speed of 30 km/h in the city)
            est_time_mins = max(1, int(distance / 30 * 60)) # Ensure at least 1 min
            st.metric("Est. Time", f"{est_time_mins} min")
        with route_col3:
            # Calculate base price
            timestamp = datetime.now() # Need a timestamp to check rush hour
            is_rush_hour_final = 0
            if ((timestamp.hour >= 7 and timestamp.hour <= 9) or 
                (timestamp.hour >= 16 and timestamp.hour <= 18)) and timestamp.weekday() < 5:
                is_rush_hour_final = 1
            base_price = calculate_base_price(distance, 'UberX', is_rush_hour_final)
            st.metric("Est. Base Price", f"${base_price:.2f}")
    
    except Exception as e:
        st.error(f"Error calculating or displaying route: {str(e)}")
        # Set default distance/price if calculation fails
        distance = 5.0
        base_price = calculate_base_price(distance, 'UberX', False)
        
    #st.markdown("</div>", unsafe_allow_html=True)  # End location-box
    
    # Time and Weather section
    #st.markdown("<div class='time-weather-section'>", unsafe_allow_html=True)
    time_weather_col1, time_weather_col2 = st.columns(2)
    
    with time_weather_col1:
        st.subheader("Trip Time")
        
        # Initialize session state for date and time if not present
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = datetime.now().date()
        if 'selected_time' not in st.session_state:
            st.session_state.selected_time = datetime.now().time()
            
        # Time details with session state
        date_input = st.date_input("Date", value=st.session_state.selected_date)
        time_input = st.time_input("Time", value=st.session_state.selected_time)
        
        # Only update session state if the values actually changed
        if date_input != st.session_state.selected_date:
            st.session_state.selected_date = date_input
            
        if time_input != st.session_state.selected_time:
            st.session_state.selected_time = time_input
        
        # Combine date and time into a datetime object
        timestamp = datetime.combine(st.session_state.selected_date, st.session_state.selected_time)
        
        # Store as a single timestamp object too
        st.session_state.selected_timestamp = timestamp
    
    with time_weather_col2:
        st.subheader("Weather Conditions")
        
        # Initialize weather session state if not present
        if 'use_real_weather' not in st.session_state:
            st.session_state.use_real_weather = True
        if 'weather_data' not in st.session_state:
            st.session_state.weather_data = None
        if 'manual_temperature' not in st.session_state:
            st.session_state.manual_temperature = 20.0
        if 'manual_humidity' not in st.session_state:
            st.session_state.manual_humidity = 50.0
        if 'manual_wind_speed' not in st.session_state:
            st.session_state.manual_wind_speed = 10.0
        if 'manual_precipitation_type' not in st.session_state:
            st.session_state.manual_precipitation_type = "clear"
        
        # Option to use real-time weather data
        use_real_weather = st.checkbox("Use real-time weather data", value=st.session_state.use_real_weather, key="use_real_weather")
        
        # Update session state if changed
        if use_real_weather != st.session_state.use_real_weather:
            st.session_state.use_real_weather = use_real_weather
            
        if use_real_weather:
            # Fetch weather for the pickup location only if needed
            if st.session_state.weather_data is None:
                with st.spinner("Fetching current weather data..."):
                    weather_data = fetch_weather_data(source_lat, source_long)
                    if weather_data:
                        st.session_state.weather_data = weather_data
            else:
                # Use cached weather data
                weather_data = st.session_state.weather_data
                
            if weather_data:
                # Show weather emoji and main condition
                weather_emoji = get_weather_emoji(weather_data['precipitation_type'])
                st.markdown(f"<h3 style='text-align: center;'>{weather_emoji} {weather_data['weather_description'].capitalize()}</h3>", unsafe_allow_html=True)
                
                # Display metrics in a grid using markdown
                st.markdown(f"""
                <div class="weather-grid">
                    <div class="weather-card">
                        <h4 style="margin:0;">Temperature</h4>
                        <h2 style="margin:5px 0;">{weather_data['temperature']:.1f}°C</h2>
                    </div>
                    <div class="weather-card">
                        <h4 style="margin:0;">Humidity</h4>
                        <h2 style="margin:5px 0;">{weather_data['humidity']}%</h2>
                    </div>
                    <div class="weather-card">
                        <h4 style="margin:0;">Wind Speed</h4>
                        <h2 style="margin:5px 0;">{weather_data['wind_speed']:.1f} km/h</h2>
                    </div>
                    <div class="weather-card">
                        <h4 style="margin:0;">Pressure</h4>
                        <h2 style="margin:5px 0;">{weather_data['pressure']} hPa</h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Weather map link
                st.markdown(f"[View Weather Map](https://openweathermap.org/weathermap?lat={source_lat}&lon={source_long})")
                
                # Set weather type for background
                weather_type = weather_data['precipitation_type']
                
                # Apply dynamic background based on time and weather
                current_hour = timestamp.hour
                bg_css = get_weather_background_css(current_hour, weather_type)
                st.markdown(bg_css, unsafe_allow_html=True)
                
                # Use the fetched data for prediction
                temperature = weather_data['temperature']
                humidity = weather_data['humidity']
                wind_speed = weather_data['wind_speed']
                precipitation_type = weather_data['precipitation_type']
            else:
                st.error("Failed to fetch weather data. Using manual inputs.")
                use_real_weather = False
                st.session_state.use_real_weather = False
        
        # Manual weather inputs as fallback or if real-time data is not selected
        if not use_real_weather:
            temperature = st.slider("Temperature (°C)", 
                                   min_value=-10.0, 
                                   max_value=40.0, 
                                   value=st.session_state.manual_temperature, 
                                   step=0.5,
                                   key="manual_temperature")
            
            humidity = st.slider("Humidity (%)", 
                               min_value=0.0, 
                               max_value=100.0, 
                               value=st.session_state.manual_humidity, 
                               step=5.0,
                               key="manual_humidity")
            
            wind_speed = st.slider("Wind Speed (km/h)", 
                                 min_value=0.0, 
                                 max_value=50.0, 
                                 value=st.session_state.manual_wind_speed, 
                                 step=1.0,
                                 key="manual_wind_speed")
            
            precipitation_type = st.selectbox("Weather Condition", 
                                           ["clear", "rain", "snow", "fog", "hail", "thunderstorm", "sleet"],
                                           index=["clear", "rain", "snow", "fog", "hail", "thunderstorm", "sleet"].index(st.session_state.manual_precipitation_type),
                                           key="manual_precipitation_type")
            
            # Update session state
            st.session_state.manual_temperature = temperature
            st.session_state.manual_humidity = humidity
            st.session_state.manual_wind_speed = wind_speed
            st.session_state.manual_precipitation_type = precipitation_type
    
    st.markdown("</div>", unsafe_allow_html=True)  # End time-weather-section
    
    # Ride options section - connect directly to time-weather section
    st.markdown("<div class='ride-options'>", unsafe_allow_html=True)
    st.subheader("Ride Options")
    
    ride_col1, ride_col2 = st.columns(2)
    
    # Define service levels for each cab type
    uber_services = ["UberX", "UberXL", "UberPOOL", "Black", "SUV"] 
    lyft_services = ["Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL", "Shared"]
    all_services = sorted(list(set(uber_services + lyft_services)))

    with ride_col1:
        cab_type = st.selectbox("Cab Type", ["Uber", "Lyft"], key="cab_type_select")
    
    with ride_col2:
        # Dynamically set service level options based on cab_type
        if cab_type == "Uber":
            available_services = uber_services
            default_service = "UberX"
        elif cab_type == "Lyft":
            available_services = lyft_services
            default_service = "Lyft"
        else: # Fallback if needed
            available_services = all_services
            default_service = available_services[0]

        # Ensure default is in the available list
        if default_service not in available_services:
             default_service = available_services[0]

        product_id = st.selectbox("Service Level", 
                                 available_services,
                                 index=available_services.index(default_service),
                                 key="product_id_select")
    
    st.markdown("</div>", unsafe_allow_html=True)  # End ride-options

    # Add prediction button using the custom class
    st.markdown("<div class='prediction-button'>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Surge Pricing", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle prediction
    if predict_button:
        with st.spinner("Calculating surge pricing..."):
            try:
                # Prepare input data
                # Map form inputs to expected feature names
                input_features = { 
                    'source_lat': source_lat,
                    'source_lon': source_long,
                    'destination_lat': dest_lat,
                    'destination_lon': dest_long,
                    'distance': distance,
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'precipitation_type': precipitation_type, 
                    'cab_type': cab_type, # Will be handled later if needed
                    'product_id': product_id, # Will be handled later if needed
                    # Add time features
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'month': timestamp.month,
                    'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
                    'is_rush_hour': 1 if (((timestamp.hour >= 7 and timestamp.hour <= 9) or (timestamp.hour >= 16 and timestamp.hour <= 18)) and timestamp.weekday() < 5) else 0,
                    'is_night': 1 if (timestamp.hour >= 22 or timestamp.hour <= 4) else 0,
                    'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
                    'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
                    'day_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
                    'day_cos': np.cos(2 * np.pi * timestamp.weekday() / 7),
                    # Add derived weather features (match training if needed)
                    'has_precipitation': 1 if precipitation_type not in ['clear', 'none', None] else 0,
                }

                # Create DataFrame from the features dictionary
                input_df = pd.DataFrame([input_features])

                # Handle categorical features like 'cab_type' and 'product_id'
                # This needs to match the encoding used during training (e.g., one-hot)
                # Assuming one-hot encoding was used and the model has feature_names attribute
                if hasattr(model, 'feature_names') and model.feature_names:
                    # --- Cab Type Encoding ---
                    # Create columns for expected cab_type features (e.g., is_uber, is_lyft)
                    if 'is_uber' in model.feature_names:
                        input_df['is_uber'] = 1 if cab_type == 'Uber' else 0
                    if 'is_lyft' in model.feature_names:
                        input_df['is_lyft'] = 1 if cab_type == 'Lyft' else 0
                    # Drop the original cab_type if encoded features were added
                    if 'is_uber' in input_df.columns or 'is_lyft' in input_df.columns:
                        if 'cab_type' in input_df.columns:
                            input_df = input_df.drop(columns=['cab_type'])

                    # --- Product ID Encoding ---
                    # Example: Create columns for specific product IDs seen during training
                    # This is a simplified example; a more robust approach might use pd.get_dummies
                    # aligned with training columns.
                    expected_product_cols = [f for f in model.feature_names if f.startswith('product_id_')]
                    for col in expected_product_cols:
                        # Extract product name from column name (e.g., product_id_UberX -> UberX)
                        expected_product_name = col.replace('product_id_', '') 
                        input_df[col] = 1 if product_id == expected_product_name else 0
                    # Drop original product_id if encoded features were added
                    if expected_product_cols and 'product_id' in input_df.columns:
                         input_df = input_df.drop(columns=['product_id'])

                    # --- Precipitation Type Encoding (if needed) ---
                    expected_precip_cols = [f for f in model.feature_names if f.startswith('precipitation_type_')]
                    for col in expected_precip_cols:
                        expected_precip_name = col.replace('precipitation_type_', '')
                        input_df[col] = 1 if precipitation_type == expected_precip_name else 0
                    if expected_precip_cols and 'precipitation_type' in input_df.columns:
                        input_df = input_df.drop(columns=['precipitation_type'])

                    # --- Ensure all expected columns exist and are in the correct order ---
                    # Create a DataFrame with zeros for all expected feature names
                    aligned_df = pd.DataFrame(columns=model.feature_names)
                    # Fill with the input data where columns match
                    for col in model.feature_names:
                        if col in input_df.columns:
                            aligned_df[col] = input_df[col]
                        else:
                            # Handle potentially missing columns (e.g., clusters not calculated here)
                            # Set to a default value like 0, or raise error if critical
                            aligned_df[col] = 0 
                    
                    input_df_final = aligned_df.fillna(0) # Fill any NaNs just in case

                else:
                    st.warning("Model feature names not found. Prediction might be inaccurate.")
                    # Fallback: Try using the input_df as is, but it might fail
                    # Drop categorical columns if they likely weren't used directly
                    cols_to_drop = [c for c in ['cab_type', 'product_id', 'precipitation_type'] if c in input_df.columns]
                    input_df_final = input_df.drop(columns=cols_to_drop)


                # --- Prediction --- 
                # Check if predict_single exists and handles feature processing
                if hasattr(model, 'predict_single') and callable(model.predict_single):
                     # Assuming predict_single takes the raw dictionary and handles alignment
                     predicted_surge = model.predict_single(input_features) 
                elif hasattr(model, 'predict') and callable(model.predict):
                     # Use the carefully aligned DataFrame
                     prediction_result = model.predict(input_df_final)
                     # Check if the result is array-like or a single value
                     if isinstance(prediction_result, (np.ndarray, list)) and len(prediction_result) > 0:
                         predicted_surge = prediction_result[0]
                     elif isinstance(prediction_result, (float, np.floating)): # Handle single float/numpy float
                         predicted_surge = prediction_result
                     else:
                         st.error(f"Unexpected prediction result type: {type(prediction_result)}. Cannot extract surge value.")
                         return # Stop execution
                else:
                     st.error("Loaded model object does not have a valid 'predict' or 'predict_single' method.")
                     return # Stop execution

                # Calculate final prices
                # Recalculate base_price here with the selected product_id
                hour = timestamp.hour
                weekday = timestamp.weekday()
                # Check for morning rush hour (7-9 AM weekdays)
                morning_rush = (7 <= hour <= 9) and (weekday < 5)
                # Check for evening rush hour (4-6 PM weekdays)
                evening_rush = (16 <= hour <= 18) and (weekday < 5)
                is_rush_hour_final = morning_rush or evening_rush

                final_base_price = calculate_base_price(distance, product_id, is_rush_hour_final)
                result = format_prediction_output(predicted_surge, final_base_price)
                
                # Display prediction results using the custom class
                st.markdown("<div class='prediction-results'>", unsafe_allow_html=True)
                st.subheader("Surge Pricing Prediction")
                
                # Show the result
                results_col1, results_col2, results_col3 = st.columns(3)
                
                with results_col1:
                    st.metric("Surge Multiplier", f"{result['surge_multiplier']}x")
                
                with results_col2:
                    st.metric("Base Price", f"${result['base_price']}")
                
                with results_col3:
                    st.metric("Total Price", f"${result['total_price']}")
                
                # Additional explanation about the surge
                if result['surge_multiplier'] > 1.5:
                    st.warning("⚠️ Surge pricing is high! Consider waiting or choosing a different time.")
                elif result['surge_multiplier'] > 1.2:
                    st.info("ℹ️ Moderate surge pricing in effect.")
                else:
                    st.success("✅ Low surge pricing. Good time to book a ride!")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                prediction_successful = True # Mark as successful
                
            except Exception as e:
                st.error(f"Error predicting surge pricing: {str(e)}")
                st.error("Please make sure all inputs are valid and try again.")
                prediction_successful = False # Mark as failed
        # End of the spinner block for main prediction

        # --- Find and Display Cheapest Time (Run this *after* the spinner, only if successful) ---
        if prediction_successful:
             with st.spinner("Analyzing best time to travel today..."):
                 # Ensure input_features dictionary exists from the successful prediction try block
                 if 'input_features' in locals(): 
                     # Use the user-selected timestamp for calculating best time
                     user_timestamp = timestamp
                     best_time, best_price, all_predictions = find_cheapest_time_today(model, input_features, user_timestamp, distance, product_id)
                 else: 
                     best_time, best_price, all_predictions = None, None, [] # Cannot analyze if base features weren't prepared
             
             if best_time is not None:
                 st.markdown("<div class='prediction-results'>", unsafe_allow_html=True)
                 st.subheader("💡 Best Time to Travel")
                 
                 # Format time for display
                 display_hour = best_time.hour % 12 if best_time.hour % 12 != 0 else 12
                 display_minute = f"{best_time.minute:02d}"
                 am_pm = "AM" if best_time.hour < 12 else "PM"
                 day_label = "Today" if best_time.date() == datetime.now().date() else "Tomorrow"
                 
                 st.success(f"Estimated cheapest time: **{day_label} at {display_hour}:{display_minute} {am_pm}**")
                 st.metric("Estimated Price", f"${best_price:.2f}", delta=f"-${result['total_price'] - best_price:.2f}")
                 
                 # Create price chart over the day
                 if len(all_predictions) > 0:
                     # Prepare data for chart
                     chart_data = pd.DataFrame(
                         [(p[0], p[1], p[2]) for p in all_predictions],
                         columns=['Time', 'Price', 'Surge']
                     )
                     
                     # Create the time series chart
                     fig = px.line(
                         chart_data, 
                         x='Time', 
                         y='Price',
                         title='Price Forecast for Next 24 Hours',
                         markers=True
                     )
                     
                     # Add point for current time and price
                     current_time_data = pd.DataFrame([
                         {'Time': timestamp, 'Price': result['total_price'], 'Type': 'Current Time'}
                     ])
                     best_time_data = pd.DataFrame([
                         {'Time': best_time, 'Price': best_price, 'Type': 'Best Time'}
                     ])
                     
                     # Highlight the best time with a different marker
                     fig.add_scatter(
                         x=best_time_data['Time'], 
                         y=best_time_data['Price'],
                         mode='markers',
                         marker=dict(size=12, color='green', symbol='star'),
                         name='Best Time'
                     )
                     
                     # Highlight current time
                     fig.add_scatter(
                         x=current_time_data['Time'], 
                         y=current_time_data['Price'],
                         mode='markers',
                         marker=dict(size=12, color='red'),
                         name='Current Time'
                     )
                     
                     # Improve layout
                     fig.update_layout(
                         hovermode="x unified",
                         xaxis_title="Time",
                         yaxis_title="Price ($)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                     )
                     
                     # Display the chart
                     st.plotly_chart(fig, use_container_width=True)
                     
                     # Show savings info
                     savings = result['total_price'] - best_price
                     if savings > 0:
                         st.info(f"💰 By traveling at the suggested time, you could save **${savings:.2f}** (about {int(savings/result['total_price']*100)}% of the current price).")
                 
                 st.caption("_Note: This prediction assumes current weather conditions persist throughout the forecast period._")
                 st.markdown("</div>", unsafe_allow_html=True)
             else:
                 st.info("Could not determine the best time to travel (analysis might have failed or no cheaper time found).")
        # --- End Cheapest Time Section ---


def display_dashboard_tab():
    """Display the dashboard tab content"""
    st.header("Surge Pricing Dashboard", anchor=False)
    
    # Load data
    data, processor, success = load_data()
    
    if not success:
        st.warning("⚠️ Required data files not found. Please upload cab_rides.csv and weather.csv files to the data folder.")
        return
    
    # Extract time features if needed
    if 'timestamp' in data.columns and 'hour' not in data.columns:
        data = data.copy()  # Make a proper copy to avoid SettingWithCopyWarning
        data.loc[:, 'hour'] = data['timestamp'].dt.hour
        data.loc[:, 'day_of_week'] = data['timestamp'].dt.dayofweek
        data.loc[:, 'is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data.loc[:, 'is_rush_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9) | 
                              (data['hour'] >= 16) & (data['hour'] <= 18)).astype(int)
    
    # Display key metrics
    st.subheader("Key Metrics")
    
    metric1, metric2, metric3, metric4 = st.columns(4)
    
    with metric1:
        avg_surge = data['surge_multiplier'].mean()
        st.metric("Average Surge", f"{avg_surge:.2f}x")
    
    with metric2:
        max_surge = data['surge_multiplier'].max()
        st.metric("Maximum Surge", f"{max_surge:.2f}x")
    
    with metric3:
        avg_price = data['price'].mean() if 'price' in data.columns else 0
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with metric4:
        data_points = len(data)
        st.metric("Data Points", f"{data_points:,}")
    
    # Visualizations
    st.subheader("Surge Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Time Analysis", "Weather Impact", "Price History"])
    
    with tab1:
        view_option = st.radio("View By:", ["Hour of Day", "Day of Week"], horizontal=True)
        
        if view_option == "Hour of Day":
            fig = create_surge_trend_plot(data, time_group='hour')
        else:
            fig = create_surge_trend_plot(data, time_group='day_of_week')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        *The chart above shows how surge pricing varies throughout the day or week. 
        Higher peaks indicate times when surge pricing is most common.*
        """)
    
    with tab2:
        fig = create_weather_impact_plot(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            *This visualization shows how different weather conditions affect surge pricing.
            Poor weather conditions like rain and snow often lead to higher surge multipliers.*
              """)
        else:
            st.info("Weather impact visualization not available for this dataset.")
    
    with tab3:
        time_window = st.slider("Time Window (hours)", min_value=6, max_value=72, value=24, step=6)
        fig = create_price_history_plot(data, time_window=time_window)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            *This chart displays the surge multiplier history over the selected time window. 
            It helps identify trends and patterns in surge pricing over time.*
            """)
        else:
            st.info("Price history visualization not available for this dataset.")


def display_training_tab():
    """Display the model training tab content"""
    st.header("Train Surge Prediction Model", anchor=False)
    
    # Check if data is available
    data, processor, success = load_data()
    
    if not success:
        st.warning("⚠️ Required data files not found. Please upload cab_rides.csv and weather.csv files to the data folder.")
        return
    
    # Model settings
    st.subheader("Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type", ["xgboost", "random_forest"])
        test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    
    with col2:
        cv_folds = st.number_input("Cross-Validation Folds", min_value=3, max_value=10, value=5, step=1)
        random_state = st.number_input("Random Seed", min_value=0, max_value=100, value=42, step=1)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Extract features from the data
    # Make sure we have time features
    if 'hour' not in data.columns and 'timestamp' in data.columns:
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_rush_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9) | 
                             (data['hour'] >= 16) & (data['hour'] <= 18)).astype(int)
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 4)).astype(int)
        
        # Time cyclic features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Create weather features if needed
    if 'precipitation_type' in data.columns and 'has_precipitation' not in data.columns:
        data['has_precipitation'] = (~data['precipitation_type'].isin(['clear', 'none', None])).astype(int)
    
    # Create ride type features if needed
    if 'cab_type' in data.columns and 'is_uber' not in data.columns:
        data['is_uber'] = (data['cab_type'] == 'Uber').astype(int)
        data['is_lyft'] = (data['cab_type'] == 'Lyft').astype(int)
    
    # Get available features
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    available_features = [col for col in numeric_cols if col not in ['surge_multiplier', 'price']]
    
    # Group features by category
    time_features = [f for f in available_features if any(tf in f for tf in ['hour', 'day', 'month', 'sin_', 'cos_', 'weekend', 'rush', 'night'])]
    weather_features = [f for f in available_features if any(wf in f for wf in ['temp', 'humidity', 'wind', 'precip', 'has_precipitation'])]
    location_features = [f for f in available_features if any(lf in f for lf in ['lat', 'lon', 'cluster', 'source', 'dest', 'distance'])]
    ride_features = [f for f in available_features if any(rf in f for rf in ['cab_type', 'uber', 'lyft', 'product'])]
    
    # Create feature selection expanders
    with st.expander("Time Features", expanded=False):
        selected_time_features = st.multiselect("Select Time Features", time_features, default=time_features)
    
    with st.expander("Weather Features", expanded=False):
        selected_weather_features = st.multiselect("Select Weather Features", weather_features, default=weather_features)
    
    with st.expander("Location Features", expanded=False):
        selected_location_features = st.multiselect("Select Location Features", location_features, default=location_features)
    
    with st.expander("Ride Features", expanded=False):
        selected_ride_features = st.multiselect("Select Ride Features", ride_features, default=ride_features)
    
    # Combine selected features
    selected_features = selected_time_features + selected_weather_features + selected_location_features + selected_ride_features
    
    # Train button
    train_col1, train_col2, train_col3 = st.columns([1, 1, 1])
    
    with train_col2:
        train_button = st.button("Train Model", type="primary", use_container_width=True)
    
    # Training process
    if train_button:
        if len(selected_features) < 3:
            st.error("Please select at least 3 features for training.")
            return
        
        # Create custom model name
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_{current_time}.joblib"
        
        # Optional: Let user name their model
        with st.expander("Model Name (Optional)", expanded=True):
            st.info("You can customize the model filename or keep the auto-generated one.")
            custom_name = st.text_input(
                "Custom Model Name (optional):", 
                value=model_filename,
                help="Use a descriptive name to identify this model later"
            )
            
            if custom_name and not custom_name.endswith('.joblib'):
                custom_name = f"{custom_name}.joblib"
            
            if custom_name:
                model_filename = custom_name
        
        with st.spinner(f"Training model {model_filename}... This may take a few minutes."):
            # Prepare features for model
            X = data[selected_features].copy()
            y = data['surge_multiplier'].copy()
            
            # Create output directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Create and train model
            model = SurgePredictor(model_type=model_type)
            
            # Train model with progress bar
            progress_bar = st.progress(0)
            placeholder = st.empty()
            
            # Mock training progress
            for i in range(101):
                progress = i / 100
                placeholder.text(f"Training progress: {i}%")
                progress_bar.progress(progress)
                if i < 80:  # Slow down the first 80%
                    time.sleep(0.05)
                else:  # Speed up the last 20%
                    time.sleep(0.02)
            
            # Actual training
            metrics = model.train(X, y, test_size=test_size)
            
            # Create model metadata
            model.metadata = {
                'model_type': model_type,
                'test_size': test_size,
                'cv_folds': cv_folds,
                'random_state': random_state,
                'feature_count': len(selected_features),
                'features': selected_features,
                'timestamp': current_time,
                'metrics': metrics
            }
            
            # Save model to the models directory with the selected filename
            model_path = os.path.join('models', model_filename)
            model.save_model(model_path)
            
            # Show final progress
            placeholder.text("Training completed!")
            progress_bar.progress(1.0)
        
        # Display training results
        st.success(f"Model trained successfully and saved as '{model_filename}'")
        
        # Store model name in session state so it can be selected in prediction tab
        st.session_state.selected_model = model_filename
        
        # Display metrics
        st.subheader("Model Performance")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
        
        with metric_col2:
            st.metric("Root MSE (RMSE)", f"{metrics['rmse']:.4f}")
        
        with metric_col3:
            st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
        
        with metric_col4:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        
        # Display feature importance
        st.subheader("Feature Importance")
        
        # Get feature importance
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            
            # Display feature importance as a table
            st.dataframe(feature_importance.head(10))
        else:
            st.info("Feature importance not available for this model type.")
        
        # Training information
        st.subheader("Training Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.text(f"Model Type: {model_type.title()}")
            st.text(f"Data Points: {len(y)}")
            st.text(f"Features Used: {len(selected_features)}")
        
        with info_col2:
            st.text(f"Test Size: {test_size:.2f}")
            st.text(f"Random Seed: {random_state}")
            st.text(f"Cross-Validation Folds: {cv_folds}")
        
        # Save notification
        st.success(f"Model saved to models/surge_model.joblib")


def display_about_tab():
    """Display the about tab content"""
    st.header("About Surge Price Predictor", anchor=False)
    
    st.markdown("""
    ### Overview
    
    The Surge Price Predictor is a machine learning application designed to predict ride-sharing surge pricing based on various factors like time, weather, location, and demand.
    
    ### Features
    
    * **Surge Price Prediction**: Get accurate predictions of surge multipliers for your ride
    * **Interactive Dashboard**: Visualize surge pricing patterns and trends
    * **Custom Model Training**: Train models with your own data and feature selection
    * **Weather Integration**: See how weather conditions affect surge pricing
    
    ### How It Works
    
    1. **Data Collection**: The system collects ride and weather data
    2. **Feature Engineering**: Extracts and transforms features from the data
    3. **Model Training**: Trains machine learning models on the processed data
    4. **Prediction**: Uses trained models to predict surge multipliers
    
    ### Technologies Used
    
    * **Python**: Core programming language
    * **Streamlit**: Web application framework
    * **Pandas & NumPy**: Data manipulation and analysis
    * **Scikit-learn & XGBoost**: Machine learning models
    * **Plotly & Matplotlib**: Data visualization
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### How to Use
    
    1. **Prediction Tab**: Enter ride details and get a surge price prediction
    2. **Dashboard Tab**: Explore surge pricing patterns and visualizations
    3. **Training Tab**: Train custom models with your own feature selection
    
    ### Data Requirements
    
    The application requires two CSV files in the `data` directory:
    
    * `cab_rides.csv`: Contains ride data with columns for timestamp, source/destination coordinates, price, surge multiplier, etc.
    * `weather.csv`: Contains weather data with columns for timestamp, temperature, humidity, wind speed, precipitation type, etc.
    """)


def main():
    """Main function to run the Streamlit app"""
    
    # Header and navigation
    st.title("🚕 Surge Price Predictor")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediction", "📊 Dashboard", "🧠 Training", "ℹ️ About"
    ])
    
    # Fill each tab with content
    with tab1:
        display_prediction_tab()
    
    with tab2:
        display_dashboard_tab()
    
    with tab3:
        display_training_tab()
    
    with tab4:
        display_about_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #888;'>
        Surge Price Predictor v1.0 | Made with Streamlit 
        </div>""", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 