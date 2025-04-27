# RideWise: Surge Price Predictor

A machine learning application to predict ride-sharing surge pricing based on various factors like time, weather, location, and demand.

## Project Structure

```
RideWise/
├── data/              # Data directory
│   ├── cab_rides.csv  # Ride-sharing data
│   └── weather.csv    # Weather data
│
├── models/            # Saved ML models
│
├── src/               # Source code
│   ├── data_processor.py     # Data processing functionality
│   ├── model.py              # ML model implementation
│   ├── utils.py              # Utility functions
│   ├── feature_engineering.py # Feature engineering code
│   ├── data_preprocessing.py  # Data preprocessing functions
│   └── model_training.py      # Model training functions
│
├── web/               # Web application
│   └── app.py         # Streamlit app code
│
└── requirements.txt   # Project dependencies
```

## Setup and Installation

There are two ways to set up the RideWise application:

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/RideWise.git
cd RideWise

# Install the package in development mode
pip install -e .
```

### Option 2: Manual setup

```bash
# Clone the repository
git clone https://github.com/yourusername/RideWise.git
cd RideWise

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## API Keys Configuration

This application requires API keys to enable weather data retrieval, route visualization, and Uber price estimates:

1. **OpenWeatherMap API Key** - For real-time weather data
2. **Google Maps API Key** - For route visualization and distance calculation
3. **Uber OAuth Credentials** - For real-time price estimates from Uber

Create a `.env` file in the project root directory with the following contents:

```
weather_api_key=YOUR_OPENWEATHERMAP_API_KEY
google_maps_api_key=YOUR_GOOGLE_MAPS_API_KEY
uber_client_id=YOUR_UBER_CLIENT_ID
uber_client_secret=YOUR_UBER_CLIENT_SECRET
```

### How to obtain the API keys:

- **OpenWeatherMap API Key**: Register at [OpenWeatherMap](https://openweathermap.org/api) and get a free API key
- **Google Maps API Key**: Get a key from the [Google Cloud Console](https://developers.google.com/maps/documentation/embed/get-api-key)
- **Uber OAuth Credentials**: 
  1. Register as a developer at the [Uber Developer Portal](https://developer.uber.com/)
  2. Create a new application
  3. Note your Client ID and Client Secret (these are different from a server token)
  4. For the OAuth 2.0 client credentials flow, you'll need to specify appropriate scopes for your application (typically 'pricing')

Note: The application will still work without these keys, but with limited functionality:
- Without the weather API key, you'll need to input weather data manually
- Without the Google Maps API key, a simplified map will be shown without route details
- Without the Uber credentials, estimated prices will be shown instead of real-time Uber prices

## Running the Application

You can run the application in two ways:

### Option 1: Using the run_app.py script (recommended)

```bash
python run_app.py
```

### Option 2: Running the Streamlit app directly

```bash
# Make sure you're in the project root directory
PYTHONPATH=$PYTHONPATH:$(pwd) streamlit run web/app.py
```

Either option will launch the application at http://localhost:8501.

## Features

- **Surge Price Prediction**: Get accurate predictions of surge multipliers for your ride
- **Interactive Dashboard**: Visualize surge pricing patterns and trends
- **Custom Model Training**: Train models with your own data and feature selection
- **Weather Integration**: See how weather conditions affect surge pricing
- **Google Maps Integration**: Visualize routes between pickup and dropoff locations

## Data Requirements

The application requires two CSV files in the `data` directory:

- `cab_rides.csv`: Contains ride data with columns for timestamp, source/destination coordinates, price, surge multiplier, etc.
- `weather.csv`: Contains weather data with columns for timestamp, temperature, humidity, wind speed, precipitation type, etc.

## Model Training

1. Navigate to the Training tab in the web interface
2. Select model type and parameters
3. Choose features to include in the model
4. Click "Train Model" to train and save your model 