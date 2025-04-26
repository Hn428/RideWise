import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os


def create_surge_trend_plot(df: pd.DataFrame, time_group: str = 'hour') -> go.Figure:
    """
    Create a plot showing surge pricing trends over time.
    
    Args:
        df: DataFrame containing ride data
        time_group: Time grouping ('hour' or 'day_of_week')
        
    Returns:
        Plotly figure object
    """
    if time_group == 'hour':
        # Group by hour
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
        labels={
            x_col: x_label,
            'surge_multiplier': 'Average Surge Multiplier'
        }
    )
    
    # Add markers
    fig.update_traces(mode='lines+markers', marker=dict(size=8))
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        xaxis_title=x_label,
        yaxis_title='Average Surge Multiplier',
        title_x=0.5,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='white', font_size=12)
    )
    
    return fig


def create_weather_impact_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a plot showing the impact of weather conditions on surge pricing.
    
    Args:
        df: DataFrame containing ride and weather data
        
    Returns:
        Plotly figure object
    """
    # Handle missing values in precipitation_type
    df_clean = df.copy()
    df_clean['precipitation_type'] = df_clean['precipitation_type'].fillna('Unknown')
    
    # Group by precipitation type
    weather_impact = df_clean.groupby('precipitation_type')['surge_multiplier'].mean().reset_index()
    
    # Create the plot
    fig = px.bar(
        weather_impact,
        x='precipitation_type',
        y='surge_multiplier',
        title='Impact of Weather Conditions on Surge Pricing',
        labels={
            'precipitation_type': 'Weather Condition',
            'surge_multiplier': 'Average Surge Multiplier'
        },
        color='surge_multiplier',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        xaxis_title='Weather Condition',
        yaxis_title='Average Surge Multiplier',
        title_x=0.5,
        coloraxis_showscale=False
    )
    
    return fig


def create_price_history_plot(df: pd.DataFrame, time_window: Optional[int] = 24) -> go.Figure:
    """
    Create a plot showing price history for the past time window.
    
    Args:
        df: DataFrame containing ride data
        time_window: Time window in hours to display (default: 24)
        
    Returns:
        Plotly figure object
    """
    # Filter data to the time window if specified
    if time_window is not None:
        # Get the latest timestamp
        latest_time = df['timestamp'].max()
        cutoff_time = latest_time - pd.Timedelta(hours=time_window)
        df_window = df[df['timestamp'] >= cutoff_time]
    else:
        df_window = df
    
    # Group by hour and cab_type
    df_window['hour_bin'] = df_window['timestamp'].dt.floor('H')
    
    # Check if cab_type exists
    if 'cab_type' in df_window.columns:
        hourly_data = df_window.groupby(['hour_bin', 'cab_type']).agg({
            'surge_multiplier': 'mean',
            'price': 'mean'
        }).reset_index()
        
        # Create the plot
        fig = px.line(
            hourly_data, 
            x='hour_bin', 
            y='surge_multiplier', 
            color='cab_type',
            title=f'Surge Multiplier History (Past {time_window} Hours)',
            labels={
                'hour_bin': 'Time',
                'surge_multiplier': 'Average Surge Multiplier',
                'cab_type': 'Service'
            }
        )
    else:
        # If no cab_type, just group by hour
        hourly_data = df_window.groupby('hour_bin').agg({
            'surge_multiplier': 'mean',
            'price': 'mean'
        }).reset_index()
        
        # Create the plot
        fig = px.line(
            hourly_data, 
            x='hour_bin', 
            y='surge_multiplier',
            title=f'Surge Multiplier History (Past {time_window} Hours)',
            labels={
                'hour_bin': 'Time',
                'surge_multiplier': 'Average Surge Multiplier'
            }
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        xaxis_title='Time',
        yaxis_title='Average Surge Multiplier',
        title_x=0.5,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='white', font_size=12)
    )
    
    return fig


def calculate_base_price(distance: float, is_rush_hour: bool = False) -> float:
    """
    Calculate base price for a ride.
    
    Args:
        distance: Distance in kilometers
        is_rush_hour: Whether the ride is during rush hour
        
    Returns:
        Base price for the ride
    """
    base_rate = 2.5  # Base fare
    per_km_rate = 1.75  # Rate per kilometer
    rush_hour_multiplier = 1.2 if is_rush_hour else 1.0
    
    return (base_rate + (distance * per_km_rate)) * rush_hour_multiplier


def format_prediction_output(predicted_surge: float, base_price: float) -> Dict[str, float]:
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


def check_data_files() -> Dict[str, bool]:
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


def save_empty_weather_template() -> str:
    """
    Create and save an empty weather data template file.
    
    Returns:
        Path to the created file
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a template DataFrame with required columns
    now = datetime.now()
    dates = [now - timedelta(hours=i) for i in range(24)]
    
    template_df = pd.DataFrame({
        'timestamp': dates,
        'temperature': [20.0] * 24,
        'humidity': [50.0] * 24,
        'wind_speed': [10.0] * 24,
        'precipitation_type': ['clear'] * 24
    })
    
    # Save to CSV
    path = os.path.join(data_dir, 'weather_template.csv')
    template_df.to_csv(path, index=False)
    
    return path


if __name__ == "__main__":
    # Test utility functions
    print("Checking data files:")
    print(check_data_files())
    
    print("\nChecking model file:")
    print(check_model_file())
    
    print("\nCalculating sample price:")
    base_price = calculate_base_price(5.0, True)
    predicted_surge = 1.8
    output = format_prediction_output(predicted_surge, base_price)
    print(f"Base Price: ${output['base_price']}")
    print(f"Surge Multiplier: {output['surge_multiplier']}x")
    print(f"Total Price: ${output['total_price']}")
    
    print("\nCreating weather template:")
    template_path = save_empty_weather_template()
    print(f"Template saved to: {template_path}") 