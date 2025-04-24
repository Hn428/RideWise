import datetime
import numpy as np

def get_city_coordinates(city_name):
    """
    Get the latitude and longitude coordinates for a given city
    
    Args:
        city_name (str): Name of the city
        
    Returns:
        tuple: (latitude, longitude)
    """
    # Dictionary of common city coordinates
    city_coordinates = {
        "New York": (40.7128, -74.0060),
        "San Francisco": (37.7749, -122.4194),
        "Chicago": (41.8781, -87.6298),
        "Los Angeles": (34.0522, -118.2437),
        "Boston": (42.3601, -71.0589),
        "Washington DC": (38.9072, -77.0369),
        "Seattle": (47.6062, -122.3321),
        "Miami": (25.7617, -80.1918),
        "Austin": (30.2672, -97.7431),
        "Denver": (39.7392, -104.9903),
        # Add more cities as needed
    }
    
    # Return coordinates if city exists, otherwise return New York as default
    return city_coordinates.get(city_name, city_coordinates["New York"])

def format_price(price):
    """
    Format a price with 2 decimal places
    
    Args:
        price (float): Price to format
        
    Returns:
        str: Formatted price
    """
    try:
        return "{:.2f}".format(float(price))
    except (ValueError, TypeError):
        return "0.00"

def format_time(dt):
    """
    Format a datetime object into a readable string
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        str: Formatted time string
    """
    if not isinstance(dt, datetime.datetime):
        return "Unknown"
    
    # Format: Today/Tomorrow at 2:30 PM
    now = datetime.datetime.now()
    today = now.date()
    tomorrow = today + datetime.timedelta(days=1)
    
    if dt.date() == today:
        day_str = "Today"
    elif dt.date() == tomorrow:
        day_str = "Tomorrow"
    else:
        # If more than a day ahead, use the day name
        day_str = dt.strftime("%A")
    
    time_str = dt.strftime("%I:%M %p").lstrip("0")
    
    return f"{day_str} at {time_str}"

def is_rush_hour(dt=None):
    """
    Check if the given datetime is during rush hour
    
    Args:
        dt (datetime, optional): Datetime to check. Defaults to current time.
        
    Returns:
        bool: True if it's rush hour, False otherwise
    """
    if dt is None:
        dt = datetime.datetime.now()
    
    hour = dt.hour
    weekday = dt.weekday()
    
    # Rush hours: 7-9 AM and 4-7 PM on weekdays
    morning_rush = 7 <= hour <= 9
    evening_rush = 16 <= hour <= 19
    is_weekday = weekday < 5  # 0-4 are Monday to Friday
    
    return is_weekday and (morning_rush or evening_rush)
