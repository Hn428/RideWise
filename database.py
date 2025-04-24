import sqlite3
import pandas as pd
import json
import datetime
import os

# Database file name
DB_FILE = "rideshare_data.db"

def dict_to_json(d):
    """Convert a dictionary to a JSON string for storage"""
    if isinstance(d, dict) or isinstance(d, list):
        return json.dumps(d)
    return d

def json_to_dict(j):
    """Convert a JSON string back to a dictionary"""
    if isinstance(j, str) and (j.startswith('{') or j.startswith('[')):
        try:
            return json.loads(j)
        except:
            return j
    return j

def init_db():
    """
    Initialize the database and create tables if they don't exist
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create table for ride data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ride_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        city TEXT,
        weather TEXT,
        events TEXT,
        traffic TEXT,
        rideshare TEXT
    )
    ''')
    
    # Create table for model data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_data BLOB,
        timestamp TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def store_data(timestamp, city, weather_data, event_data, traffic_data, rideshare_data):
    """
    Store collected data in the database
    
    Args:
        timestamp (datetime): Time the data was collected
        city (str): City name
        weather_data (dict): Weather information
        event_data (list): Event information
        traffic_data (dict): Traffic information
        rideshare_data (dict): Ride-sharing prices and info
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Convert data to JSON for storage
    weather_json = dict_to_json(weather_data)
    events_json = dict_to_json(event_data)
    traffic_json = dict_to_json(traffic_data)
    rideshare_json = dict_to_json(rideshare_data)
    
    # Store the data
    cursor.execute('''
    INSERT INTO ride_data (timestamp, city, weather, events, traffic, rideshare)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp.isoformat(), city, weather_json, events_json, traffic_json, rideshare_json))
    
    conn.commit()
    conn.close()

def get_historical_data(city=None, start_date=None, end_date=None):
    """
    Retrieve historical data from the database
    
    Args:
        city (str, optional): City to filter by
        start_date (datetime, optional): Start date for data range
        end_date (datetime, optional): End date for data range
        
    Returns:
        list: List of dictionaries containing the historical data
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Build the query based on filters
    query = "SELECT * FROM ride_data"
    params = []
    
    conditions = []
    if city:
        conditions.append("city = ?")
        params.append(city)
    
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date.isoformat())
    
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date.isoformat())
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    # Order by timestamp
    query += " ORDER BY timestamp DESC"
    
    # Execute query
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert rows to dictionaries
    data = []
    for row in rows:
        try:
            item = {
                "id": row[0],
                "timestamp": datetime.datetime.fromisoformat(row[1]),
                "city": row[2],
                "weather": json_to_dict(row[3]),
                "events": json_to_dict(row[4]),
                "traffic": json_to_dict(row[5]),
                "rideshare": json_to_dict(row[6])
            }
            data.append(item)
        except Exception as e:
            print(f"Error processing row: {e}")
    
    conn.close()
    return data

def store_model(model_name, model_data):
    """
    Store a trained model in the database
    
    Args:
        model_name (str): Name of the model
        model_data (bytes): Serialized model data
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if model already exists
    cursor.execute("SELECT id FROM models WHERE model_name = ?", (model_name,))
    existing = cursor.fetchone()
    
    timestamp = datetime.datetime.now().isoformat()
    
    if existing:
        # Update existing model
        cursor.execute(
            "UPDATE models SET model_data = ?, timestamp = ? WHERE model_name = ?",
            (model_data, timestamp, model_name)
        )
    else:
        # Insert new model
        cursor.execute(
            "INSERT INTO models (model_name, model_data, timestamp) VALUES (?, ?, ?)",
            (model_name, model_data, timestamp)
        )
    
    conn.commit()
    conn.close()

def get_model(model_name):
    """
    Retrieve a model from the database
    
    Args:
        model_name (str): Name of the model to retrieve
        
    Returns:
        bytes: Serialized model data or None if not found
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT model_data FROM models WHERE model_name = ?", (model_name,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return result[0]
    return None

def get_data_as_dataframe(city=None, start_date=None, end_date=None):
    """
    Get historical data as a pandas DataFrame
    
    Args:
        city (str, optional): City to filter by
        start_date (datetime, optional): Start date for data range
        end_date (datetime, optional): End date for data range
        
    Returns:
        DataFrame: Historical data in a pandas DataFrame
    """
    raw_data = get_historical_data(city, start_date, end_date)
    
    if not raw_data:
        return pd.DataFrame()
    
    # Prepare lists for each column
    timestamps = []
    cities = []
    
    # Weather features
    temperatures = []
    weather_descriptions = []
    
    # Traffic features
    congestion_levels = []
    current_speeds = []
    
    # Event features
    event_counts = []
    
    # Rideshare features
    uber_prices = []
    uber_surges = []
    lyft_prices = []
    lyft_surges = []
    
    # Time features
    hours = []
    days_of_week = []
    is_weekend = []
    is_rush_hour = []
    
    for item in raw_data:
        timestamp = item["timestamp"]
        
        # Basic info
        timestamps.append(timestamp)
        cities.append(item["city"])
        
        # Extract weather data
        weather = item.get("weather", {})
        temperatures.append(weather.get("temperature", None))
        weather_descriptions.append(weather.get("description", None))
        
        # Extract traffic data
        traffic = item.get("traffic", {})
        congestion_levels.append(traffic.get("congestion_value", None))
        current_speeds.append(traffic.get("current_speed", None))
        
        # Extract event data
        events = item.get("events", [])
        event_counts.append(len(events))
        
        # Extract rideshare data
        rideshare = item.get("rideshare", {})
        uber = rideshare.get("uber", {})
        lyft = rideshare.get("lyft", {})
        
        uber_prices.append(uber.get("price", None))
        uber_surges.append(uber.get("surge_multiplier", None))
        lyft_prices.append(lyft.get("price", None))
        lyft_surges.append(lyft.get("surge_multiplier", None))
        
        # Calculate time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        hours.append(hour)
        days_of_week.append(day_of_week)
        is_weekend.append(1 if day_of_week >= 5 else 0)
        is_rush_hour.append(1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "city": cities,
        "temperature": temperatures,
        "weather_description": weather_descriptions,
        "congestion_level": congestion_levels,
        "current_speed": current_speeds,
        "event_count": event_counts,
        "uber_price": uber_prices,
        "uber_surge": uber_surges,
        "lyft_price": lyft_prices,
        "lyft_surge": lyft_surges,
        "hour": hours,
        "day_of_week": days_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour
    })
    
    return df
