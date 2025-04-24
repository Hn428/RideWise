import requests
import datetime
import pandas as pd
import numpy as np
import os
import json
import time
import random

def fetch_weather_data(coordinates):
    """
    Fetch weather data for the given coordinates
    
    Args:
        coordinates (tuple): (latitude, longitude) of the location
        
    Returns:
        dict: Weather data including temperature, description, etc.
    """
    try:
        lat, lon = coordinates
        # In a real implementation, this would use OpenWeatherMap API
        # For now, simulate response
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
        if api_key:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "description": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "timestamp": datetime.datetime.now()
                }
            
        # If API key not available or request failed, return basic weather information
        # based on current date/time - this is just for demo purposes
        current_month = datetime.datetime.now().month
        if current_month in [12, 1, 2]:  # Winter
            temp = np.random.normal(3, 5)
            conditions = ["Clear", "Cloudy", "Snow", "Light snow", "Heavy snow", "Freezing rain"]
        elif current_month in [3, 4, 5]:  # Spring
            temp = np.random.normal(15, 7)
            conditions = ["Clear", "Partly cloudy", "Cloudy", "Light rain", "Rain", "Thunderstorm"]
        elif current_month in [6, 7, 8]:  # Summer
            temp = np.random.normal(25, 5)
            conditions = ["Clear", "Partly cloudy", "Cloudy", "Light rain", "Heavy rain", "Thunderstorm"]
        else:  # Fall
            temp = np.random.normal(15, 7)
            conditions = ["Clear", "Partly cloudy", "Cloudy", "Light rain", "Rain"]
            
        description = np.random.choice(conditions)
        return {
            "temperature": round(temp, 1),
            "feels_like": round(temp - 2 if description in ["Rain", "Snow"] else temp, 1),
            "description": description,
            "humidity": np.random.randint(30, 90),
            "wind_speed": round(np.random.uniform(0, 10), 1),
            "timestamp": datetime.datetime.now()
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {}

def fetch_event_data(coordinates):
    """
    Fetch events data for the given coordinates
    
    Args:
        coordinates (tuple): (latitude, longitude) of the location
        
    Returns:
        list: List of events happening near the location
    """
    try:
        lat, lon = coordinates
        # In a real implementation, this would use Ticketmaster API
        # For now, simulate response
        api_key = os.getenv("TICKETMASTER_API_KEY", "")
        
        if api_key:
            url = f"https://app.ticketmaster.com/discovery/v2/events.json?latlong={lat},{lon}&radius=10&apikey={api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                events = []
                if "_embedded" in data and "events" in data["_embedded"]:
                    for event in data["_embedded"]["events"]:
                        events.append({
                            "name": event["name"],
                            "venue": event["_embedded"]["venues"][0]["name"],
                            "date": event["dates"]["start"]["dateTime"],
                            "type": event["type"]
                        })
                return events
        
        # If API key not available or request failed, generate synthetic data
        event_count = np.random.randint(0, 5)  # 0 to 4 events
        events = []
        
        # Get current time to determine what types of events are likely
        current_hour = datetime.datetime.now().hour
        current_day = datetime.datetime.now().weekday()
        
        # More events on evenings and weekends
        if current_hour >= 18 or current_day >= 5:  # Evening or weekend
            event_count += np.random.randint(1, 3)
        
        event_types = ["Concert", "Sports", "Theater", "Conference", "Festival"]
        venues = ["Madison Square Garden", "Barclays Center", "MetLife Stadium", 
                 "Radio City Music Hall", "Central Park", "Convention Center"]
        
        for _ in range(event_count):
            event_type = np.random.choice(event_types)
            venue = np.random.choice(venues)
            
            # Create a reasonable event name based on type
            if event_type == "Concert":
                artists = ["The Weekend", "Taylor Swift", "BTS", "Drake", "Beyoncé", "Ed Sheeran"]
                name = f"{np.random.choice(artists)} Concert"
            elif event_type == "Sports":
                teams = ["Lakers", "Yankees", "Knicks", "Giants", "Patriots", "Bulls"]
                name = f"{np.random.choice(teams)} vs {np.random.choice(teams)} Game"
            elif event_type == "Theater":
                shows = ["Hamilton", "The Lion King", "Wicked", "Chicago", "The Phantom of the Opera"]
                name = f"{np.random.choice(shows)}"
            elif event_type == "Conference":
                conferences = ["Tech Summit", "Business Conference", "Medical Convention", "Developer Conference"]
                name = f"{np.random.choice(conferences)}"
            else:  # Festival
                festivals = ["Food Festival", "Music Festival", "Film Festival", "Art Fair"]
                name = f"{np.random.choice(festivals)}"
            
            # Calculate a random start time between now and 6 hours from now
            hours_ahead = np.random.randint(0, 6)
            start_time = datetime.datetime.now() + datetime.timedelta(hours=hours_ahead)
            
            events.append({
                "name": name,
                "venue": venue,
                "date": start_time.isoformat(),
                "type": event_type
            })
            
        return events
    except Exception as e:
        print(f"Error fetching event data: {e}")
        return []

def fetch_traffic_data(coordinates):
    """
    Fetch traffic data for the given coordinates
    
    Args:
        coordinates (tuple): (latitude, longitude) of the location
        
    Returns:
        dict: Traffic data including congestion levels, etc.
    """
    try:
        lat, lon = coordinates
        # In a real implementation, this would use Google Maps or TomTom API
        # For now, simulate response
        api_key = os.getenv("TOMTOM_API_KEY", "")
        
        if api_key:
            url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Extract the traffic data
                return {
                    "congestion_level": data.get("flowSegmentData", {}).get("currentSpeed", 0),
                    "free_flow_speed": data.get("flowSegmentData", {}).get("freeFlowSpeed", 0),
                    "current_speed": data.get("flowSegmentData", {}).get("currentSpeed", 0),
                    "timestamp": datetime.datetime.now()
                }
        
        # If API key not available or request failed, generate synthetic data
        # Base traffic conditions on time of day
        current_hour = datetime.datetime.now().hour
        current_day = datetime.datetime.now().weekday()
        
        # Define rush hours
        morning_rush = 7 <= current_hour <= 9
        evening_rush = 16 <= current_hour <= 19
        
        # Weekend vs weekday
        is_weekend = current_day >= 5  # 5 = Saturday, 6 = Sunday
        
        # Determine congestion level
        if is_weekend:
            if 10 <= current_hour <= 18:  # Midday weekend
                congestion_options = ["Low", "Medium", "Medium", "High"]
            else:
                congestion_options = ["Very Low", "Low", "Low", "Medium"]
        else:  # Weekday
            if morning_rush or evening_rush:
                congestion_options = ["Medium", "High", "High", "Very High"]
            elif 10 <= current_hour <= 15:  # Midday
                congestion_options = ["Low", "Medium", "Medium", "High"]
            else:
                congestion_options = ["Very Low", "Low", "Medium"]
                
        congestion_level = np.random.choice(congestion_options)
        
        # Translate text congestion to numeric values
        congestion_map = {
            "Very Low": 0.2,
            "Low": 0.4,
            "Medium": 0.6,
            "High": 0.8,
            "Very High": 0.95
        }
        
        congestion_value = congestion_map[congestion_level]
        free_flow_speed = np.random.randint(45, 65)  # mph
        current_speed = max(5, int(free_flow_speed * (1 - congestion_value)))
        
        return {
            "congestion_level": congestion_level,
            "congestion_value": congestion_value,
            "free_flow_speed": free_flow_speed,
            "current_speed": current_speed,
            "timestamp": datetime.datetime.now()
        }
    except Exception as e:
        print(f"Error fetching traffic data: {e}")
        return {}

def fetch_rideshare_data(coordinates, origin, destination):
    """
    Fetch ride-sharing data (Uber/Lyft) for the given coordinates and route
    
    Args:
        coordinates (tuple): (latitude, longitude) of the location
        origin (str): Starting point of the ride
        destination (str): Ending point of the ride
        
    Returns:
        dict: Ride-sharing data including prices, surge multipliers, etc.
    """
    try:
        lat, lon = coordinates
        
        # In a real implementation, this would use Uber and Lyft APIs if available
        # For now, simulate response
        
        # Base price calculation
        distance_miles = np.random.uniform(2, 15)  # Random distance between 2 and 15 miles
        
        # Factors affecting price
        current_hour = datetime.datetime.now().hour
        current_day = datetime.datetime.now().weekday()
        
        # Define baseline prices
        uber_base_price = 2.5 + (distance_miles * 1.75)
        lyft_base_price = 2.0 + (distance_miles * 1.70)
        
        # Time factors
        morning_rush = 7 <= current_hour <= 9
        evening_rush = 16 <= current_hour <= 19
        late_night = current_hour >= 22 or current_hour <= 4
        
        # Weekend vs weekday
        is_weekend = current_day >= 5  # 5 = Saturday, 6 = Sunday
        
        # Calculate surge multipliers
        uber_surge = 1.0
        lyft_surge = 1.0
        
        if morning_rush and not is_weekend:
            uber_surge *= np.random.uniform(1.3, 1.8)
            lyft_surge *= np.random.uniform(1.2, 1.7)
        
        if evening_rush and not is_weekend:
            uber_surge *= np.random.uniform(1.4, 2.0)
            lyft_surge *= np.random.uniform(1.3, 1.9)
        
        if late_night:
            uber_surge *= np.random.uniform(1.2, 1.6)
            lyft_surge *= np.random.uniform(1.1, 1.5)
        
        if is_weekend and 20 <= current_hour <= 23:  # Weekend nights
            uber_surge *= np.random.uniform(1.3, 1.9)
            lyft_surge *= np.random.uniform(1.2, 1.8)
            
        # Adjust for randomness (to simulate real-world fluctuations)
        uber_surge *= np.random.uniform(0.9, 1.1)
        lyft_surge *= np.random.uniform(0.9, 1.1)
        
        # Calculate final prices
        uber_price = uber_base_price * uber_surge
        lyft_price = lyft_base_price * lyft_surge
        
        # Format data for return
        result = {
            "uber": {
                "price": round(uber_price, 2),
                "base_price": round(uber_base_price, 2),
                "surge_multiplier": round(uber_surge, 2),
                "estimated_distance": round(distance_miles, 1),
                "estimated_duration": int(distance_miles * 3 + 5),  # approx. minutes
                "timestamp": datetime.datetime.now()
            },
            "lyft": {
                "price": round(lyft_price, 2),
                "base_price": round(lyft_base_price, 2),
                "surge_multiplier": round(lyft_surge, 2),
                "estimated_distance": round(distance_miles, 1),
                "estimated_duration": int(distance_miles * 3 + 4),  # approx. minutes
                "timestamp": datetime.datetime.now()
            }
        }
        
        # Generate surge zones around the city
        # These would be areas with different surge levels
        surge_zones = []
        
        # Generate 5-10 zones around the city center
        num_zones = np.random.randint(5, 11)
        for _ in range(num_zones):
            # Random offset from city center
            lat_offset = np.random.uniform(-0.05, 0.05)
            lon_offset = np.random.uniform(-0.05, 0.05)
            
            # Calculate surge level for this zone
            zone_surge = np.random.uniform(0.8, max(2.5, uber_surge * 1.5, lyft_surge * 1.5))
            
            surge_zones.append({
                "lat": lat + lat_offset,
                "lon": lon + lon_offset,
                "surge_level": round(zone_surge, 2)
            })
            
        result["surge_zones"] = surge_zones
        
        return result
    except Exception as e:
        print(f"Error fetching rideshare data: {e}")
        return {}
