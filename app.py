import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import datetime
import time
from data_collection import fetch_weather_data, fetch_event_data, fetch_traffic_data, fetch_rideshare_data
from database import init_db, store_data, get_historical_data
from model import train_models, predict_surge, predict_best_times
from utils import get_city_coordinates, format_price, format_time

# Set page configuration
st.set_page_config(
    page_title="RideWise - Avoid Surge Pricing",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the database on first run
init_db()

# Initialize session state for storing data across reruns
if 'location' not in st.session_state:
    st.session_state.location = "New York"
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = get_city_coordinates("New York")
if 'current_fare' not in st.session_state:
    st.session_state.current_fare = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# Header with title and description
st.title("🚗 RideWise: Beat the Surge")
st.markdown("### Find the optimal time to book your ride and avoid surge pricing")

# Sidebar for inputs
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1474625417279-a1308b1bb4a2", width=250)
    st.header("Trip Settings")
    
    # Location selection
    city = st.selectbox("Select your city", 
                       ["New York", "San Francisco", "Chicago", "Los Angeles", "Boston", 
                        "Washington DC", "Seattle", "Miami", "Austin", "Denver"])
    
    if city != st.session_state.location:
        st.session_state.location = city
        st.session_state.coordinates = get_city_coordinates(city)
        st.rerun()
    
    # Origin and destination
    origin = st.text_input("Origin", "Current Location")
    destination = st.text_input("Destination", "Airport")
    
    # Update button
    if st.button("Update Fare Information"):
        with st.spinner("Fetching current data..."):
            # Fetch real-time data
            weather_data = fetch_weather_data(st.session_state.coordinates)
            event_data = fetch_event_data(st.session_state.coordinates)
            traffic_data = fetch_traffic_data(st.session_state.coordinates)
            rideshare_data = fetch_rideshare_data(st.session_state.coordinates, origin, destination)
            
            # Store data
            current_time = datetime.datetime.now()
            store_data(current_time, st.session_state.location, weather_data, 
                     event_data, traffic_data, rideshare_data)
            
            # Update current fare information
            st.session_state.current_fare = rideshare_data
            
            # Train/update models if needed
            if not st.session_state.models_trained:
                historical_data = get_historical_data(st.session_state.location)
                if len(historical_data) > 0:  # Only train if we have data
                    with st.spinner("Training prediction models..."):
                        train_models(historical_data)
                        st.session_state.models_trained = True
            
            # Get predictions
            if st.session_state.models_trained:
                st.session_state.prediction_data = predict_best_times(
                    st.session_state.location, origin, destination)

# Main content area - split into sections
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Current Surge Status")
    
    # Display current fare info if available
    if st.session_state.current_fare:
        current_data = st.session_state.current_fare
        
        # Create metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            uber_price = current_data.get("uber", {}).get("price", 0)
            uber_surge = current_data.get("uber", {}).get("surge_multiplier", 1.0)
            st.metric("Uber", f"${format_price(uber_price)}", 
                     f"{format_price((uber_surge-1)*100)}% surge" if uber_surge > 1 else "No surge")
        
        with metric_col2:
            lyft_price = current_data.get("lyft", {}).get("price", 0)
            lyft_surge = current_data.get("lyft", {}).get("surge_multiplier", 1.0)
            st.metric("Lyft", f"${format_price(lyft_price)}", 
                     f"{format_price((lyft_surge-1)*100)}% surge" if lyft_surge > 1 else "No surge")
        
        with metric_col3:
            # Get the cheaper option
            if uber_price <= lyft_price:
                cheaper = "Uber"
                savings = lyft_price - uber_price
            else:
                cheaper = "Lyft"
                savings = uber_price - lyft_price
            
            st.metric("Best Option Now", cheaper, f"Save ${format_price(savings)}")
    else:
        st.info("Click 'Update Fare Information' to fetch current prices.")
    
    # Show prediction information if available
    if st.session_state.prediction_data:
        st.subheader("Recommended Booking Times")
        
        pred_data = st.session_state.prediction_data
        
        # Create a dataframe for the predicted best times
        times = []
        for service in ["uber", "lyft"]:
            if service in pred_data:
                best_time = pred_data[service].get("best_time")
                lowest_price = pred_data[service].get("lowest_price")
                current_price = st.session_state.current_fare.get(service, {}).get("price", 0) if st.session_state.current_fare else 0
                
                if best_time and lowest_price:
                    savings = current_price - lowest_price
                    times.append({
                        "Service": service.capitalize(),
                        "Best Time": format_time(best_time),
                        "Expected Price": f"${format_price(lowest_price)}",
                        "Potential Savings": f"${format_price(savings)}",
                        "Wait Time": f"{int((best_time - datetime.datetime.now()).total_seconds() / 60)} min"
                    })
        
        if times:
            times_df = pd.DataFrame(times)
            st.table(times_df)
            
            # Add a visual chart for predicted pricing throughout the day
            if "hourly_predictions" in pred_data:
                st.subheader("Price Forecast (Next 24 Hours)")
                
                hourly_data = pred_data["hourly_predictions"]
                fig = px.line(hourly_data, x="hour", y=["uber_price", "lyft_price"], 
                             labels={"value": "Estimated Price ($)", "hour": "Hour", "variable": "Service"},
                             title="Estimated Prices Over Next 24 Hours",
                             color_discrete_map={"uber_price": "#276EF1", "lyft_price": "#FF00BF"})
                
                # Add a vertical line for current time
                current_hour = datetime.datetime.now().hour
                fig.add_vline(x=current_hour, line_dash="dash", line_color="green", annotation_text="Now")
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No optimal booking times predicted yet. Check back later when more data is available.")
    else:
        st.info("Prediction models are being trained. Please check back after updating fare information.")

with col2:
    st.subheader("Factors Affecting Prices")
    
    # Show data about current conditions
    if st.session_state.current_fare:
        # Weather information
        weather = st.session_state.current_fare.get("weather", {})
        if weather:
            st.markdown(f"**Current Weather:** {weather.get('description', 'Unknown')}, {weather.get('temperature', 'N/A')}°C")
        
        # Traffic information
        traffic = st.session_state.current_fare.get("traffic", {})
        if traffic:
            traffic_level = traffic.get("congestion_level", "Unknown")
            st.markdown(f"**Traffic Conditions:** {traffic_level} congestion")
        
        # Nearby events
        events = st.session_state.current_fare.get("events", [])
        if events:
            st.markdown("**Nearby Events:**")
            for event in events[:3]:  # Show top 3 events
                st.markdown(f"- {event.get('name')}: {event.get('venue')}")
        
        # Price drivers
        st.markdown("**Current Price Drivers:**")
        drivers = []
        
        # Simple logic to determine price drivers
        surge_level = max(
            st.session_state.current_fare.get("uber", {}).get("surge_multiplier", 1.0),
            st.session_state.current_fare.get("lyft", {}).get("surge_multiplier", 1.0)
        )
        
        # Time factors
        now = datetime.datetime.now()
        is_rush_hour = (now.hour >= 7 and now.hour <= 9) or (now.hour >= 16 and now.hour <= 19)
        is_weekend = now.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        
        if is_rush_hour:
            drivers.append("Peak Rush Hour")
        if is_weekend:
            drivers.append("Weekend Demand")
        if weather and "rain" in weather.get("description", "").lower():
            drivers.append("Rainy Weather")
        if events and len(events) > 0:
            drivers.append("Nearby Events")
        if traffic and traffic.get("congestion_level") in ["High", "Very High"]:
            drivers.append("Heavy Traffic")
        
        if not drivers:
            drivers.append("Normal conditions")
        
        for driver in drivers:
            st.markdown(f"- {driver}")
        
        # Image related to condition
        if "Heavy Traffic" in drivers:
            st.image("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df", caption="Traffic conditions in your area")
        elif "Rainy Weather" in drivers:
            st.image("https://images.unsplash.com/photo-1515338541898-3f66e1659770", caption="Current weather impact")

    # Small map showing surge zones
    st.subheader("Current Surge Map")
    
    if st.session_state.coordinates:
        # Center coordinates
        lat, lon = st.session_state.coordinates
        
        # Create a map
        m = folium.Map(location=[lat, lon], zoom_start=12)
        
        # If we have surge data, create a heatmap-like visualization
        if st.session_state.current_fare and "surge_zones" in st.session_state.current_fare:
            surge_zones = st.session_state.current_fare["surge_zones"]
            
            # Add circles for each surge zone
            for zone in surge_zones:
                zone_lat = zone.get("lat")
                zone_lon = zone.get("lon")
                surge_level = zone.get("surge_level", 1.0)
                
                # Color based on surge level
                if surge_level < 1.2:
                    color = "green"
                elif surge_level < 1.5:
                    color = "yellow"
                elif surge_level < 2.0:
                    color = "orange"
                else:
                    color = "red"
                
                folium.Circle(
                    location=[zone_lat, zone_lon],
                    radius=500,  # 500m radius
                    color=color,
                    fill=True,
                    fill_opacity=0.4,
                    tooltip=f"Surge level: {surge_level}x"
                ).add_to(m)
        else:
            # If no surge data, just show a marker at the city center
            folium.Marker(
                location=[lat, lon],
                tooltip="City Center"
            ).add_to(m)
        
        # Display the map
        folium_static(m)
    else:
        st.info("Select a location to view surge areas.")

# Footer section
st.markdown("---")
st.markdown("**RideWise** helps you save money by predicting the best times to book ride-sharing services.")
st.info("💡 **Tip:** Prices are typically lowest in mid-morning and mid-afternoon between rush hours.")

# Add app usage info at the bottom
with st.expander("How to use this app"):
    st.markdown("""
    1. **Select your city** from the dropdown menu
    2. **Enter your origin and destination** in the text fields
    3. **Click 'Update Fare Information'** to get current prices and predictions
    4. **Review the current prices** and surge status
    5. **Check the recommended booking times** to save money
    6. **View the surge map** to see areas with high demand
    
    The app learns from historical data to provide better predictions over time.
    """)
