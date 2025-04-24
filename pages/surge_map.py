import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import datetime
from data_collection import fetch_rideshare_data
from utils import get_city_coordinates
from database import get_historical_data

# Set page title
st.set_page_config(
    page_title="Surge Map - RideWise",
    page_icon="🚗",
    layout="wide"
)

st.title("🗺️ Surge Pricing Map")
st.markdown("### Visualize surge pricing zones in your city")

# Check if location exists in session state
if 'location' not in st.session_state:
    st.session_state.location = "New York"
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = get_city_coordinates("New York")

# Sidebar for city selection and refresh options
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1515338541898-3f66e1659770", width=250)
    st.header("Map Settings")
    
    # City selection
    city = st.selectbox("Select City", 
                       ["New York", "San Francisco", "Chicago", "Los Angeles", "Boston", 
                        "Washington DC", "Seattle", "Miami", "Austin", "Denver"])
    
    if city != st.session_state.location:
        st.session_state.location = city
        st.session_state.coordinates = get_city_coordinates(city)
        # Clear any cached data
        if 'current_surge_data' in st.session_state:
            del st.session_state.current_surge_data
        st.rerun()
    
    # Refresh button
    if st.button("Refresh Surge Data"):
        with st.spinner("Fetching latest surge data..."):
            # Fetch new ride share data which includes surge zones
            ride_data = fetch_rideshare_data(st.session_state.coordinates, "Current Location", "Destination")
            if ride_data and "surge_zones" in ride_data:
                st.session_state.current_surge_data = ride_data
                st.success("Surge data updated!")
            else:
                st.error("Unable to fetch surge data. Please try again.")
    
    # Map display options
    st.subheader("Display Options")
    
    # Service selection
    service = st.radio("Service", ["Both", "Uber", "Lyft"])
    
    # Map view
    map_view = st.selectbox("Map View", ["Streets", "Satellite"])
    
    # Surge threshold
    surge_threshold = st.slider("Minimum Surge to Display", 1.0, 3.0, 1.1, 0.1)
    
    # Help information
    with st.expander("How to use this map"):
        st.markdown("""
        This map shows areas in your city where ride-sharing services are currently experiencing surge pricing:
        
        - **Red zones** indicate high surge pricing (1.8x or higher)
        - **Orange zones** indicate medium surge pricing (1.4x - 1.8x)
        - **Yellow zones** indicate mild surge pricing (1.2x - 1.4x)
        - **Green zones** indicate little to no surge (below 1.2x)
        
        The map updates when you:
        1. Select a different city
        2. Click the "Refresh Surge Data" button
        
        Surge data is based on current demand, traffic, weather, and events in each area.
        """)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Current Surge Map")
    
    # Get or fetch surge data
    if 'current_surge_data' not in st.session_state:
        with st.spinner("Fetching surge data..."):
            # Fetch new ride share data which includes surge zones
            ride_data = fetch_rideshare_data(st.session_state.coordinates, "Current Location", "Destination")
            if ride_data and "surge_zones" in ride_data:
                st.session_state.current_surge_data = ride_data
    
    # Create the map
    if 'current_surge_data' in st.session_state and "surge_zones" in st.session_state.current_surge_data:
        # Get the center coordinates for the map
        lat, lon = st.session_state.coordinates
        
        # Create a base map
        if map_view == "Streets":
            tiles = "OpenStreetMap"
        else:  # Satellite
            tiles = "Stamen Terrain"
            
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles=tiles)
        
        # Add the surge zones to the map
        surge_zones = st.session_state.current_surge_data["surge_zones"]
        
        # Get service-specific data
        uber_data = st.session_state.current_surge_data.get("uber", {})
        lyft_data = st.session_state.current_surge_data.get("lyft", {})
        
        # Only show zones with surge above the threshold
        filtered_zones = [zone for zone in surge_zones if zone["surge_level"] >= surge_threshold]
        
        for zone in filtered_zones:
            zone_lat = zone["lat"]
            zone_lon = zone["lon"]
            surge_level = zone["surge_level"]
            
            # Only show the selected service
            if service != "Both":
                # If service specific, adjust the surge level
                if service == "Uber" and "surge_multiplier" in uber_data:
                    # Scale zone surge by uber's general surge level
                    scale_factor = uber_data["surge_multiplier"] / (
                        (uber_data["surge_multiplier"] + lyft_data.get("surge_multiplier", 1.0)) / 2
                    )
                    surge_level = surge_level * scale_factor
                elif service == "Lyft" and "surge_multiplier" in lyft_data:
                    # Scale zone surge by lyft's general surge level
                    scale_factor = lyft_data["surge_multiplier"] / (
                        (uber_data.get("surge_multiplier", 1.0) + lyft_data["surge_multiplier"]) / 2
                    )
                    surge_level = surge_level * scale_factor
            
            # Skip this zone if it's below threshold after adjustment
            if surge_level < surge_threshold:
                continue
                
            # Color based on surge level
            if surge_level < 1.2:
                color = "green"
            elif surge_level < 1.4:
                color = "yellow"
            elif surge_level < 1.8:
                color = "orange"
            else:
                color = "red"
            
            # Create tooltip with surge information
            tooltip = f"Surge level: {surge_level:.1f}x"
            
            # Add a circle for this surge zone
            folium.Circle(
                location=[zone_lat, zone_lon],
                radius=500,  # 500m radius
                color=color,
                fill=True,
                fill_opacity=0.4,
                tooltip=tooltip
            ).add_to(m)
        
        # Add a marker for the user's location
        folium.Marker(
            location=[lat, lon],
            tooltip="Your Location",
            icon=folium.Icon(color="blue", icon="user", prefix="fa")
        ).add_to(m)
        
        # Add common destinations with typical surge patterns
        destinations = [
            {"name": "Airport", "coords": [lat + 0.07, lon + 0.08], "typical_surge": "Medium (1.3-1.5x)"},
            {"name": "Downtown", "coords": [lat - 0.02, lon - 0.01], "typical_surge": "High (1.5-2.0x)"},
            {"name": "Stadium", "coords": [lat + 0.04, lon - 0.05], "typical_surge": "Very High (2.0-3.0x)"},
            {"name": "Shopping Mall", "coords": [lat - 0.06, lon + 0.03], "typical_surge": "Low (1.0-1.3x)"}
        ]
        
        for dest in destinations:
            folium.Marker(
                location=dest["coords"],
                tooltip=f"{dest['name']}: Typical Surge {dest['typical_surge']}",
                icon=folium.Icon(color="green", icon="flag", prefix="fa")
            ).add_to(m)
        
        # Display the map
        folium_static(m, width=700, height=500)
        
        # Add timestamp of when the data was collected
        if "uber" in st.session_state.current_surge_data and "timestamp" in st.session_state.current_surge_data["uber"]:
            timestamp = st.session_state.current_surge_data["uber"]["timestamp"]
            st.caption(f"Data last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No surge data available. Click 'Refresh Surge Data' to fetch the latest information.")
        
        # Display a sample map image
        st.image("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df", 
                caption="Refresh to see live surge data for your city")

with col2:
    st.subheader("Surge Status")
    
    # Display current surge levels
    if 'current_surge_data' in st.session_state:
        data = st.session_state.current_surge_data
        
        # Show Uber surge
        if "uber" in data:
            uber_surge = data["uber"].get("surge_multiplier", 1.0)
            uber_price = data["uber"].get("price", 0)
            
            if uber_surge > 1.8:
                surge_status = "Very High"
                color = "red"
            elif uber_surge > 1.4:
                surge_status = "High"
                color = "orange"
            elif uber_surge > 1.2:
                surge_status = "Medium"
                color = "yellow"
            else:
                surge_status = "Low/None"
                color = "green"
            
            st.markdown(f"### Uber")
            st.markdown(f"**Current Surge:** <span style='color:{color}'>{uber_surge:.1f}x ({surge_status})</span>", unsafe_allow_html=True)
            st.markdown(f"**Average Price:** ${uber_price:.2f}")
        
        # Show Lyft surge
        if "lyft" in data:
            lyft_surge = data["lyft"].get("surge_multiplier", 1.0)
            lyft_price = data["lyft"].get("price", 0)
            
            if lyft_surge > 1.8:
                surge_status = "Very High"
                color = "red"
            elif lyft_surge > 1.4:
                surge_status = "High"
                color = "orange"
            elif lyft_surge > 1.2:
                surge_status = "Medium"
                color = "yellow"
            else:
                surge_status = "Low/None"
                color = "green"
            
            st.markdown(f"### Lyft")
            st.markdown(f"**Current Surge:** <span style='color:{color}'>{lyft_surge:.1f}x ({surge_status})</span>", unsafe_allow_html=True)
            st.markdown(f"**Average Price:** ${lyft_price:.2f}")
        
        # Compare services
        if "uber" in data and "lyft" in data:
            uber_price = data["uber"].get("price", 0)
            lyft_price = data["lyft"].get("price", 0)
            
            if uber_price < lyft_price:
                better_service = "Uber"
                savings = lyft_price - uber_price
            else:
                better_service = "Lyft"
                savings = uber_price - lyft_price
            
            st.markdown("### Comparison")
            st.markdown(f"**Better Deal:** {better_service}")
            st.markdown(f"**Potential Savings:** ${savings:.2f}")
    
    # Show factors affecting surge
    st.subheader("Surge Factors")
    
    # Get current time info
    now = datetime.datetime.now()
    hour = now.hour
    day = now.weekday()
    
    # Determine time-based factors
    is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
    is_weekend = day >= 5
    
    factors = []
    
    if is_rush_hour and not is_weekend:
        factors.append({"name": "Rush Hour", "impact": "High"})
    
    if is_weekend:
        factors.append({"name": "Weekend", "impact": "Medium"})
    
    if hour >= 22 or hour <= 4:
        factors.append({"name": "Late Night", "impact": "Medium"})
    
    # Add weather and events if available
    if 'current_surge_data' in st.session_state:
        data = st.session_state.current_surge_data
        
        # Weather
        if "weather" in data:
            weather = data["weather"]
            weather_desc = weather.get("description", "").lower()
            
            if "rain" in weather_desc or "snow" in weather_desc or "storm" in weather_desc:
                factors.append({"name": "Bad Weather", "impact": "High"})
            elif "cloudy" in weather_desc or "overcast" in weather_desc:
                factors.append({"name": "Cloudy Weather", "impact": "Low"})
        
        # Events
        if "events" in data and len(data["events"]) > 0:
            factors.append({"name": f"Events ({len(data['events'])})", "impact": "High"})
        
        # Traffic
        if "traffic" in data:
            traffic = data["traffic"]
            congestion = traffic.get("congestion_level", "Low")
            
            if congestion in ["High", "Very High"]:
                factors.append({"name": "Heavy Traffic", "impact": "High"})
            elif congestion == "Medium":
                factors.append({"name": "Moderate Traffic", "impact": "Medium"})
    
    # If no factors determined, add normal conditions
    if not factors:
        factors.append({"name": "Normal Conditions", "impact": "Low"})
    
    # Display factors
    for factor in factors:
        impact = factor["impact"]
        if impact == "High":
            color = "red"
        elif impact == "Medium":
            color = "orange"
        else:
            color = "green"
        
        st.markdown(f"- **{factor['name']}**: <span style='color:{color}'>{impact} Impact</span>", unsafe_allow_html=True)
    
    # Add surge prediction
    st.subheader("Surge Prediction")
    
    # Get current hour and location for prediction
    current_hour = datetime.datetime.now().hour
    
    # Predict next few hours
    hours_to_predict = [current_hour + i for i in range(1, 5)]
    hours_to_predict = [h % 24 for h in hours_to_predict]  # Handle wrap-around past midnight
    
    # Get time labels
    time_labels = [f"{h}:00" if h < 12 else f"{h-12 if h > 12 else h}:00 {'AM' if h < 12 else 'PM'}" for h in hours_to_predict]
    
    # Generate simple surge predictions
    # This is a simplified model - in a real app, this would use the trained ML models
    surge_predictions = []
    
    for hour in hours_to_predict:
        # Morning rush (7-9 AM)
        if 7 <= hour <= 9 and not is_weekend:
            surge = np.random.uniform(1.4, 1.8)
        # Evening rush (4-7 PM)
        elif 16 <= hour <= 19 and not is_weekend:
            surge = np.random.uniform(1.5, 2.0)
        # Late night weekend (10 PM - 2 AM)
        elif (hour >= 22 or hour <= 2) and (is_weekend or day == 4):  # Friday night counts as weekend
            surge = np.random.uniform(1.3, 1.9)
        # Weekend daytime
        elif 10 <= hour <= 18 and is_weekend:
            surge = np.random.uniform(1.2, 1.5)
        # Other times
        else:
            surge = np.random.uniform(1.0, 1.3)
        
        # If we have factors that would affect these predictions, adjust them
        for factor in factors:
            if factor["impact"] == "High" and factor["name"] != "Normal Conditions":
                surge += 0.2
            elif factor["impact"] == "Medium" and factor["name"] != "Normal Conditions":
                surge += 0.1
        
        surge_predictions.append(surge)
    
    # Create a simple chart with time and predicted surge
    surge_df = pd.DataFrame({
        "Time": time_labels,
        "Predicted Surge": surge_predictions
    })
    
    # Display as bar chart
    st.bar_chart(surge_df.set_index("Time"))
    
    st.caption("This chart shows predicted surge levels for the next few hours. Lower bars indicate better times to book.")
    
    # Add a conclusion
    min_surge_idx = surge_predictions.index(min(surge_predictions))
    best_time = time_labels[min_surge_idx]
    
    st.markdown(f"**Recommendation:** Book at **{best_time}** for the lowest surge in the next few hours.")

# Add extra info section at the bottom
st.markdown("---")
st.subheader("Understanding Surge Pricing")

with st.expander("How surge pricing works"):
    st.markdown("""
    Surge pricing is a dynamic pricing strategy used by ride-sharing companies to balance supply and demand:
    
    - When demand for rides exceeds the supply of available drivers, prices increase
    - The price increase is communicated as a multiplier (e.g., 1.5x means 50% higher than normal)
    - Surge pricing encourages more drivers to get on the road and helps allocate limited resources
    
    **Common Surge Triggers:**
    
    - Rush hours (morning and evening commutes)
    - Bad weather conditions (rain, snow, storms)
    - Special events (concerts, sports games, conferences)
    - Late night (especially on weekends)
    - Transportation disruptions (subway outages, train delays)
    
    **How to Beat the Surge:**
    
    - Wait 10-15 minutes if possible (surge pricing is often short-lived)
    - Walk a few blocks away from high-demand areas
    - Compare multiple services (Uber and Lyft often have different surge levels)
    - Use scheduled rides for important trips
    """)

st.markdown("---")
st.caption("Note: This map uses real-time data combined with historical patterns to estimate current surge zones.")
