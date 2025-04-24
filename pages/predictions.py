import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from model import predict_best_times
from utils import format_price, format_time

# Set page title
st.set_page_config(
    page_title="Predictions - RideWise",
    page_icon="🚗",
    layout="wide"
)

st.title("🔮 Ride Price Predictions")
st.markdown("### Plan ahead to avoid surge pricing")

# Check if location exists in session state
if 'location' not in st.session_state:
    st.session_state.location = "New York"
if 'coordinates' not in st.session_state:
    from utils import get_city_coordinates
    st.session_state.coordinates = get_city_coordinates("New York")

# Sidebar for input parameters
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1694878982063-9c41c1d94f4c", width=250)
    st.header("Trip Parameters")
    
    # Date selection (today or tomorrow)
    date_options = ["Today", "Tomorrow", "Next 7 days"]
    selected_date = st.selectbox("When are you planning to ride?", date_options)
    
    # Route selection
    origin_options = ["Home", "Work", "Airport", "Downtown", "Custom"]
    destination_options = ["Home", "Work", "Airport", "Downtown", "Custom"]
    
    origin = st.selectbox("Origin", origin_options)
    if origin == "Custom":
        origin = st.text_input("Enter origin address")
    
    destination = st.selectbox("Destination", destination_options)
    if destination == "Custom":
        destination = st.text_input("Enter destination address")
    
    # Prevent same origin and destination
    if origin == destination and origin != "Custom" and destination != "Custom":
        st.warning("Origin and destination cannot be the same")
    
    # Generate predictions button
    if st.button("Generate Predictions"):
        with st.spinner("Analyzing price patterns..."):
            # Get prediction data
            st.session_state.prediction_data = predict_best_times(
                st.session_state.location, origin, destination)
            
            # Add additional info for the selected date range
            if selected_date == "Today":
                # Only use predictions for today
                now = datetime.datetime.now()
                if "hourly_predictions" in st.session_state.prediction_data:
                    st.session_state.prediction_data["hourly_predictions"] = [
                        p for p in st.session_state.prediction_data["hourly_predictions"] 
                        if p["datetime"].date() == now.date()
                    ]
            elif selected_date == "Tomorrow":
                # Only use predictions for tomorrow
                tomorrow = datetime.datetime.now().date() + datetime.timedelta(days=1)
                if "hourly_predictions" in st.session_state.prediction_data:
                    st.session_state.prediction_data["hourly_predictions"] = [
                        p for p in st.session_state.prediction_data["hourly_predictions"] 
                        if p["datetime"].date() == tomorrow
                    ]

# Main content
if 'prediction_data' in st.session_state and st.session_state.prediction_data:
    pred_data = st.session_state.prediction_data
    
    # Display prediction summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ride Optimization Summary")
        
        # Best times summary
        best_times_data = []
        
        if "uber" in pred_data and "best_time" in pred_data["uber"]:
            best_times_data.append({
                "Service": "Uber",
                "Best Time": format_time(pred_data["uber"]["best_time"]),
                "Expected Price": f"${format_price(pred_data['uber']['lowest_price'])}",
            })
            
        if "lyft" in pred_data and "best_time" in pred_data["lyft"]:
            best_times_data.append({
                "Service": "Lyft",
                "Best Time": format_time(pred_data["lyft"]["best_time"]),
                "Expected Price": f"${format_price(pred_data['lyft']['lowest_price'])}",
            })
        
        if best_times_data:
            st.table(pd.DataFrame(best_times_data))
        else:
            st.info("No optimal time predictions available for the selected date.")
        
        # Route summary
        st.subheader("Selected Route")
        st.markdown(f"**From:** {origin}")
        st.markdown(f"**To:** {destination}")
        st.markdown(f"**City:** {st.session_state.location}")
        
        # Factors affecting prices
        st.subheader("Factors Affecting Prices")
        
        # Determine factors based on predictions
        factors = []
        
        # Time-based factors
        now = datetime.datetime.now()
        if 6 <= now.hour <= 9:
            factors.append("Morning Rush Hour")
        elif 16 <= now.hour <= 19:
            factors.append("Evening Rush Hour")
            
        if now.weekday() >= 5:  # Weekend
            factors.append("Weekend Travel")
            
        # Weather impact (placeholder - in a real app, this would be based on forecast)
        factors.append("Normal Weather Conditions")
        
        # List factors
        for factor in factors:
            st.markdown(f"- {factor}")
            
    with col2:
        st.subheader("Price Forecast")
        
        # Price forecast chart
        if "hourly_predictions" in pred_data and pred_data["hourly_predictions"]:
            hourly_data = pd.DataFrame(pred_data["hourly_predictions"])
            
            # Format the datetime for display
            hourly_data["time_label"] = hourly_data["datetime"].apply(
                lambda x: x.strftime("%I %p").lstrip("0")
            )
            
            # Create price chart
            fig = px.line(hourly_data, x="time_label", y=["uber_price", "lyft_price"], 
                          labels={
                              "value": "Estimated Price ($)", 
                              "time_label": "Time", 
                              "variable": "Service"
                          },
                          title="Price Forecast",
                          color_discrete_map={"uber_price": "#276EF1", "lyft_price": "#FF00BF"})
            
            # Add current time marker
            current_hour = datetime.datetime.now().hour
            current_hour_label = datetime.datetime.now().strftime("%I %p").lstrip("0")
            
            # Only add the marker if we're showing today's forecast
            if selected_date == "Today" and current_hour_label in hourly_data["time_label"].values:
                fig.add_vline(x=current_hour_label, line_dash="dash", line_color="green", 
                              annotation_text="Now")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find best booking windows
            st.subheader("Best Booking Windows")
            
            # Calculate average price
            hourly_data["avg_price"] = (hourly_data["uber_price"] + hourly_data["lyft_price"]) / 2
            
            # Find periods with prices below average
            hourly_data["is_good_time"] = hourly_data["avg_price"] < hourly_data["avg_price"].mean()
            
            # Group consecutive good times
            good_windows = []
            current_window = None
            
            for _, row in hourly_data.iterrows():
                if row["is_good_time"]:
                    if current_window is None:
                        current_window = {
                            "start": row["datetime"],
                            "end": row["datetime"],
                            "avg_price": row["avg_price"]
                        }
                    else:
                        current_window["end"] = row["datetime"]
                        current_window["avg_price"] = (current_window["avg_price"] + row["avg_price"]) / 2
                else:
                    if current_window is not None:
                        good_windows.append(current_window)
                        current_window = None
            
            if current_window is not None:
                good_windows.append(current_window)
            
            # Display the windows
            if good_windows:
                window_data = []
                for i, window in enumerate(good_windows):
                    window_data.append({
                        "Window": f"Window {i+1}",
                        "Time Range": f"{format_time(window['start'])} - {window['end'].strftime('%I:%M %p').lstrip('0')}",
                        "Avg. Price": f"${format_price(window['avg_price'])}"
                    })
                
                st.table(pd.DataFrame(window_data))
            else:
                st.info("No significantly cheaper booking windows found.")
        
        else:
            st.info("No hourly predictions available. Please generate predictions first.")

    # Additional insights
    st.subheader("Price Comparison by Hour")
    
    if "hourly_predictions" in pred_data and pred_data["hourly_predictions"]:
        hourly_data = pd.DataFrame(pred_data["hourly_predictions"])
        
        # Reshape for grouped bar chart
        hour_comparison = hourly_data[["hour", "uber_price", "lyft_price"]].copy()
        hour_comparison["hour_label"] = hour_comparison["hour"].apply(
            lambda x: f"{x}:00" if x < 12 else f"{x-12 if x > 12 else x}:00 {'AM' if x < 12 else 'PM'}"
        )
        
        fig = px.bar(hour_comparison, x="hour_label", y=["uber_price", "lyft_price"],
                    labels={
                        "value": "Price ($)",
                        "hour_label": "Hour",
                        "variable": "Service"
                    },
                    title="Price Comparison by Hour",
                    barmode="group",
                    color_discrete_map={"uber_price": "#276EF1", "lyft_price": "#FF00BF"})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price difference chart
        st.subheader("Uber vs. Lyft Price Difference")
        
        hourly_data["price_diff"] = hourly_data["uber_price"] - hourly_data["lyft_price"]
        hourly_data["cheaper_service"] = hourly_data["price_diff"].apply(
            lambda x: "Lyft cheaper" if x > 0 else "Uber cheaper" if x < 0 else "Same price"
        )
        
        fig = px.bar(hourly_data, x="time_label", y="price_diff",
                    labels={
                        "price_diff": "Price Difference ($)",
                        "time_label": "Time"
                    },
                    title="Price Difference (Uber - Lyft)",
                    color="cheaper_service",
                    color_discrete_map={
                        "Uber cheaper": "#276EF1",
                        "Lyft cheaper": "#FF00BF",
                        "Same price": "#888888"
                    })
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Generate predictions to see detailed price comparison.")

else:
    # Show placeholder content if no predictions are available
    st.info("Click 'Generate Predictions' to see price forecasts for your route.")
    
    # Display instruction image
    st.image("https://images.unsplash.com/photo-1694878981905-b742a32f8121", 
            caption="Generate predictions to find the best time to book your ride")
    
    # Provide tips
    st.subheader("Tips for Avoiding Surge Pricing")
    st.markdown("""
    - **Avoid rush hours** - Early mornings (7-9 AM) and evenings (4-7 PM) typically have higher prices
    - **Check prices before events** - Concerts, sports games, and other large events often trigger surge pricing
    - **Wait out short-term surges** - Sometimes waiting just 15-20 minutes can result in lower prices
    - **Compare both services** - Uber and Lyft often have different surge patterns
    """)
