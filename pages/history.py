import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
from database import get_data_as_dataframe

# Set page title
st.set_page_config(
    page_title="Price History - RideWise",
    page_icon="🚗",
    layout="wide"
)

st.title("📊 Ride Price History")
st.markdown("### Track pricing trends and identify patterns")

# Check if location exists in session state
if 'location' not in st.session_state:
    st.session_state.location = "New York"

# Get historical data from database
with st.spinner("Loading historical data..."):
    # Get data for the current city
    historical_df = get_data_as_dataframe(city=st.session_state.location)
    
    if historical_df.empty:
        # If no data for the current city, get data for any city to show sample charts
        historical_df = get_data_as_dataframe()
    
    # If still no data, create sample data
    if historical_df.empty:
        # Create sample data for demonstration
        dates = pd.date_range(end=datetime.datetime.now(), periods=100, freq='H')
        
        historical_df = pd.DataFrame({
            'timestamp': dates,
            'city': st.session_state.location,
            'temperature': np.random.normal(20, 5, size=100),
            'uber_price': np.random.normal(25, 8, size=100),
            'uber_surge': np.random.normal(1.2, 0.3, size=100).clip(1.0, 3.0),
            'lyft_price': np.random.normal(23, 7, size=100),
            'lyft_surge': np.random.normal(1.15, 0.25, size=100).clip(1.0, 3.0),
            'hour': [d.hour for d in dates],
            'day_of_week': [d.weekday() for d in dates],
            'is_weekend': [(1 if d.weekday() >= 5 else 0) for d in dates],
            'is_rush_hour': [(1 if (7 <= d.hour <= 9 or 16 <= d.hour <= 19) else 0) for d in dates]
        })

# Sidebar for filters
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523985231622-ce12bb51849a", width=250)
    st.header("Filter Options")
    
    # Date range filter
    if not historical_df.empty:
        min_date = historical_df['timestamp'].min().date()
        max_date = historical_df['timestamp'].max().date()
        
        # Default to last 7 days if we have enough data
        default_start = max(min_date, max_date - datetime.timedelta(days=7))
        
        start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)
        
        # Filter data by date range
        mask = (historical_df['timestamp'].dt.date >= start_date) & (historical_df['timestamp'].dt.date <= end_date)
        filtered_df = historical_df[mask]
    else:
        filtered_df = historical_df
    
    # Service selection
    services = st.multiselect("Services", ["Uber", "Lyft"], default=["Uber", "Lyft"])
    
    # Time of day filter
    time_of_day = st.multiselect(
        "Time of Day",
        ["Morning (5-9 AM)", "Midday (9 AM-4 PM)", "Evening (4-8 PM)", "Night (8 PM-5 AM)"],
        default=["Morning (5-9 AM)", "Midday (9 AM-4 PM)", "Evening (4-8 PM)", "Night (8 PM-5 AM)"]
    )
    
    # Apply time of day filter
    if filtered_df is not None and len(time_of_day) < 4:  # Only apply if not all options selected
        time_masks = []
        if "Morning (5-9 AM)" in time_of_day:
            time_masks.append((filtered_df['hour'] >= 5) & (filtered_df['hour'] < 9))
        if "Midday (9 AM-4 PM)" in time_of_day:
            time_masks.append((filtered_df['hour'] >= 9) & (filtered_df['hour'] < 16))
        if "Evening (4-8 PM)" in time_of_day:
            time_masks.append((filtered_df['hour'] >= 16) & (filtered_df['hour'] < 20))
        if "Night (8 PM-5 AM)" in time_of_day:
            time_masks.append((filtered_df['hour'] >= 20) | (filtered_df['hour'] < 5))
        
        if time_masks:
            combined_mask = time_masks[0]
            for mask in time_masks[1:]:
                combined_mask = combined_mask | mask
            filtered_df = filtered_df[combined_mask]
    
    # Day type filter
    day_type = st.multiselect(
        "Day Type",
        ["Weekdays", "Weekends"],
        default=["Weekdays", "Weekends"]
    )
    
    # Apply day type filter
    if filtered_df is not None and len(day_type) < 2:  # Only apply if not all options selected
        if "Weekdays" in day_type:
            filtered_df = filtered_df[filtered_df['is_weekend'] == 0]
        elif "Weekends" in day_type:
            filtered_df = filtered_df[filtered_df['is_weekend'] == 1]
    
    # Export data option
    if not filtered_df.empty and st.button("Export Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ridewise_data_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )

# Main content - Historical analysis
if not filtered_df.empty:
    # Price trends over time
    st.subheader("Price Trends Over Time")
    
    # Prepare data for plotting
    plot_data = filtered_df.copy()
    
    # Create time series plot
    fig = go.Figure()
    
    if "Uber" in services:
        fig.add_trace(go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['uber_price'],
            name="Uber",
            line=dict(color="#276EF1", width=2)
        ))
    
    if "Lyft" in services:
        fig.add_trace(go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['lyft_price'],
            name="Lyft",
            line=dict(color="#FF00BF", width=2)
        ))
    
    fig.update_layout(
        title="Price Trends",
        xaxis_title="Date/Time",
        yaxis_title="Price ($)",
        legend_title="Service",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price analysis by time of day and day of week
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prices by Time of Day")
        
        # Group by hour and calculate average prices
        hour_data = filtered_df.groupby('hour').agg({
            'uber_price': 'mean',
            'lyft_price': 'mean',
            'uber_surge': 'mean',
            'lyft_surge': 'mean'
        }).reset_index()
        
        # Create time of day chart
        hour_fig = go.Figure()
        
        if "Uber" in services:
            hour_fig.add_trace(go.Scatter(
                x=hour_data['hour'],
                y=hour_data['uber_price'],
                name="Uber",
                line=dict(color="#276EF1", width=2)
            ))
        
        if "Lyft" in services:
            hour_fig.add_trace(go.Scatter(
                x=hour_data['hour'],
                y=hour_data['lyft_price'],
                name="Lyft",
                line=dict(color="#FF00BF", width=2)
            ))
        
        hour_fig.update_layout(
            title="Average Price by Hour of Day",
            xaxis_title="Hour (24-hour format)",
            yaxis_title="Price ($)",
            legend_title="Service",
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 24, 3)),
                ticktext=[f"{h}:00" for h in range(0, 24, 3)]
            )
        )
        
        st.plotly_chart(hour_fig, use_container_width=True)
    
    with col2:
        st.subheader("Prices by Day of Week")
        
        # Group by day of week and calculate average prices
        day_data = filtered_df.groupby('day_of_week').agg({
            'uber_price': 'mean',
            'lyft_price': 'mean',
            'uber_surge': 'mean',
            'lyft_surge': 'mean'
        }).reset_index()
        
        # Map day of week numbers to names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_data['day_name'] = day_data['day_of_week'].apply(lambda x: day_names[x])
        
        # Sort by day of week (Monday first)
        day_data = day_data.sort_values('day_of_week')
        
        # Create day of week chart
        day_fig = go.Figure()
        
        if "Uber" in services:
            day_fig.add_trace(go.Bar(
                x=day_data['day_name'],
                y=day_data['uber_price'],
                name="Uber",
                marker_color="#276EF1"
            ))
        
        if "Lyft" in services:
            day_fig.add_trace(go.Bar(
                x=day_data['day_name'],
                y=day_data['lyft_price'],
                name="Lyft",
                marker_color="#FF00BF"
            ))
        
        day_fig.update_layout(
            title="Average Price by Day of Week",
            xaxis_title="Day",
            yaxis_title="Price ($)",
            legend_title="Service",
            barmode='group'
        )
        
        st.plotly_chart(day_fig, use_container_width=True)
    
    # Surge analysis
    st.subheader("Surge Pricing Analysis")
    
    # Calculate surge frequency
    uber_surge_freq = (filtered_df['uber_surge'] > 1.0).mean() * 100
    lyft_surge_freq = (filtered_df['lyft_surge'] > 1.0).mean() * 100
    
    # Display surge metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if "Uber" in services:
            st.metric("Uber Surge Frequency", f"{uber_surge_freq:.1f}%")
            st.metric("Avg. Uber Surge Multiplier", f"{filtered_df['uber_surge'].mean():.2f}x")
        else:
            st.info("Select Uber to see metrics")
    
    with metric_col2:
        if "Lyft" in services:
            st.metric("Lyft Surge Frequency", f"{lyft_surge_freq:.1f}%")
            st.metric("Avg. Lyft Surge Multiplier", f"{filtered_df['lyft_surge'].mean():.2f}x")
        else:
            st.info("Select Lyft to see metrics")
    
    with metric_col3:
        # Calculate which service has lower surge on average
        if "Uber" in services and "Lyft" in services:
            uber_avg = filtered_df['uber_surge'].mean()
            lyft_avg = filtered_df['lyft_surge'].mean()
            
            lower_surge = "Uber" if uber_avg < lyft_avg else "Lyft"
            diff = abs(uber_avg - lyft_avg)
            
            st.metric("Lower Average Surge", lower_surge)
            st.metric("Surge Difference", f"{diff:.2f}x")
        else:
            st.info("Select both services to compare")
    
    # Surge heatmap by hour and day
    st.subheader("Surge Heatmap (by hour and day)")
    
    # Create pivot tables for heatmaps
    if "Uber" in services:
        uber_pivot = filtered_df.pivot_table(
            values='uber_surge', 
            index='day_of_week', 
            columns='hour', 
            aggfunc='mean'
        )
        
        # Fill NaN values (if any)
        uber_pivot = uber_pivot.fillna(1.0)
        
        # Add day names as index
        uber_pivot.index = [day_names[i] for i in uber_pivot.index]
        
        # Create heatmap
        fig_uber_heatmap = px.imshow(
            uber_pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="Surge Multiplier"),
            x=[f"{h}:00" for h in uber_pivot.columns],
            y=uber_pivot.index,
            color_continuous_scale="Reds",
            title="Uber Surge by Hour and Day"
        )
        
        st.plotly_chart(fig_uber_heatmap, use_container_width=True)
    
    if "Lyft" in services:
        lyft_pivot = filtered_df.pivot_table(
            values='lyft_surge', 
            index='day_of_week', 
            columns='hour', 
            aggfunc='mean'
        )
        
        # Fill NaN values (if any)
        lyft_pivot = lyft_pivot.fillna(1.0)
        
        # Add day names as index
        lyft_pivot.index = [day_names[i] for i in lyft_pivot.index]
        
        # Create heatmap
        fig_lyft_heatmap = px.imshow(
            lyft_pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="Surge Multiplier"),
            x=[f"{h}:00" for h in lyft_pivot.columns],
            y=lyft_pivot.index,
            color_continuous_scale="Purples",
            title="Lyft Surge by Hour and Day"
        )
        
        st.plotly_chart(fig_lyft_heatmap, use_container_width=True)
    
    # Price correlation with factors
    st.subheader("Price Correlation With Factors")
    
    # Calculate correlations
    correlation_data = filtered_df[['uber_price', 'lyft_price', 'temperature', 
                                   'is_weekend', 'is_rush_hour', 'event_count']].copy()
    
    # Only include columns that exist (some might be missing in the actual data)
    valid_columns = [col for col in correlation_data.columns if col in filtered_df.columns]
    
    if len(valid_columns) > 1:  # Need at least 2 columns for correlation
        correlation_matrix = correlation_data[valid_columns].corr()
        
        # Plot correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            labels=dict(x="Factor", y="Factor", color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale="RdBu_r",
            title="Correlation Between Price and Factors"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Explain the correlations
        st.markdown("### Understanding Price Factors")
        st.markdown("""
        This correlation matrix shows how different factors relate to ride prices:
        
        - **Positive correlation (red)** means as one factor increases, the other tends to increase as well
        - **Negative correlation (blue)** means as one factor increases, the other tends to decrease
        - **Values close to 0** indicate weak or no correlation
        
        For example, a strong positive correlation between `is_rush_hour` and prices would confirm 
        that rush hours typically have higher fares.
        """)
        
else:
    st.info("No historical data available. Start collecting data by using the app more frequently.")
    
    # Show placeholder image
    st.image("https://images.unsplash.com/photo-1473889803946-6a3923603697", 
            caption="Start tracking ride prices to unlock historical insights")
