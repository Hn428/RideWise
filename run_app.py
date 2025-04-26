"""
RideWise: Surge Price Predictor
Run this script to launch the Streamlit application.
"""

import os
import sys
import streamlit.web.cli as stcli
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the Streamlit app"""
    # Add the current directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the Streamlit app
    web_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "app.py")
    
    # Check if the app file exists
    if not os.path.exists(web_app_path):
        print(f"Error: Could not find the app file at {web_app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", web_app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 