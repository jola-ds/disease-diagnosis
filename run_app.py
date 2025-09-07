#!/usr/bin/env python3
"""
Script to run the Disease Diagnosis Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the app
    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()
