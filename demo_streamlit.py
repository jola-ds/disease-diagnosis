#!/usr/bin/env python3
"""
Demo script to show how to run the Disease Diagnosis Streamlit app
"""

import subprocess
import sys
import os

def main():
    print("🏥 Disease Diagnosis AI - Streamlit App Demo")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/final_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Model not found. Please run the training pipeline first:")
        print("   python3 main.py")
        return
    
    # Check if dataset exists
    data_path = "data/synthetic_dataset.csv"
    if not os.path.exists(data_path):
        print("❌ Dataset not found. Please generate the synthetic dataset first.")
        return
    
    print("✅ Model and dataset found!")
    print("\n🚀 Starting Streamlit app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("\n" + "=" * 50)
    
    try:
        # Run the streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")

if __name__ == "__main__":
    main()
