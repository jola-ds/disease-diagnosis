#!/usr/bin/env python3
"""
Script to run the Disease Diagnosis API server
"""

import uvicorn
import os
import sys

def main():
    """Run the API server"""
    # Check if model exists
    model_path = "models/final_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please run 'python main.py' first to train the model.")
        sys.exit(1)
    
    print("🚀 Starting Disease Diagnosis API server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    main()
