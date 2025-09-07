#!/usr/bin/env python3
"""
Test script for the Disease Diagnosis API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Root endpoint working: {data['message']}")
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_diseases_endpoint():
    """Test the diseases endpoint"""
    print("\n🦠 Testing diseases endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/diseases")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Diseases endpoint working: {len(data['diseases'])} diseases available")
        return True
    except Exception as e:
        print(f"❌ Diseases endpoint failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n📊 Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Model info endpoint working: {data['model_type']}")
        print(f"   Accuracy: {data['accuracy']}")
        return True
    except Exception as e:
        print(f"❌ Model info endpoint failed: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🔮 Testing single prediction endpoint...")
    
    # Sample patient data
    patient_data = {
        "age_band": "25-44",
        "gender": "female",
        "setting": "urban",
        "region": "north",
        "season": "dry",
        "fever": 1,
        "headache": 1,
        "cough": 0,
        "fatigue": 1,
        "body_ache": 1,
        "chills": 1,
        "sweats": 0,
        "nausea": 0,
        "vomiting": 0,
        "diarrhea": 0,
        "abdominal_pain": 0,
        "loss_of_appetite": 1,
        "sore_throat": 0,
        "runny_nose": 0,
        "dysuria": 0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=patient_data)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Single prediction working:")
        print(f"   Predicted disease: {data['predicted_disease']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print(f"   Top 3 probabilities:")
        sorted_probs = sorted(data['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for disease, prob in sorted_probs[:3]:
            print(f"     {disease}: {prob:.2%}")
        return True
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n📦 Testing batch prediction endpoint...")
    
    # Sample batch data
    batch_data = {
        "patients": [
            {
                "age_band": "25-44",
                "gender": "female",
                "setting": "urban",
                "region": "north",
                "season": "dry",
                "fever": 1,
                "headache": 1,
                "cough": 0,
                "fatigue": 1,
                "body_ache": 1,
                "chills": 1,
                "sweats": 0,
                "nausea": 0,
                "vomiting": 0,
                "diarrhea": 0,
                "abdominal_pain": 0,
                "loss_of_appetite": 1,
                "sore_throat": 0,
                "runny_nose": 0,
                "dysuria": 0
            },
            {
                "age_band": "5-14",
                "gender": "male",
                "setting": "rural",
                "region": "south",
                "season": "rainy",
                "fever": 0,
                "headache": 0,
                "cough": 0,
                "fatigue": 1,
                "body_ache": 0,
                "chills": 0,
                "sweats": 0,
                "nausea": 0,
                "vomiting": 0,
                "diarrhea": 0,
                "abdominal_pain": 0,
                "loss_of_appetite": 0,
                "sore_throat": 0,
                "runny_nose": 0,
                "dysuria": 0
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Batch prediction working:")
        print(f"   Total patients processed: {data['total_patients']}")
        for i, prediction in enumerate(data['predictions']):
            print(f"   Patient {i+1}: {prediction['predicted_disease']} ({prediction['confidence']:.2%})")
        return True
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Disease Diagnosis API Test Suite")
    print("=" * 50)
    
    # Wait a moment for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        test_health_check,
        test_root_endpoint,
        test_diseases_endpoint,
        test_model_info,
        test_single_prediction,
        test_batch_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
