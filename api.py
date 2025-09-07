"""
Disease Diagnosis API

A FastAPI application that serves the trained XGBoost model for disease prediction.
This API provides endpoints for predicting diseases based on patient symptoms and demographics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="Disease Diagnosis API",
    description="A machine learning API for predicting diseases from symptoms and patient demographics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
# Get allowed origins from environment variable or use default
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if allowed_origins == ["*"]:
    allowed_origins = ["*"]  # Keep as list for development
else:
    allowed_origins = [origin.strip() for origin in allowed_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables for model and class names
model = None
class_names = [
    "diabetes", "gastroenteritis", "healthy", "hiv", "hypertension",
    "malaria", "measles", "peptic_ulcer", "pneumonia", "tuberculosis", "typhoid"
]

# Pydantic models for request/response validation
class PatientInput(BaseModel):
    """Patient input data for disease prediction"""
    age_band: str = Field(..., description="Age band: '0-4', '5-14', '15-24', '25-44', '45-64', '65+'")
    gender: str = Field(..., description="Gender: 'male' or 'female'")
    setting: str = Field(..., description="Setting: 'urban' or 'rural'")
    region: str = Field(..., description="Region: 'north', 'south', 'east', 'west', 'middle_belt'")
    season: str = Field(..., description="Season: 'dry', 'rainy', or 'transition'")
    
    # General symptoms (binary: 0 or 1)
    fever: int = Field(..., ge=0, le=1, description="Fever (0=no, 1=yes)")
    headache: int = Field(..., ge=0, le=1, description="Headache (0=no, 1=yes)")
    cough: int = Field(..., ge=0, le=1, description="Cough (0=no, 1=yes)")
    chronic_cough: int = Field(..., ge=0, le=1, description="Chronic cough (0=no, 1=yes)")
    productive_cough: int = Field(..., ge=0, le=1, description="Productive cough (0=no, 1=yes)")
    fatigue: int = Field(..., ge=0, le=1, description="Fatigue (0=no, 1=yes)")
    body_ache: int = Field(..., ge=0, le=1, description="Body ache (0=no, 1=yes)")
    chills: int = Field(..., ge=0, le=1, description="Chills (0=no, 1=yes)")
    sweats: int = Field(..., ge=0, le=1, description="Sweats (0=no, 1=yes)")
    night_sweats: int = Field(..., ge=0, le=1, description="Night sweats (0=no, 1=yes)")
    weight_loss: int = Field(..., ge=0, le=1, description="Weight loss (0=no, 1=yes)")
    loss_of_appetite: int = Field(..., ge=0, le=1, description="Loss of appetite (0=no, 1=yes)")
    
    # GI symptoms
    nausea: int = Field(..., ge=0, le=1, description="Nausea (0=no, 1=yes)")
    vomiting: int = Field(..., ge=0, le=1, description="Vomiting (0=no, 1=yes)")
    diarrhea: int = Field(..., ge=0, le=1, description="Diarrhea (0=no, 1=yes)")
    constipation: int = Field(..., ge=0, le=1, description="Constipation (0=no, 1=yes)")
    abdominal_pain: int = Field(..., ge=0, le=1, description="Abdominal pain (0=no, 1=yes)")
    epigastric_pain: int = Field(..., ge=0, le=1, description="Epigastric pain (0=no, 1=yes)")
    heartburn: int = Field(..., ge=0, le=1, description="Heartburn (0=no, 1=yes)")
    hunger_pain: int = Field(..., ge=0, le=1, description="Hunger pain (0=no, 1=yes)")
    
    # Respiratory symptoms
    sore_throat: int = Field(..., ge=0, le=1, description="Sore throat (0=no, 1=yes)")
    runny_nose: int = Field(..., ge=0, le=1, description="Runny nose (0=no, 1=yes)")
    chest_pain: int = Field(..., ge=0, le=1, description="Chest pain (0=no, 1=yes)")
    shortness_of_breath: int = Field(..., ge=0, le=1, description="Shortness of breath (0=no, 1=yes)")
    rapid_breathing: int = Field(..., ge=0, le=1, description="Rapid breathing (0=no, 1=yes)")
    hemoptysis: int = Field(..., ge=0, le=1, description="Hemoptysis (0=no, 1=yes)")
    
    # Genitourinary symptoms
    dysuria: int = Field(..., ge=0, le=1, description="Dysuria (0=no, 1=yes)")
    polyuria: int = Field(..., ge=0, le=1, description="Polyuria (0=no, 1=yes)")
    oliguria: int = Field(..., ge=0, le=1, description="Oliguria (0=no, 1=yes)")
    
    # Metabolic symptoms
    polydipsia: int = Field(..., ge=0, le=1, description="Polydipsia (0=no, 1=yes)")
    polyphagia: int = Field(..., ge=0, le=1, description="Polyphagia (0=no, 1=yes)")
    blurred_vision: int = Field(..., ge=0, le=1, description="Blurred vision (0=no, 1=yes)")
    
    # Neurological symptoms
    dizziness: int = Field(..., ge=0, le=1, description="Dizziness (0=no, 1=yes)")
    confusion: int = Field(..., ge=0, le=1, description="Confusion (0=no, 1=yes)")
    
    # Dermatological/Physical signs
    rash: int = Field(..., ge=0, le=1, description="Rash (0=no, 1=yes)")
    maculopapular_rash: int = Field(..., ge=0, le=1, description="Maculopapular rash (0=no, 1=yes)")
    rose_spots: int = Field(..., ge=0, le=1, description="Rose spots (0=no, 1=yes)")
    conjunctivitis: int = Field(..., ge=0, le=1, description="Conjunctivitis (0=no, 1=yes)")
    lymph_nodes: int = Field(..., ge=0, le=1, description="Lymph nodes (0=no, 1=yes)")
    
    # Infection-related symptoms
    recurrent_infections: int = Field(..., ge=0, le=1, description="Recurrent infections (0=no, 1=yes)")
    oral_thrush: int = Field(..., ge=0, le=1, description="Oral thrush (0=no, 1=yes)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_band": "25-44",
                "gender": "female",
                "setting": "urban",
                "region": "north",
                "season": "dry",
                "fever": 1,
                "headache": 1,
                "cough": 0,
                "chronic_cough": 0,
                "productive_cough": 0,
                "fatigue": 1,
                "body_ache": 1,
                "chills": 1,
                "sweats": 0,
                "night_sweats": 0,
                "weight_loss": 0,
                "loss_of_appetite": 1,
                "nausea": 0,
                "vomiting": 0,
                "diarrhea": 0,
                "constipation": 0,
                "abdominal_pain": 0,
                "epigastric_pain": 0,
                "heartburn": 0,
                "hunger_pain": 0,
                "sore_throat": 0,
                "runny_nose": 0,
                "chest_pain": 0,
                "shortness_of_breath": 0,
                "rapid_breathing": 0,
                "hemoptysis": 0,
                "dysuria": 0,
                "polyuria": 0,
                "oliguria": 0,
                "polydipsia": 0,
                "polyphagia": 0,
                "blurred_vision": 0,
                "dizziness": 0,
                "confusion": 0,
                "rash": 0,
                "maculopapular_rash": 0,
                "rose_spots": 0,
                "conjunctivitis": 0,
                "lymph_nodes": 0,
                "recurrent_infections": 0,
                "oral_thrush": 0
            }
        }
    }

class PredictionResponse(BaseModel):
    """Response model for disease prediction"""
    predicted_disease: str = Field(..., description="The predicted disease")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all diseases")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    patients: List[PatientInput] = Field(..., description="List of patients for prediction")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_patients: int = Field(..., description="Total number of patients processed")

# Startup event to load the model
@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    model_path = "models/final_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    try:
        model = joblib.load(model_path)
        print(f"[INFO] Model loaded successfully from {model_path}")
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Disease Diagnosis API",
        "description": "A machine learning API for predicting diseases from symptoms and patient demographics",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(patient: PatientInput):
    """
    Predict disease for a single patient based on symptoms and demographics.
    
    This endpoint takes patient information including demographics and symptoms,
    and returns the most likely disease along with confidence scores.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        patient_data = patient.dict()
        df = pd.DataFrame([patient_data])
        
        # Make prediction
        probabilities = model.predict_proba(df)[0]
        predicted_index = np.argmax(probabilities)
        predicted_disease = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        
        # Create probabilities dictionary
        all_probabilities = {
            disease: float(prob) for disease, prob in zip(class_names, probabilities)
        }
        
        return PredictionResponse(
            predicted_disease=predicted_disease,
            confidence=confidence,
            all_probabilities=all_probabilities,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(patients: BatchPredictionRequest):
    """
    Predict diseases for multiple patients in a single request.
    
    This endpoint is useful for processing multiple patients efficiently.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        predictions = []
        
        for patient in patients.patients:
            # Convert input to DataFrame
            patient_data = patient.dict()
            df = pd.DataFrame([patient_data])
            
            # Make prediction
            probabilities = model.predict_proba(df)[0]
            predicted_index = np.argmax(probabilities)
            predicted_disease = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])
            
            # Create probabilities dictionary
            all_probabilities = {
                disease: float(prob) for disease, prob in zip(class_names, probabilities)
            }
            
            predictions.append(PredictionResponse(
                predicted_disease=predicted_disease,
                confidence=confidence,
                all_probabilities=all_probabilities,
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_patients=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Get available diseases endpoint
@app.get("/diseases")
async def get_diseases():
    """Get list of all possible diseases the model can predict"""
    return {
        "diseases": class_names,
        "total_diseases": len(class_names)
    }

# Get model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "classes": class_names,
        "total_classes": len(class_names),
        "features": [
            "age_band", "gender", "setting", "region", "season",
            "fever", "headache", "cough", "chronic_cough", "productive_cough", 
            "fatigue", "body_ache", "chills", "sweats", "night_sweats", 
            "weight_loss", "loss_of_appetite", "nausea", "vomiting", "diarrhea", 
            "constipation", "abdominal_pain", "epigastric_pain", "heartburn", 
            "hunger_pain", "sore_throat", "runny_nose", "chest_pain", 
            "shortness_of_breath", "rapid_breathing", "hemoptysis", "dysuria", 
            "polyuria", "oliguria", "polydipsia", "polyphagia", "blurred_vision", 
            "dizziness", "confusion", "rash", "maculopapular_rash", "rose_spots", 
            "conjunctivitis", "lymph_nodes", "recurrent_infections", "oral_thrush"
        ],
        "accuracy": "74.6% (on test set)",
        "description": "Trained on synthetic Nigerian hospital data for disease prediction"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
