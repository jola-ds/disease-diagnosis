import joblib
import pandas as pd
import numpy as np

def predict_single_sample(input_dict, model_path="models/final_model.pkl", class_names=None):
    """
    Load the trained model and predict for a single patient dictionary.
    
    Parameters
    ----------
    input_dict : dict
        Patient features. Keys must match the training feature names.
    model_path : str
        Path to the saved .pkl model.
    class_names : list of str, optional
        Human-readable class names. If None, class indices are returned.
    
    Returns
    -------
    dict
        {
            "predicted_disease": str or int,
            "confidence": float,
            "all_probabilities": dict[class_name -> probability]
        }
    """
    model = joblib.load(model_path)
    sample = pd.DataFrame([input_dict])
    probs = model.predict_proba(sample)[0]
    pred_index = np.argmax(probs)
    predicted_class = class_names[pred_index] if class_names is not None else pred_index
    confidence = probs[pred_index]
    return {
        "predicted_disease": predicted_class,
        "confidence": float(confidence),
        "all_probabilities": dict(zip(class_names if class_names is not None else range(len(probs)), probs))
    }

if __name__ == "__main__":
    # Example demo: toggle demographics and symptoms here
    sample_input = {
        "age_band": "21-40",
        "gender": "female",
        "setting": "urban",
        "region": "north",
        "season": "dry",
        "fever": 1,
        "headache": 1,
        "cough": 0,
        "fatigue": 1,
        "chills": 1,
        "body_ache": 1,
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

    result = predict_single_sample(
        sample_input,
        class_names=[
            "diabetes", "gastroenteritis", "healthy", "hiv", "hypertension",
            "malaria", "measles", "peptic_ulcer", "pneumonia", "tuberculosis", "typhoid"
        ]
    )

    print("Predicted disease:", result["predicted_disease"])
    print("Confidence:", f"{result['confidence']:.2%}")
