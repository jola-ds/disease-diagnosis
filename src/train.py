import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def build_preprocessor(categorical_features):
    return ColumnTransformer(
        transformers=[
    
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

def build_xgb_pipeline(categorical_features):
    preprocessor = build_preprocessor(categorical_features)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    return Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])

def train_and_save_model(X_train, y_train, categorical_features, model_path="models/final_model.pkl"):
    """Train XGBoost and save it."""
    pipeline = build_xgb_pipeline(categorical_features)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    print(f"[INFO] Model saved to {model_path}")
    return pipeline
