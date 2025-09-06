from src.train import train_and_save_model
from src.eval import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # ===== Load dataset =====
    df = pd.read_csv("data/synthetic_dataset.csv")

    # ===== Split features & labels =====
    X = df.drop("disease", axis=1)
    y = df["disease"]

    # ===== Encode labels =====
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ===== Train-test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ===== Train & Save Model =====
    categorical_features = ["age_band", "gender", "setting", "region", "season"]
    model = train_and_save_model(X_train, y_train, categorical_features)

    # ===== Evaluate Model =====
    evaluate_model(model, X_test, y_test, class_names=le.classes_)

    print("[INFO] Pipeline completed. Model ready for demo.")
