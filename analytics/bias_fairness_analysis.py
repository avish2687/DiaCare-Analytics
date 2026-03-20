import pandas as pd
import joblib

from src.preprocess import preprocess_diabetes_data
from analytics.fairness_check import group_fairness_metrics

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "models/readmission_model.pkl"

def main():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("📦 Loading model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    print("🧼 Preprocessing (inference)...")
    X, _, _ = preprocess_diabetes_data(
        df,
        training=False,
        feature_names=feature_names
    )

    print("🤖 Predicting...")
    y_pred = model.predict(X)

    print("⚖️ Running fairness metrics...")
    results = group_fairness_metrics(
        df=df,
        y_pred=y_pred,
        sensitive_column="gender"
    )

    print("\n📊 Fairness Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
