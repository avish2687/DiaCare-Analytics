import pandas as pd
import joblib

MODEL_PATH = "models/readmission_model.pkl"

def predict_with_explanation(input_df: pd.DataFrame):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    return {
        "prediction": int(prediction[0]),
        "readmission_risk": float(probability[0][1])
    }
