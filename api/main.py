from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from src.preprocess import preprocess_diabetes_data

app = FastAPI()

# Allow the HTML file to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model bundle
bundle = joblib.load("models/readmission_model.pkl")
model  = bundle["model"]
encoder = bundle.get("encoder")

# Serve the HTML page at http://localhost:8000
@app.get("/")
def serve_frontend():
    return FileResponse("diabetes-readmission.html")

# Input schema
class PatientInput(BaseModel):
    age: str
    gender: str
    race: str
    time_in_hospital: int
    num_medications: int
    num_lab_procedures: int
    number_diagnoses: int
    number_inpatient: int
    insulin: str
    A1Cresult: str
    max_glu_serum: str
    diabetesMed: str
    change: str

# Prediction endpoint called by the HTML page
@app.post("/predict")
def predict(patient: PatientInput):
    df = pd.DataFrame([patient.dict()])
    X, _, _, _ = preprocess_diabetes_data(df, training=False, encoder=encoder)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    
    if prob < 0.15:
        risk_level = "Low"
    elif prob < 0.30:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_probability": round(prob, 4),
        "risk_level": risk_level,
        "prediction": pred
    }