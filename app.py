import streamlit as st
import pandas as pd
import joblib
from src.preprocess import preprocess_diabetes_data

MODEL_PATH = "models/readmission_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def predict_from_dataframe(df: pd.DataFrame):
    X, _, _, _ = preprocess_diabetes_data(df, training=False)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs

def run_streamlit():
    st.title("Diabetes Readmission Predictor")
    st.write("Streamlit UI here")


    age = st.selectbox("Age Group", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)",
        "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ])
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
    num_medications = st.slider("Number of Medications", 1, 50, 10)

    input_df = pd.DataFrame([{
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_medications": num_medications
    }])

    if st.button("Predict Readmission"):
        pred, prob = predict_from_dataframe(input_df)
        st.success(f"Readmission Risk: {prob[0]*100:.2f}%")

if __name__ == "__main__":
    run_streamlit()
