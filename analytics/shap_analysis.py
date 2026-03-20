import shap
import joblib
import pandas as pd

MODEL_PATH = "models/readmission_model.pkl"
DATA_PATH = "data/diabetes.csv"

print("📦 Loading model...")
pipeline = joblib.load(MODEL_PATH)

print("📥 Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["readmitted"])

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(shap_values[1], X_transformed)
