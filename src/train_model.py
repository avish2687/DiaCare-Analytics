import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import preprocess_diabetes_data

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "models/readmission_model.pkl"

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("🧼 Preprocessing (training)...")
X, y, feature_names = preprocess_diabetes_data(df, training=True)

print("✂️ Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🤖 Training...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("💾 Saving model + features...")
joblib.dump(
    {
        "model": model,
        "feature_names": feature_names
    },
    MODEL_PATH
)

print("✅ Training complete")
